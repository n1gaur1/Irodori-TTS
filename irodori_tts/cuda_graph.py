"""CUDA Graph replay for the repeated DiT forward inside the samplers.

The RF sampler calls ``forward_with_encoded_conditions`` dozens of times per
request with identical shapes, and at batch size 1 the forward is bound by
kernel launch overhead rather than GPU work. ``CUDAGraphForward`` records the
forward once and replays it for the remaining steps. Replay runs the same
kernels on the same values, so the output matches eager execution bit for bit.

Inputs fall into two groups:

* Step inputs (``x_t``, ``t``, ``delta_t``) change every call. They are copied
  into graph-owned static buffers before each replay.
* Every other tensor (encoded conditions, masks, the context K/V cache) stays
  the same object for the whole request. The graph reads the caller's tensor
  directly, and the object identities are part of the graph key. Nothing is
  duplicated, in-place updates such as the speaker K/V scaling are visible to
  the next replay exactly as in eager mode, and the runner never writes into
  a caller-owned tensor. Two calls with equal shapes but different condition
  objects (CFG cond/uncond) simply use two graphs.

Graphs hold references to the request's tensors, so ``end_request()`` must be
called when sampling finishes; it releases every graph.

``persistent=True`` keeps graphs across requests instead, for callers that
replay the same shapes often (short interactive lines). Every tensor input is
then copied into graph-owned buffers before each replay, so only shapes select
a graph. Copies of the same input slot and shape are shared by all graphs, and
all graphs share one memory pool, which keeps the VRAM cost of holding many
lengths low. A persistent graph still reads module state directly, so
``keepalive`` must return any buffer the module may replace (such as the RoPE
cache), and ``fn`` must not bake input values into host-side decisions (such as
an attention plan built from mask contents). Keys whose capture failed stay
eager until ``clear()``, which is also required when the wrapped module
changes.

One instance must not be used from several threads at once. Other threads may
keep running CUDA work during a capture, except random ops on PyTorch's default
CUDA generator, which PyTorch rejects while any thread is capturing.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch

CUDA_GRAPH_MODES = ("auto", "on", "off")
_CUDA_GRAPH_ENV = "IRODORI_CUDA_GRAPH"
DEFAULT_STEP_INPUTS = ("x_t", "t", "delta_t")


def resolve_cuda_graph_mode(mode: str, *, device: torch.device, compile_model: bool) -> bool:
    """Decide whether sampler forwards should be replayed through CUDA Graphs.

    ``auto`` enables graphs on CUDA unless ``torch.compile`` is in use, and the
    ``IRODORI_CUDA_GRAPH`` environment variable (auto/on/off) overrides it.
    ``on`` fails loudly when graphs cannot be used instead of silently falling
    back.
    """
    requested = str(mode).strip().lower()
    if requested not in CUDA_GRAPH_MODES:
        raise ValueError(
            f"Unsupported cuda_graph={mode!r}. Expected one of: {', '.join(CUDA_GRAPH_MODES)}."
        )
    if requested == "auto":
        env_value = os.environ.get(_CUDA_GRAPH_ENV, "").strip().lower()
        if env_value in {"0", "false", "off", "no"}:
            requested = "off"
        elif env_value in {"1", "true", "on", "yes"}:
            requested = "on"
        elif env_value not in {"", "auto"}:
            raise ValueError(
                f"Unsupported {_CUDA_GRAPH_ENV}={env_value!r}. Expected one of: auto, on, off."
            )
    if requested == "off":
        return False
    supported = device.type == "cuda" and not compile_model
    if requested == "on" and not supported:
        raise ValueError(
            "cuda_graph='on' requires a CUDA model device and compile_model=False, "
            f"got device={device!s} compile_model={compile_model}."
        )
    return supported


def _consume_pending_capture_error(device: torch.device) -> None:
    """Clear the error a failed capture leaves behind.

    The capture error is not sticky, but the next kernel launch on the device
    reports it once, which would break the eager fallback. Launch a trivial
    kernel to absorb it; the second attempt runs on a clean state.
    """
    for _ in range(2):
        try:
            torch.zeros(1, device=device).add_(1)
            torch.cuda.synchronize(device)
            return
        except Exception:
            continue


def _shape_key(value: Any) -> Any:
    """Hashable description of shapes, dtypes and constants (not identities)."""
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_shape_key(item) for item in value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("const", value)
    raise TypeError(f"Unsupported CUDA graph input type: {type(value)!r}")


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _tensor_leaves(item)]
    return []


def _tree_bytes(value: Any) -> int:
    return sum(leaf.numel() * leaf.element_size() for leaf in _tensor_leaves(value))


def _replace_leaves(value: Any, leaves: Iterator[torch.Tensor]) -> Any:
    """``value`` with its tensors replaced, in ``_tensor_leaves`` order, by ``leaves``."""
    if isinstance(value, torch.Tensor):
        return next(leaves)
    if isinstance(value, list):
        return [_replace_leaves(item, leaves) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_leaves(item, leaves) for item in value)
    return value


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_steps: dict[str, torch.Tensor]
    static_output: torch.Tensor
    # References that keep the read-in-place inputs and module buffers alive
    # (their ids are part of the key, so they must not be recycled).
    held: tuple[object, ...]
    # Persistent mode only: graph-owned copies of every other tensor input,
    # refreshed from the caller's tensors before each replay.
    static_conditions: dict[str, tuple[torch.Tensor, ...]] = field(default_factory=dict)


@dataclass
class CUDAGraphStats:
    captures: int = 0
    replays: int = 0
    eager_calls: int = 0
    capture_failures: int = 0
    evictions: int = 0
    skipped_for_memory: int = 0
    step_copy_bytes: int = 0


class CUDAGraphForward:
    """Replay ``fn(**kwargs)`` through CUDA Graphs within one request (or across
    requests with ``persistent=True``).

    ``capture_after`` eager calls of a key run before it is captured, so
    one-off calls never pay for a capture. The capturing call returns the
    result of its side-stream warmup run. Keys whose capture fails stay eager
    until the request ends.
    """

    def __init__(
        self,
        fn: Callable[..., torch.Tensor],
        *,
        step_inputs: Iterable[str] = DEFAULT_STEP_INPUTS,
        capture_after: int = 1,
        max_entries: int = 4,
        memory_reserve_ratio: float = 0.15,
        keepalive: Callable[[], Iterable[object]] | None = None,
        persistent: bool = False,
    ) -> None:
        if capture_after < 0:
            raise ValueError(f"capture_after must be >= 0, got {capture_after}")
        if max_entries <= 0:
            raise ValueError(f"max_entries must be > 0, got {max_entries}")
        if not 0.0 <= memory_reserve_ratio < 1.0:
            raise ValueError(f"memory_reserve_ratio must be in [0, 1), got {memory_reserve_ratio}")
        self._fn = fn
        self._step_inputs = frozenset(step_inputs)
        self._capture_after = int(capture_after)
        self._max_entries = int(max_entries)
        self._memory_reserve_ratio = float(memory_reserve_ratio)
        self._keepalive = keepalive
        self._entries: OrderedDict[Any, _GraphEntry] = OrderedDict()
        self._eager_counts: dict[Any, int] = {}
        self._failed: set[Any] = set()
        # Keeps objects whose ids appear in _eager_counts/_failed alive.
        self._pinned: list[object] = []
        self._persistent = bool(persistent)
        # Persistent mode: condition copies shared by every graph with the same
        # input slot and shape, and one memory pool for all graph intermediates.
        self._shared_conditions: dict[Any, torch.Tensor] = {}
        self._pool: Any = None
        self.stats = CUDAGraphStats()

    @property
    def num_entries(self) -> int:
        return len(self._entries)

    @property
    def persistent(self) -> bool:
        return self._persistent

    @property
    def static_bytes(self) -> int:
        steps = sum(
            _tree_bytes(list(entry.static_steps.values())) for entry in self._entries.values()
        )
        return steps + _tree_bytes(list(self._shared_conditions.values()))

    def begin_request(self) -> None:
        """Start a sampling loop; per-request mode drops graphs from earlier requests."""
        self.end_request()

    def end_request(self) -> None:
        """Release every reference to the request's tensors.

        Per-request mode also releases every graph. Persistent graphs hold only
        their own copies of the inputs, so they are kept for later requests.
        """
        if self._persistent:
            return
        self._release_graphs()

    def clear(self) -> None:
        """Release every captured graph; required after the wrapped module changes."""
        self._release_graphs()
        self._shared_conditions.clear()
        self._pool = None

    def _release_graphs(self) -> None:
        self._entries.clear()
        self._eager_counts.clear()
        self._failed.clear()
        self._pinned.clear()

    def _key(self, kwargs: dict[str, Any]) -> Any:
        parts = []
        for name in sorted(kwargs):
            value = kwargs[name]
            # Persistent graphs copy every input, so only shapes select a graph.
            identity = (
                ()
                if self._persistent or name in self._step_inputs
                else tuple(id(leaf) for leaf in _tensor_leaves(value))
            )
            parts.append((name, _shape_key(value), identity))
        return tuple(parts)

    def __call__(self, **kwargs: Any) -> torch.Tensor:
        key = self._key(kwargs)
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            return self._replay(entry, kwargs)
        if key in self._failed:
            return self._eager(kwargs)
        seen = self._eager_counts.get(key, 0)
        if seen < self._capture_after:
            self._eager_counts[key] = seen + 1
            self._pin(kwargs)
            return self._eager(kwargs)
        return self._capture(key, kwargs)

    def _pin(self, kwargs: dict[str, Any]) -> None:
        if self._persistent:
            # Persistent keys hold no ids, so nothing has to outlive the request.
            return
        for name, value in kwargs.items():
            if name not in self._step_inputs:
                self._pinned.extend(_tensor_leaves(value))

    def _eager(self, kwargs: dict[str, Any]) -> torch.Tensor:
        self.stats.eager_calls += 1
        return self._fn(**kwargs)

    def _has_memory_for(self, working_bytes: int, device: torch.device) -> bool:
        free, total = torch.cuda.mem_get_info(device)
        # Blocks cached by this process's allocator are reusable but not free
        # from the driver's point of view.
        free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        # The graph-private pool holds the forward's intermediates; the input
        # size is a rough proxy for how large they get.
        return free - working_bytes >= int(total * self._memory_reserve_ratio)

    def _capture(self, key: Any, kwargs: dict[str, Any]) -> torch.Tensor:
        tensors = _tensor_leaves(list(kwargs.values()))
        device = tensors[0].device if tensors else torch.device("cpu")
        if device.type != "cuda":
            self._failed.add(key)
            self._pin(kwargs)
            return self._eager(kwargs)
        while len(self._entries) >= self._max_entries:
            self._entries.popitem(last=False)
            self.stats.evictions += 1
        if not self._has_memory_for(_tree_bytes(list(kwargs.values())), device):
            self.stats.skipped_for_memory += 1
            return self._eager(kwargs)

        static_steps: dict[str, torch.Tensor] = {}
        graph_kwargs = dict(kwargs)
        for name in self._step_inputs.intersection(kwargs):
            value = kwargs[name]
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Step input {name!r} must be a tensor or None.")
            static_steps[name] = value.detach().clone()
            graph_kwargs[name] = static_steps[name]
        static_conditions: dict[str, tuple[torch.Tensor, ...]] = {}
        if self._persistent:
            for name, value in kwargs.items():
                if name in self._step_inputs:
                    continue
                copies = tuple(
                    self._shared_condition(name, index, leaf)
                    for index, leaf in enumerate(_tensor_leaves(value))
                )
                static_conditions[name] = copies
                graph_kwargs[name] = _replace_leaves(value, iter(copies))
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            # Warm up on a side stream as torch.cuda.graph requires; this run's
            # output is the answer for the current call.
            warm_output = self._fn(**graph_kwargs)
        torch.cuda.current_stream(device).wait_stream(stream)
        if not isinstance(warm_output, torch.Tensor):
            self._failed.add(key)
            self._pin(kwargs)
            self.stats.capture_failures += 1
            return warm_output

        graph = torch.cuda.CUDAGraph()
        try:
            # thread_local: the default "global" mode makes CUDA calls from
            # other threads (e.g. a concurrent request's kernels and syncs)
            # fail while this thread captures.
            # Persistent graphs share one pool: replays never overlap and each
            # output is cloned before the next replay can reuse its memory.
            with torch.cuda.graph(
                graph, pool=self._pool, stream=stream, capture_error_mode="thread_local"
            ):
                static_output = self._fn(**graph_kwargs)
        except Exception:
            del graph
            self._failed.add(key)
            self._pin(kwargs)
            self.stats.capture_failures += 1
            _consume_pending_capture_error(device)
            return warm_output
        held: list[object] = (
            []
            if self._persistent
            else [
                leaf
                for name, value in kwargs.items()
                if name not in self._step_inputs
                for leaf in _tensor_leaves(value)
            ]
        )
        if self._keepalive is not None:
            held.extend(self._keepalive())
        self._entries[key] = _GraphEntry(
            graph=graph,
            static_steps=static_steps,
            static_output=static_output,
            held=tuple(held),
            static_conditions=static_conditions,
        )
        self.stats.captures += 1
        return warm_output

    def _shared_condition(self, name: str, index: int, leaf: torch.Tensor) -> torch.Tensor:
        """The persistent copy slot for one condition tensor, filled with ``leaf``."""
        slot = (name, index, tuple(leaf.shape), tuple(leaf.stride()), leaf.dtype, leaf.device)
        copy = self._shared_conditions.get(slot)
        if copy is None:
            copy = leaf.detach().clone()
            self._shared_conditions[slot] = copy
        else:
            copy.copy_(leaf)
        return copy

    def _replay(self, entry: _GraphEntry, kwargs: dict[str, Any]) -> torch.Tensor:
        for name, static in entry.static_steps.items():
            static.copy_(kwargs[name], non_blocking=True)
            self.stats.step_copy_bytes += static.numel() * static.element_size()
        for name, copies in entry.static_conditions.items():
            # Copied every replay, so in-place updates of a condition are seen.
            for copy, leaf in zip(copies, _tensor_leaves(kwargs[name]), strict=True):
                copy.copy_(leaf, non_blocking=True)
        entry.graph.replay()
        self.stats.replays += 1
        return entry.static_output.clone()

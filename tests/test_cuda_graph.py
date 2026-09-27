from __future__ import annotations

import pytest
import torch

from irodori_tts.cuda_graph import CUDAGraphForward, resolve_cuda_graph_mode

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


@pytest.mark.parametrize(
    ("mode", "env", "device", "compile_model", "expected"),
    [
        ("auto", "", "cuda", False, True),
        ("auto", "", "cpu", False, False),
        ("auto", "", "cuda", True, False),
        ("auto", "off", "cuda", False, False),
        ("auto", "0", "cuda", False, False),
        ("auto", "on", "cuda", False, True),
        ("on", "", "cuda", False, True),
        ("off", "", "cuda", False, False),
        ("off", "on", "cuda", False, False),
        ("OFF", "", "cuda", False, False),
    ],
)
def test_resolve_cuda_graph_mode(monkeypatch, mode, env, device, compile_model, expected):
    monkeypatch.setenv("IRODORI_CUDA_GRAPH", env)

    assert (
        resolve_cuda_graph_mode(mode, device=torch.device(device), compile_model=compile_model)
        is expected
    )


@pytest.mark.parametrize(("device", "compile_model"), [("cpu", False), ("cuda", True)])
def test_resolve_cuda_graph_mode_on_rejects_unsupported(monkeypatch, device, compile_model):
    monkeypatch.delenv("IRODORI_CUDA_GRAPH", raising=False)

    with pytest.raises(ValueError, match="cuda_graph='on'"):
        resolve_cuda_graph_mode("on", device=torch.device(device), compile_model=compile_model)


def test_resolve_cuda_graph_mode_env_on_rejects_cpu(monkeypatch):
    monkeypatch.setenv("IRODORI_CUDA_GRAPH", "on")

    with pytest.raises(ValueError, match="cuda_graph='on'"):
        resolve_cuda_graph_mode("auto", device=torch.device("cpu"), compile_model=False)


@pytest.mark.parametrize("mode", ["yes", "graph", ""])
def test_resolve_cuda_graph_mode_rejects_unknown_mode(monkeypatch, mode):
    monkeypatch.delenv("IRODORI_CUDA_GRAPH", raising=False)

    with pytest.raises(ValueError, match="Unsupported cuda_graph"):
        resolve_cuda_graph_mode(mode, device=torch.device("cuda"), compile_model=False)


def test_resolve_cuda_graph_mode_rejects_unknown_env(monkeypatch):
    monkeypatch.setenv("IRODORI_CUDA_GRAPH", "maybe")

    with pytest.raises(ValueError, match="IRODORI_CUDA_GRAPH"):
        resolve_cuda_graph_mode("auto", device=torch.device("cuda"), compile_model=False)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"capture_after": -1}, "capture_after"),
        ({"max_entries": 0}, "max_entries"),
        ({"memory_reserve_ratio": 1.0}, "memory_reserve_ratio"),
        ({"memory_reserve_ratio": -0.1}, "memory_reserve_ratio"),
    ],
)
def test_runner_rejects_invalid_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CUDAGraphForward(lambda **_: torch.zeros(1), **kwargs)


def test_runner_stays_eager_on_cpu_tensors():
    calls = []

    def fn(*, x_t):
        calls.append(x_t)
        return x_t * 2

    runner = CUDAGraphForward(fn, capture_after=0)
    x = torch.arange(4.0)
    for _ in range(3):
        assert torch.equal(runner(x_t=x), x * 2)

    assert len(calls) == 3
    assert runner.num_entries == 0
    assert runner.stats.captures == 0


def test_runner_rejects_unsupported_input_types():
    runner = CUDAGraphForward(lambda **_: torch.zeros(1))

    with pytest.raises(TypeError, match="Unsupported CUDA graph input type"):
        runner(x={"nested": torch.zeros(1)})


# --- CUDA ---------------------------------------------------------------


def _affine(*, x_t, cache, scale):
    # x_t is a step input; the nested cache mirrors context_kv_cache.
    out = x_t * scale
    for k, v in cache:
        out = out + (k * v).sum()
    return out


def _inputs(device, *, rows=4, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, 8, generator=gen).to(device)
    cache = [
        (torch.randn(3, generator=gen).to(device), torch.randn(3, generator=gen).to(device))
        for _ in range(2)
    ]
    return x, cache


@requires_cuda
def test_graph_replay_matches_eager_for_new_step_inputs():
    device = torch.device("cuda")
    runner = CUDAGraphForward(_affine, capture_after=1)
    x, cache = _inputs(device)
    runner.begin_request()

    for step in range(4):
        x_step = x + step  # a fresh tensor every step, like x_t
        expected = _affine(x_t=x_step, cache=cache, scale=1.5)
        assert torch.equal(runner(x_t=x_step, cache=cache, scale=1.5), expected)
    runner.end_request()

    assert runner.stats.eager_calls == 1
    assert runner.stats.captures == 1
    assert runner.stats.replays == 2
    assert runner.stats.step_copy_bytes == 2 * x.numel() * x.element_size()


@requires_cuda
def test_graph_output_is_not_overwritten_by_next_replay():
    device = torch.device("cuda")
    runner = CUDAGraphForward(_affine, capture_after=0)
    x, cache = _inputs(device)
    runner.begin_request()

    first = runner(x_t=x, cache=cache, scale=1.0)
    second = runner(x_t=x + 1, cache=cache, scale=1.0)
    third = runner(x_t=x + 2, cache=cache, scale=1.0)

    assert torch.equal(first, _affine(x_t=x, cache=cache, scale=1.0))
    assert torch.equal(second, _affine(x_t=x + 1, cache=cache, scale=1.0))
    assert torch.equal(third, _affine(x_t=x + 2, cache=cache, scale=1.0))


@requires_cuda
def test_in_place_update_of_condition_input_reaches_replay():
    """The speaker K/V scaling case: same object, new values, no copy needed."""
    device = torch.device("cuda")
    runner = CUDAGraphForward(_affine, capture_after=0)
    x, cache = _inputs(device)
    runner.begin_request()
    runner(x_t=x, cache=cache, scale=1.0)  # capture
    runner(x_t=x + 1, cache=cache, scale=1.0)  # replay

    for layer in cache:
        layer[0].mul_(3.0)
        layer[1].mul_(0.5)
    fresh = runner(x_t=x + 2, cache=cache, scale=1.0)

    assert torch.equal(fresh, _affine(x_t=x + 2, cache=cache, scale=1.0))
    assert runner.stats.replays == 2


@requires_cuda
def test_same_shape_different_condition_objects_use_separate_graphs():
    """CFG cond/uncond: equal shapes, different tensors. Never write into either."""
    device = torch.device("cuda")
    runner = CUDAGraphForward(_affine, capture_after=0)
    x, cond = _inputs(device, seed=0)
    _, uncond = _inputs(device, seed=1)
    cond_before = [tuple(t.clone() for t in layer) for layer in cond]
    uncond_before = [tuple(t.clone() for t in layer) for layer in uncond]
    runner.begin_request()

    for step in range(3):
        assert torch.equal(
            runner(x_t=x + step, cache=cond, scale=1.0),
            _affine(x_t=x + step, cache=cond, scale=1.0),
        )
        assert torch.equal(
            runner(x_t=x + step, cache=uncond, scale=1.0),
            _affine(x_t=x + step, cache=uncond, scale=1.0),
        )

    assert runner.stats.captures == 2
    assert runner.stats.replays == 4
    for layer, before in zip(cond + uncond, cond_before + uncond_before, strict=True):
        for tensor, original in zip(layer, before, strict=True):
            assert torch.equal(tensor, original)


@requires_cuda
def test_end_request_releases_graphs_and_references():
    device = torch.device("cuda")
    runner = CUDAGraphForward(_affine, capture_after=0)
    x, cache = _inputs(device)
    runner.begin_request()
    runner(x_t=x, cache=cache, scale=1.0)
    assert runner.num_entries == 1

    runner.end_request()

    assert runner.num_entries == 0
    assert runner.static_bytes == 0
    assert runner._pinned == []


@requires_cuda
def test_next_request_recaptures_for_new_condition_objects():
    device = torch.device("cuda")
    runner = CUDAGraphForward(_affine, capture_after=0)
    x, cache_a = _inputs(device, seed=0)
    _, cache_b = _inputs(device, seed=1)

    runner.begin_request()
    runner(x_t=x, cache=cache_a, scale=1.0)
    runner.end_request()
    runner.begin_request()
    out = runner(x_t=x, cache=cache_b, scale=1.0)
    again = runner(x_t=x + 1, cache=cache_b, scale=1.0)
    runner.end_request()

    assert runner.stats.captures == 2
    assert torch.equal(out, _affine(x_t=x, cache=cache_b, scale=1.0))
    assert torch.equal(again, _affine(x_t=x + 1, cache=cache_b, scale=1.0))


@requires_cuda
def test_short_long_short_within_a_request():
    device = torch.device("cuda")
    runner = CUDAGraphForward(_affine, capture_after=0, max_entries=2)
    short, cache = _inputs(device, rows=2)
    long, _ = _inputs(device, rows=16)
    runner.begin_request()

    values = (short, long, short, long)
    outs = [runner(x_t=value, cache=cache, scale=2.0) for value in values]

    for value, out in zip(values, outs, strict=True):
        assert torch.equal(out, _affine(x_t=value, cache=cache, scale=2.0))
    assert runner.stats.captures == 2
    assert runner.stats.replays == 2


@requires_cuda
def test_lru_eviction_bounds_entries():
    device = torch.device("cuda")
    runner = CUDAGraphForward(_affine, capture_after=0, max_entries=2)
    _, cache = _inputs(device)
    runner.begin_request()

    for rows in (1, 2, 3, 1):
        x, _ = _inputs(device, rows=rows)
        assert torch.equal(
            runner(x_t=x, cache=cache, scale=1.0), _affine(x_t=x, cache=cache, scale=1.0)
        )

    assert runner.num_entries == 2
    assert runner.stats.evictions == 2
    assert runner.stats.captures == 4  # rows=1 was evicted before it came back


@requires_cuda
def test_constant_arguments_are_part_of_the_key():
    device = torch.device("cuda")
    runner = CUDAGraphForward(_affine, capture_after=0)
    x, cache = _inputs(device)
    runner.begin_request()

    assert torch.equal(
        runner(x_t=x, cache=cache, scale=1.0), _affine(x_t=x, cache=cache, scale=1.0)
    )
    assert torch.equal(
        runner(x_t=x, cache=cache, scale=3.0), _affine(x_t=x, cache=cache, scale=3.0)
    )

    assert runner.stats.captures == 2


@requires_cuda
def test_none_step_input_is_supported():
    device = torch.device("cuda")

    def fn(*, x_t, delta_t, cache):
        return _affine(x_t=x_t, cache=cache, scale=1.0) if delta_t is None else x_t

    runner = CUDAGraphForward(fn, capture_after=0)
    x, cache = _inputs(device)
    runner.begin_request()

    for step in range(3):
        assert torch.equal(
            runner(x_t=x + step, delta_t=None, cache=cache),
            _affine(x_t=x + step, cache=cache, scale=1.0),
        )
    assert runner.stats.replays == 2


@requires_cuda
def test_capture_failure_falls_back_to_eager():
    device = torch.device("cuda")

    def host_copy_inside(*, x_t):
        # A host-to-device copy is not allowed while a stream is capturing.
        return x_t + torch.tensor(1.0, device=x_t.device)

    runner = CUDAGraphForward(host_copy_inside, capture_after=0)
    x = torch.ones(4, device=device)
    runner.begin_request()

    outs = [runner(x_t=x) for _ in range(3)]

    assert all(torch.equal(out, x + 1) for out in outs)
    assert runner.stats.capture_failures == 1
    assert runner.num_entries == 0
    # CUDA keeps working normally after the failed capture, graphs included.
    assert torch.equal(x * 2, torch.full((4,), 2.0, device=device))
    healthy = CUDAGraphForward(_affine, capture_after=0)
    y, cache = _inputs(device)
    healthy.begin_request()
    for step in range(3):
        assert torch.equal(
            healthy(x_t=y + step, cache=cache, scale=1.0),
            _affine(x_t=y + step, cache=cache, scale=1.0),
        )
    assert healthy.stats.replays == 2


@requires_cuda
def test_capture_does_not_break_cuda_work_in_other_threads():
    import threading

    device = torch.device("cuda")
    weight = torch.randn(64, 64, device=device) / 8

    def chain(*, x_t, weight):
        y = x_t
        for _ in range(50):
            y = torch.tanh(y @ weight)
        return y

    stop = threading.Event()
    other_errors: list[str] = []

    def other_request():
        # Kernels, a sync and its own generator, like a concurrent request.
        while not stop.is_set():
            try:
                gen = torch.Generator(device=device).manual_seed(0)
                torch.randn(32, device=device, generator=gen).sum().item()
            except Exception as exc:
                other_errors.append(str(exc))

    worker = threading.Thread(target=other_request, daemon=True)
    worker.start()
    runner = CUDAGraphForward(chain, capture_after=0)
    try:
        for index in range(10):
            runner.begin_request()
            x = torch.full((4, 64), (index + 1) / 10, device=device)
            for step in range(3):
                assert torch.equal(
                    runner(x_t=x + step, weight=weight), chain(x_t=x + step, weight=weight)
                )
            runner.end_request()
    finally:
        stop.set()
        worker.join(timeout=5)

    assert runner.stats.capture_failures == 0
    assert runner.stats.captures == 10
    assert other_errors == []


@requires_cuda
def test_insufficient_memory_skips_capture(monkeypatch):
    device = torch.device("cuda")
    runner = CUDAGraphForward(_affine, capture_after=0)
    x, cache = _inputs(device)
    monkeypatch.setattr(runner, "_has_memory_for", lambda *_: False)
    runner.begin_request()

    out = runner(x_t=x, cache=cache, scale=1.0)

    assert torch.equal(out, _affine(x_t=x, cache=cache, scale=1.0))
    assert runner.stats.skipped_for_memory == 1
    assert runner.num_entries == 0


@requires_cuda
def test_clear_releases_entries():
    device = torch.device("cuda")
    runner = CUDAGraphForward(_affine, capture_after=0)
    x, cache = _inputs(device)
    runner.begin_request()
    runner(x_t=x, cache=cache, scale=1.0)
    assert runner.num_entries == 1

    runner.clear()

    assert runner.num_entries == 0


@requires_cuda
def test_keepalive_holds_replaced_buffer_storage():
    device = torch.device("cuda")
    holder = {"table": torch.arange(4.0, device=device)}

    def lookup(*, x_t):
        return x_t + holder["table"]

    runner = CUDAGraphForward(lookup, capture_after=0, keepalive=lambda: (holder["table"],))
    x = torch.zeros(4, device=device)
    runner.begin_request()
    runner(x_t=x)
    # Drop the only outside reference first, then allocate the same size: the
    # caching allocator would hand the freed block straight back if nothing
    # kept it alive (like a RoPE cache being replaced).
    holder["table"] = None
    holder["table"] = torch.full((4,), 100.0, device=device)

    out = runner(x_t=x + 1)

    assert torch.equal(out, torch.arange(4.0, device=device) + 1)

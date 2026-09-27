from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest
import soundfile as sf
import torch

from irodori_tts.config import ModelConfig
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    ReferenceLatentCache,
    SamplingRequest,
    find_flattening_point,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def _sequential_flattening_point(
    latent, target_value=0.0, window_size=20, std_threshold=0.05, mean_threshold=0.1
):
    """The original window-by-window scan, kept as the reference behavior."""
    total_steps = int(latent.shape[0])
    if total_steps <= 0 or window_size <= 0:
        return total_steps
    pad = torch.zeros((window_size, latent.shape[1]), device=latent.device, dtype=latent.dtype)
    padded = torch.cat([latent, pad], dim=0)
    for i in range(padded.shape[0] - window_size):
        window = padded[i : i + window_size]
        window_std = window.std(unbiased=False)
        window_mean = window.mean()
        if window_std < std_threshold and torch.abs(window_mean - target_value) < mean_threshold:
            return int(i)
    return total_steps


def _latent_cases(device, dtype):
    gen = torch.Generator(device="cpu").manual_seed(0)
    cases = []
    for length in (1, 5, 19, 20, 21, 64, 257):
        for tail in ("none", "zeros", "quiet", "noisy"):
            latent = torch.randn(length, 8, generator=gen)
            cut = int(length * 0.6)
            if tail == "zeros":
                latent[cut:] = 0
            elif tail == "quiet":
                latent[cut:] = torch.randn(length - cut, 8, generator=gen) * 0.04
            elif tail == "noisy":
                latent[cut:] = torch.randn(length - cut, 8, generator=gen) * 0.3
            cases.append(latent.to(device=device, dtype=dtype))
    # Windows sitting right at the std threshold: the screening must not drop them.
    edge = torch.zeros(80, 8)
    edge[:, ::2] = 0.05
    edge[:, 1::2] = -0.05
    cases.append(edge.to(device=device, dtype=dtype))
    cases.append((edge * (1 - 1e-6)).to(device=device, dtype=dtype))
    # Windows just above the threshold (inside the screening margin) before a
    # truly flat tail: the screen admits them and the exact check must reject.
    near = torch.zeros(90, 8)
    near[:60, ::2] = 0.051
    near[:60, 1::2] = -0.051
    cases.append(near.to(device=device, dtype=dtype))
    cases.append(torch.full((40, 8), float("nan")).to(device=device, dtype=dtype))
    return cases


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("window_size", "std_threshold", "mean_threshold", "target_value"),
    [
        (20, 0.05, 0.1, 0.0),
        (1, 0.05, 0.1, 0.0),
        (7, 0.2, 0.02, 0.0),
        (20, 0.0, 0.1, 0.0),
        (5, 0.1, 0.1, 0.5),
    ],
)
def test_find_flattening_point_matches_sequential_scan(
    device, dtype, window_size, std_threshold, mean_threshold, target_value
):
    for latent in _latent_cases(device, dtype):
        kwargs = {
            "window_size": window_size,
            "std_threshold": std_threshold,
            "mean_threshold": mean_threshold,
            "target_value": target_value,
        }
        assert find_flattening_point(latent, **kwargs) == _sequential_flattening_point(
            latent, **kwargs
        )


def test_find_flattening_point_edge_arguments():
    latent = torch.zeros(10, 4)

    assert find_flattening_point(latent, window_size=0) == 10
    assert find_flattening_point(torch.zeros(0, 4)) == 0
    with pytest.raises(ValueError, match="Expected latent shape"):
        find_flattening_point(torch.zeros(10))


# --- reference latent cache --------------------------------------------------


def _latent(steps: int, dim: int = 4) -> torch.Tensor:
    return torch.zeros(1, steps, dim)


def test_reference_cache_lru_by_entries():
    cache = ReferenceLatentCache(max_entries=2, max_bytes=1 << 20)
    for name in ("a", "b", "c"):
        cache.put((name,), _latent(3), None)

    assert cache.get(("a",)) is None
    assert cache.get(("b",)) is not None
    assert cache.get(("c",)) is not None
    assert len(cache) == 2
    assert cache.evictions == 1


def test_reference_cache_lru_by_bytes():
    one = _latent(8)  # 8 * 4 * 4 bytes = 128
    cache = ReferenceLatentCache(max_entries=10, max_bytes=300)
    cache.put(("a",), one, None)
    cache.put(("b",), one, None)
    assert cache.num_bytes == 256
    cache.get(("a",))  # a becomes most recent
    cache.put(("c",), one, None)

    assert cache.get(("b",)) is None
    assert cache.get(("a",)) is not None
    assert cache.num_bytes == 256


def test_reference_cache_skips_entry_larger_than_budget():
    cache = ReferenceLatentCache(max_entries=10, max_bytes=100)
    cache.put(("a",), _latent(8), None)

    assert len(cache) == 0


def test_reference_cache_replacing_a_key_keeps_byte_count():
    cache = ReferenceLatentCache(max_entries=10, max_bytes=1 << 20)
    cache.put(("a",), _latent(8), None)
    cache.put(("a",), _latent(2), None)

    assert len(cache) == 1
    assert cache.num_bytes == 2 * 4 * 4


@pytest.mark.parametrize(("entries", "size"), [(0, 1 << 20), (4, 0)])
def test_reference_cache_disabled(entries, size):
    cache = ReferenceLatentCache(max_entries=entries, max_bytes=size)
    cache.put(("a",), _latent(1), None)

    assert not cache.enabled
    assert len(cache) == 0


@pytest.mark.parametrize(("entries", "size"), [(-1, 10), (1, -1)])
def test_reference_cache_rejects_negative_limits(entries, size):
    with pytest.raises(ValueError):
        ReferenceLatentCache(max_entries=entries, max_bytes=size)


def test_file_fingerprint_changes_with_content(tmp_path):
    path = tmp_path / "ref.bin"
    path.write_bytes(b"abc")
    first = ReferenceLatentCache.file_fingerprint(path)
    assert ReferenceLatentCache.file_unchanged(first)

    stat = os.stat(path)
    path.write_bytes(b"abd")  # same size
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # same mtime too
    second = ReferenceLatentCache.file_fingerprint(path)

    assert first[:3] == second[:3]
    assert first[3] != second[3]


def test_file_unchanged_detects_rewrite_and_removal(tmp_path):
    path = tmp_path / "ref.bin"
    path.write_bytes(b"abc")
    fingerprint = ReferenceLatentCache.file_fingerprint(path)

    time.sleep(0.01)
    path.write_bytes(b"abcdef")
    assert not ReferenceLatentCache.file_unchanged(fingerprint)
    path.unlink()
    assert not ReferenceLatentCache.file_unchanged(fingerprint)


class _FakeCodec:
    sample_rate = 100
    model = SimpleNamespace(hop_length=10)

    def __init__(self, *, deterministic_encode: bool = True) -> None:
        self.deterministic_encode = deterministic_encode
        self.calls = 0

    def encode_waveform(self, wav, *, sample_rate, normalize_db, ensure_max):
        self.calls += 1
        # (1, T/hop, latent_dim): deterministic function of the waveform.
        frames = wav.shape[-1] // 10
        base = wav[..., : frames * 10].reshape(1, frames, 10).mean(dim=-1, keepdim=True)
        offset = 0.0 if normalize_db is None else float(normalize_db)
        return (base + offset).expand(1, frames, 4).clone()


def _runtime(codec, *, max_entries=8, default_max_ref_seconds=30.0):
    runtime = InferenceRuntime.__new__(InferenceRuntime)
    runtime.model = torch.nn.Linear(1, 1)
    runtime.model_cfg = ModelConfig(latent_dim=4, use_speaker_condition=True)
    runtime.codec = codec
    runtime.model_device = torch.device("cpu")
    runtime.default_max_ref_seconds = default_max_ref_seconds
    runtime.reference_cache = ReferenceLatentCache(max_entries=max_entries, max_bytes=1 << 20)
    return runtime


def _write_wav(path, seconds=1.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    data = (torch.rand(int(100 * seconds), generator=gen) - 0.5).numpy()
    sf.write(str(path), data, 100)
    return str(path)


def _load(runtime, **kwargs):
    messages: list[str] = []
    latent, mask = runtime._load_reference_latent(
        req=SamplingRequest(text="x", **kwargs), batch_size=1, messages=messages
    )
    return latent, mask, messages


def test_reference_cache_hit_returns_identical_latent(tmp_path):
    codec = _FakeCodec()
    runtime = _runtime(codec)
    path = _write_wav(tmp_path / "a.wav")

    first, first_mask, first_messages = _load(runtime, ref_wav=path)
    second, second_mask, second_messages = _load(runtime, ref_wav=path)

    assert codec.calls == 1
    assert torch.equal(first, second)
    assert torch.equal(first_mask, second_mask)
    assert second_messages == first_messages + ["info: reused 1/1 cached reference latent(s)."]


def test_reference_cache_hit_does_not_alias_cached_tensor(tmp_path):
    runtime = _runtime(_FakeCodec())
    path = _write_wav(tmp_path / "a.wav")
    first, _, _ = _load(runtime, ref_wav=path)
    first.add_(100.0)

    second, _, _ = _load(runtime, ref_wav=path)

    assert not torch.equal(first, second)


def test_reference_cache_misses_when_file_content_changes(tmp_path):
    codec = _FakeCodec()
    runtime = _runtime(codec)
    path = tmp_path / "a.wav"
    _write_wav(path, seed=0)
    first, _, _ = _load(runtime, ref_wav=str(path))

    _write_wav(path, seed=1)
    second, _, _ = _load(runtime, ref_wav=str(path))

    assert codec.calls == 2
    assert not torch.equal(first, second)


@pytest.mark.parametrize(
    "changed",
    [
        {"ref_normalize_db": -20.0},
        {"ref_normalize_db": None},
        {"ref_ensure_max": False},
        {"max_ref_seconds": 0.5},
    ],
)
def test_reference_cache_key_includes_encode_settings(tmp_path, changed):
    codec = _FakeCodec()
    runtime = _runtime(codec)
    path = _write_wav(tmp_path / "a.wav")
    _load(runtime, ref_wav=path)

    _load(runtime, ref_wav=path, **changed)

    assert codec.calls == 2


def test_reference_cache_bypassed_for_stochastic_encode(tmp_path):
    codec = _FakeCodec(deterministic_encode=False)
    runtime = _runtime(codec)
    path = _write_wav(tmp_path / "a.wav")

    _load(runtime, ref_wav=path)
    _load(runtime, ref_wav=path)

    assert codec.calls == 2
    assert len(runtime.reference_cache) == 0


def test_reference_cache_disabled_by_zero_entries(tmp_path):
    codec = _FakeCodec()
    runtime = _runtime(codec, max_entries=0)
    path = _write_wav(tmp_path / "a.wav")

    _load(runtime, ref_wav=path)
    _load(runtime, ref_wav=path)

    assert codec.calls == 2


def test_reference_cache_replays_trim_warning(tmp_path):
    codec = _FakeCodec()
    runtime = _runtime(codec)
    path = _write_wav(tmp_path / "long.wav", seconds=3.0)

    first, _, first_messages = _load(runtime, ref_wav=path, max_ref_seconds=1.0)
    second, _, second_messages = _load(runtime, ref_wav=path, max_ref_seconds=1.0)

    assert codec.calls == 1
    assert torch.equal(first, second)
    assert any("Trimming from 3.00s to 1.00s" in m for m in first_messages)
    assert second_messages[:-1] == first_messages


def test_reference_cache_with_multiple_wavs(tmp_path):
    codec = _FakeCodec()
    runtime = _runtime(codec)
    paths = [_write_wav(tmp_path / f"{i}.wav", seed=i) for i in range(2)]

    first, _, _ = _load(runtime, ref_wavs=paths)
    second, _, messages = _load(runtime, ref_wavs=paths)

    assert codec.calls == 2
    assert torch.equal(first, second)
    assert "info: reused 2/2 cached reference latent(s)." in messages


def test_missing_reference_raises_the_loader_error_with_cache_enabled(tmp_path):
    missing = str(tmp_path / "missing.wav")
    errors = []
    for max_entries in (0, 8):  # 0 is the original, uncached path
        runtime = _runtime(_FakeCodec(), max_entries=max_entries)
        with pytest.raises(Exception) as info:
            _load(runtime, ref_wav=missing)
        errors.append(type(info.value))

    assert errors[0] is errors[1]


def test_loading_lora_clears_cuda_graphs(tmp_path, monkeypatch):
    import irodori_tts.inference_runtime as runtime_module

    cleared = []
    runtime = InferenceRuntime.__new__(InferenceRuntime)
    runtime.key = SimpleNamespace(compile_model=False)
    runtime.model = torch.nn.Linear(1, 1)
    runtime.model_device = torch.device("cpu")
    runtime._model_dtype = torch.float32
    runtime._lora_adapter_names = {}
    runtime._graph_forward = SimpleNamespace(clear=lambda: cleared.append(True))
    monkeypatch.setattr(runtime_module, "is_lora_adapter_dir", lambda _path: True)
    monkeypatch.setattr(runtime_module, "load_lora_adapter", lambda model, *_a, **_k: model)

    runtime._prepare_lora_for_request_inner(str(tmp_path), messages=[], log_fn=lambda _m: None)

    assert cleared == [True]
    # Any loaded adapter keeps sampling eager from now on (see synthesize()).
    assert runtime._lora_adapter_names


def test_reference_cache_not_stored_when_file_changes_during_read(tmp_path, monkeypatch):
    codec = _FakeCodec()
    runtime = _runtime(codec)
    path = _write_wav(tmp_path / "a.wav")
    monkeypatch.setattr(ReferenceLatentCache, "file_unchanged", staticmethod(lambda _: False))

    _load(runtime, ref_wav=path)

    assert len(runtime.reference_cache) == 0

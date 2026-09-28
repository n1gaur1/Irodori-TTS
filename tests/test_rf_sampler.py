from __future__ import annotations

import pytest
import torch

from irodori_tts.model import _TIMESTEP_FREQS_CACHE, get_timestep_embedding
from irodori_tts.rf import sample_euler_rf_cfg
from tests.tiny_model import build_tiny_model, tiny_inputs

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def _sample(model, inputs, **kwargs):
    params = {
        "sequence_length": 7,
        "num_steps": 8,
        "seed": 3,
        "cfg_scale_text": 2.0,
        "cfg_scale_caption": 1.5,
        "cfg_scale_speaker": 2.5,
        "cfg_guidance_mode": "independent",
        **kwargs,
    }
    return sample_euler_rf_cfg(model=model, **inputs, **params)


def test_timestep_embedding_matches_uncached_formula():
    timestep = torch.rand(5)
    dim = 16
    half = dim // 2
    freqs = 1000.0 * torch.exp(
        -torch.log(torch.tensor(10000.0, dtype=torch.float32))
        * torch.arange(half, dtype=torch.float32)
        / half
    )
    args = timestep[:, None].float() * freqs[None, :]
    expected = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    assert torch.equal(get_timestep_embedding(timestep, dim), expected)
    assert torch.equal(get_timestep_embedding(timestep, dim), expected)  # cached path
    assert (torch.device("cpu"), half) in _TIMESTEP_FREQS_CACHE


def test_timestep_cache_filled_in_inference_mode_still_supports_backward():
    _TIMESTEP_FREQS_CACHE.clear()
    with torch.inference_mode():
        get_timestep_embedding(torch.rand(2), 16)

    timestep = torch.rand(3, requires_grad=True)
    get_timestep_embedding(timestep, 16).sum().backward()

    assert timestep.grad is not None


@pytest.mark.parametrize("mode", ["independent", "joint", "alternating"])
def test_reused_conditions_match_fresh_encoding(tiny_model, mode):
    inputs = tiny_inputs()
    scales = (
        {"cfg_scale_text": 2.0, "cfg_scale_caption": 2.0, "cfg_scale_speaker": 2.0}
        if mode == "joint"
        else {}
    )
    with torch.inference_mode():
        encoded = tiny_model.encode_conditions(**inputs)
        fresh = _sample(tiny_model, inputs, cfg_guidance_mode=mode, **scales)
        reused = _sample(
            tiny_model, inputs, cfg_guidance_mode=mode, encoded_conditions=encoded, **scales
        )
        # The argument is really used: different conditions, different output.
        other = _sample(
            tiny_model,
            inputs,
            cfg_guidance_mode=mode,
            encoded_conditions=(encoded[0] * 2.0, *encoded[1:]),
            **scales,
        )

    assert torch.equal(fresh, reused)
    assert not torch.equal(fresh, other)


def test_meanflow_reused_conditions_match_fresh_encoding(tiny_model):
    from irodori_tts.meanflow import sample_euler_meanflow

    tiny_model.enable_meanflow_parameterization()
    with torch.no_grad():
        for param in tiny_model.delta_cond_module.parameters():
            param.normal_(0.0, 0.2)  # the zero init would hide the delta path
    inputs = tiny_inputs()

    with torch.inference_mode():
        encoded = tiny_model.encode_conditions(**inputs)
        fresh = sample_euler_meanflow(model=tiny_model, sequence_length=7, seed=5, **inputs)
        reused = sample_euler_meanflow(
            model=tiny_model, sequence_length=7, seed=5, encoded_conditions=encoded, **inputs
        )
        # The argument is really used: different conditions, different output.
        other = sample_euler_meanflow(
            model=tiny_model,
            sequence_length=7,
            seed=5,
            encoded_conditions=(encoded[0] * 2.0, *encoded[1:]),
            **inputs,
        )

    assert torch.isfinite(fresh).all()
    assert torch.equal(fresh, reused)
    assert not torch.equal(fresh, other)


def test_forward_fn_replaces_model_forward(tiny_model):
    inputs = tiny_inputs()
    calls = []

    def spy(**kwargs):
        calls.append(kwargs["x_t"].shape[0])
        return tiny_model.forward_with_encoded_conditions(**kwargs)

    with torch.inference_mode():
        default = _sample(tiny_model, inputs, cfg_min_t=0.5, cfg_max_t=1.0)
        routed = _sample(tiny_model, inputs, cfg_min_t=0.5, cfg_max_t=1.0, forward_fn=spy)

    assert torch.equal(default, routed)
    # 8 steps: t = 0.999 * (1 - i/8); the first four are within [0.5, 1.0] and
    # run cond + text/speaker/caption uncond in one batch.
    assert calls == [4, 4, 4, 4, 1, 1, 1, 1]


def test_speaker_kv_restore_happens_at_exact_float32_threshold(tiny_model):
    inputs = tiny_inputs()
    num_steps = 8
    t_values = ((1.0 - torch.linspace(0.0, 1.0, num_steps + 1)) * 0.999).tolist()
    # A threshold equal to a schedule value: the restore must happen right
    # after that step (t >= min_t and t_next < min_t), as the tensor compare did.
    boundary_step = 3
    min_t = t_values[boundary_step]
    seen = []

    def spy(**kwargs):
        # Speaker K of layer 0; scaled by 2 while the boost is active.
        seen.append(kwargs["context_kv_cache"][0][2].abs().sum().item())
        return tiny_model.forward_with_encoded_conditions(**kwargs)

    with torch.inference_mode():
        _sample(
            tiny_model,
            inputs,
            num_steps=num_steps,
            cfg_scale_text=0.0,
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
            speaker_kv_scale=2.0,
            speaker_kv_min_t=min_t,
            forward_fn=spy,
        )

    boosted = seen[: boundary_step + 1]
    restored = seen[boundary_step + 1 :]
    assert all(value == pytest.approx(boosted[0]) for value in boosted)
    assert all(value == pytest.approx(boosted[0] / 2.0, rel=1e-5) for value in restored)


def test_sway_schedule_that_is_not_decreasing_is_rejected(tiny_model):
    inputs = tiny_inputs()

    with (
        torch.inference_mode(),
        pytest.raises(ValueError, match="strictly decreasing"),
    ):
        _sample(tiny_model, inputs, t_schedule_mode="sway", sway_coeff=-5.0)


def test_rescale_uses_same_schedule_value(tiny_model):
    inputs = tiny_inputs()

    with torch.inference_mode():
        a = _sample(tiny_model, inputs, rescale_k=1.2, rescale_sigma=0.5)
        b = _sample(tiny_model, inputs, rescale_k=1.2, rescale_sigma=0.5)

    assert torch.isfinite(a).all()
    assert torch.equal(a, b)


@requires_cuda
@pytest.mark.parametrize("mode", ["independent", "joint", "alternating"])
@pytest.mark.parametrize("speaker_kv_scale", [None, 1.7])
def test_cuda_graph_sampling_matches_eager(mode, speaker_kv_scale):
    from irodori_tts.cuda_graph import CUDAGraphForward

    device = torch.device("cuda")
    model = build_tiny_model(device)
    inputs = tiny_inputs(device)
    scales = (
        {"cfg_scale_text": 2.0, "cfg_scale_caption": 2.0, "cfg_scale_speaker": 2.0}
        if mode == "joint"
        else {}
    )
    runner = CUDAGraphForward(model.forward_with_encoded_conditions, capture_after=1)
    common = {
        "cfg_guidance_mode": mode,
        "speaker_kv_scale": speaker_kv_scale,
        "speaker_kv_min_t": 0.6,
        "num_steps": 10,
        **scales,
    }

    with torch.inference_mode():
        eager = _sample(model, inputs, **common)
        runner.begin_request()
        graphed = _sample(model, inputs, forward_fn=runner, **common)
        runner.end_request()

    assert runner.stats.replays > 0
    assert torch.equal(eager, graphed)


@requires_cuda
def test_cuda_graph_speaker_kv_on_off_alternation_does_not_leak():
    from irodori_tts.cuda_graph import CUDAGraphForward

    device = torch.device("cuda")
    model = build_tiny_model(device)
    inputs = tiny_inputs(device)
    runner = CUDAGraphForward(model.forward_with_encoded_conditions, capture_after=1)

    with torch.inference_mode():
        for speaker_kv_scale in (1.8, None, 1.8, None):
            common = {"speaker_kv_scale": speaker_kv_scale, "speaker_kv_min_t": 0.7}
            eager = _sample(model, inputs, **common)
            runner.begin_request()
            graphed = _sample(model, inputs, forward_fn=runner, **common)
            runner.end_request()
            assert torch.equal(eager, graphed)


@requires_cuda
def test_cuda_graph_short_long_short_matches_eager():
    from irodori_tts.cuda_graph import CUDAGraphForward

    device = torch.device("cuda")
    model = build_tiny_model(device)
    inputs = tiny_inputs(device)
    runner = CUDAGraphForward(model.forward_with_encoded_conditions, capture_after=1)
    outs = []

    with torch.inference_mode():
        for length in (5, 23, 5):
            eager = _sample(model, inputs, sequence_length=length)
            runner.begin_request()
            graphed = _sample(model, inputs, sequence_length=length, forward_fn=runner)
            runner.end_request()
            assert torch.equal(eager, graphed)
            outs.append(graphed)

    assert torch.equal(outs[0], outs[2])


@requires_cuda
@pytest.mark.parametrize("mode", ["independent", "joint", "alternating"])
@pytest.mark.parametrize("speaker_kv_scale", [None, 1.7])
def test_persistent_cuda_graph_matches_eager_across_requests(mode, speaker_kv_scale):
    """Graphs kept across requests must give eager results for new texts and lengths."""
    from irodori_tts.cuda_graph import CUDAGraphForward

    device = torch.device("cuda")
    model = build_tiny_model(device)
    first = tiny_inputs(device)
    # A different valid text length checks that no mask contents are baked in.
    second_mask = first["text_mask"].clone()
    second_mask[:, -2] = True
    second = {
        **first,
        "text_input_ids": (first["text_input_ids"] + 7) % 50,
        "text_mask": second_mask,
    }
    scales = (
        {"cfg_scale_text": 2.0, "cfg_scale_caption": 2.0, "cfg_scale_speaker": 2.0}
        if mode == "joint"
        else {}
    )
    common = {
        "cfg_guidance_mode": mode,
        "speaker_kv_scale": speaker_kv_scale,
        "speaker_kv_min_t": 0.6,
        "num_steps": 10,
        **scales,
    }
    runner = CUDAGraphForward(
        model.forward_with_encoded_conditions,
        capture_after=1,
        max_entries=16,
        persistent=True,
        # As in InferenceRuntime: a longer length replaces the RoPE cache buffer,
        # and graphs of shorter lengths still read the old one.
        keepalive=lambda: tuple(model.buffers()),
    )
    requests = [(first, 5), (second, 5), (first, 23), (second, 5), (first, 23)]

    with torch.inference_mode():
        for index, (inputs, length) in enumerate(requests):
            eager = _sample(model, inputs, sequence_length=length, **common)
            runner.begin_request()
            graphed = _sample(model, inputs, sequence_length=length, forward_fn=runner, **common)
            runner.end_request()
            assert torch.equal(eager, graphed), f"request {index}"
            if index == 2:
                captures_after_both_lengths = runner.stats.captures

    # The last two requests reuse lengths seen before, so they capture nothing.
    assert runner.stats.captures == captures_after_both_lengths
    assert runner.stats.replays > 0

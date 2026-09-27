from __future__ import annotations

import torch

from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT


def build_tiny_model(device: str | torch.device = "cpu", *, seed: int = 0) -> TextToLatentRFDiT:
    """A randomly initialized DiT small enough for unit tests.

    The real checkpoints zero-initialize several output projections, which
    would make every prediction zero; all parameters are re-drawn instead.
    """
    cfg = ModelConfig(
        latent_dim=8,
        model_dim=64,
        num_layers=2,
        num_heads=4,
        text_vocab_size=50,
        text_dim=32,
        text_layers=1,
        text_heads=2,
        use_caption_condition=True,
        use_speaker_condition=True,
        caption_dim=32,
        caption_layers=1,
        caption_heads=2,
        speaker_dim=32,
        speaker_layers=1,
        speaker_heads=2,
        timestep_embed_dim=16,
        adaln_rank=8,
    )
    torch.manual_seed(seed)
    model = TextToLatentRFDiT(cfg)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.2)
    return model.to(device).eval()


def tiny_inputs(device: str | torch.device = "cpu", *, text_len: int = 6, ref_len: int = 5):
    gen = torch.Generator(device="cpu").manual_seed(1)
    text_ids = torch.randint(0, 50, (1, text_len), generator=gen)
    text_mask = torch.ones(1, text_len, dtype=torch.bool)
    text_mask[:, -2:] = False
    caption_ids = torch.randint(0, 50, (1, 4), generator=gen)
    caption_mask = torch.ones(1, 4, dtype=torch.bool)
    ref_latent = torch.randn(1, ref_len, 8, generator=gen)
    ref_mask = torch.ones(1, ref_len, dtype=torch.bool)
    return {
        "text_input_ids": text_ids.to(device),
        "text_mask": text_mask.to(device),
        "caption_input_ids": caption_ids.to(device),
        "caption_mask": caption_mask.to(device),
        "ref_latent": ref_latent.to(device),
        "ref_mask": ref_mask.to(device),
    }

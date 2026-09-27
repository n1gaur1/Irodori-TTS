from __future__ import annotations

import pytest

from irodori_tts.model import TextToLatentRFDiT
from tests.tiny_model import build_tiny_model


@pytest.fixture
def tiny_model() -> TextToLatentRFDiT:
    return build_tiny_model()

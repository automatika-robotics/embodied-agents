"""Tests for LocalTTS wrapper — no ROS needed."""

import sys
import io
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_deps():
    """Mock kokoro_onnx and soundfile."""
    mock_kokoro_mod = MagicMock()
    mock_sf = MagicMock()
    with patch.dict(
        sys.modules, {"kokoro_onnx": mock_kokoro_mod, "soundfile": mock_sf}
    ):
        yield mock_kokoro_mod, mock_sf


@pytest.fixture
def local_tts(mock_deps):
    mock_kokoro_mod, mock_sf = mock_deps
    from agents.utils.local_tts import LocalTTS

    tts = LocalTTS.__new__(LocalTTS)
    tts._kokoro = MagicMock()
    tts.device = "cpu"
    tts.ncpu = 1
    return tts, mock_sf


class TestLocalTTSCall:
    def test_with_text(self, local_tts):
        tts, mock_sf = local_tts
        samples = np.zeros(16000, dtype=np.float32)
        tts._kokoro.create.return_value = (samples, 24000)

        # Mock soundfile.write to actually write something
        def fake_write(buf, data, sr, format):
            buf.write(b"RIFF_fake_wav_data")

        mock_sf.write.side_effect = fake_write

        result = tts({"query": "Hello world"})
        assert isinstance(result["output"], bytes)
        assert len(result["output"]) > 0
        tts._kokoro.create.assert_called_once_with(
            "Hello world", voice="af_heart", speed=1.0
        )

    def test_empty_text(self, local_tts):
        tts, _ = local_tts
        result = tts({"query": ""})
        assert result["output"] == b""

"""Tests for LocalTTS wrapper — no ROS needed."""

import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_sherpa():
    """Mock sherpa_onnx before importing LocalTTS."""
    mock = MagicMock()
    with patch.dict(sys.modules, {"sherpa_onnx": mock}):
        yield mock


@pytest.fixture
def local_tts(mock_sherpa):
    from agents.utils.local_tts import LocalTTS

    tts = LocalTTS.__new__(LocalTTS)
    tts._tts = MagicMock()
    tts.device = "cpu"
    tts.ncpu = 1
    tts.speaker_id = 0
    tts.stream = False
    tts._generation_config = None
    return tts


class TestLocalTTSCall:
    def test_with_text(self, local_tts):
        samples = np.zeros(16000, dtype=np.float32)
        mock_audio = MagicMock()
        mock_audio.samples = samples
        mock_audio.sample_rate = 24000
        local_tts._tts.generate.return_value = mock_audio

        result = local_tts({"query": "Hello world"})
        assert isinstance(result["output"], bytes)
        assert len(result["output"]) > 0
        # Verify WAV header
        assert result["output"][:4] == b"RIFF"
        # terminal punctuation is appended before synthesis
        local_tts._tts.generate.assert_called_once_with(
            "Hello world.", sid=0, speed=1.0
        )

    def test_empty_text(self, local_tts):
        result = local_tts({"query": ""})
        assert result["output"] == b""

    def test_terminal_punctuation_added(self, local_tts):
        """Unpunctuated text destabilizes LM stop prediction (runaway
        babbling) — a terminal period must be appended before synthesis."""
        mock_audio = MagicMock()
        mock_audio.samples = np.zeros(100, dtype=np.float32)
        mock_audio.sample_rate = 24000
        local_tts._tts.generate.return_value = mock_audio

        local_tts({"query": "who are you"})
        local_tts._tts.generate.assert_called_once_with(
            "who are you.", sid=0, speed=1.0
        )
        local_tts._tts.generate.reset_mock()

        local_tts({"query": "who are you?  "})
        local_tts._tts.generate.assert_called_once_with(
            "who are you?", sid=0, speed=1.0
        )

    def test_speaker_id_passed_to_generate(self, local_tts):
        mock_audio = MagicMock()
        mock_audio.samples = np.zeros(100, dtype=np.float32)
        mock_audio.sample_rate = 24000
        local_tts._tts.generate.return_value = mock_audio
        local_tts.speaker_id = 3

        local_tts({"query": "Hello"})
        local_tts._tts.generate.assert_called_once_with("Hello.", sid=3, speed=1.0)


class TestLoadModelRepoLocalPath:
    def test_local_directory_returned_as_is(self, tmp_path):
        from agents.utils.utils import load_model_repo

        bundle = tmp_path / "my_bundle"
        bundle.mkdir()
        assert load_model_repo("local_tts", str(bundle)) == str(bundle)


class TestLocalTTSStreaming:
    def test_stream_yields_wav_chunks(self, local_tts):
        local_tts.stream = True
        local_tts._tts.sample_rate = 24000

        def fake_generate(text, sid=0, speed=1.0, callback=None):
            # emulate sherpa-onnx invoking the callback per synthesized chunk
            callback(np.zeros(1000, dtype=np.float32), 0.5)
            callback(np.full(500, 0.1, dtype=np.float32), 1.0)
            return MagicMock()

        local_tts._tts.generate.side_effect = fake_generate

        result = local_tts({"query": "Hello"})
        chunks = list(result["output"])
        assert len(chunks) == 2
        assert all(chunk[:4] == b"RIFF" for chunk in chunks)

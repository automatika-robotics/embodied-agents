"""Tests for LocalSTT wrapper — no ROS needed."""

import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_sherpa():
    """Mock sherpa_onnx before importing LocalSTT."""
    mock = MagicMock()
    with patch.dict(sys.modules, {"sherpa_onnx": mock}):
        yield mock


@pytest.fixture
def local_stt(mock_sherpa):
    from agents.utils.local_stt import LocalSTT

    stt = LocalSTT.__new__(LocalSTT)
    stt._recognizer = MagicMock()
    stt.device = "cpu"
    stt.ncpu = 1
    return stt


class TestLocalSTTCall:
    def test_with_bytes(self, local_stt):
        # Create int16 audio bytes
        audio = np.array([0, 100, -100, 32767], dtype=np.int16)
        audio_bytes = audio.tobytes()
        mock_result = MagicMock()
        mock_result.text = "hello world"
        local_stt._recognizer.recognize.return_value = mock_result

        result = local_stt({"query": audio_bytes})
        assert result["output"] == "hello world"
        local_stt._recognizer.recognize.assert_called_once()

    def test_with_numpy(self, local_stt):
        audio = np.array([0.1, 0.2, -0.1], dtype=np.float32)
        mock_result = MagicMock()
        mock_result.text = "test"
        local_stt._recognizer.recognize.return_value = mock_result

        result = local_stt({"query": audio})
        assert result["output"] == "test"

    def test_multidimensional_flattened(self, local_stt):
        audio = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        mock_result = MagicMock()
        mock_result.text = "flat"
        local_stt._recognizer.recognize.return_value = mock_result

        result = local_stt({"query": audio})
        assert result["output"] == "flat"
        # Verify the array was flattened
        call_arg = local_stt._recognizer.recognize.call_args[0][0]
        assert call_arg.ndim == 1

    def test_unsupported_type(self, local_stt):
        result = local_stt({"query": 12345})
        assert result["output"] == ""

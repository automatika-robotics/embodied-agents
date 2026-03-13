"""Tests for LocalSTT wrapper — no ROS needed."""

import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_moonshine():
    """Mock moonshine_onnx before importing LocalSTT."""
    mock = MagicMock()
    with patch.dict(sys.modules, {"moonshine_onnx": mock}):
        yield mock


@pytest.fixture
def local_stt(mock_moonshine):
    from agents.utils.local_stt import LocalSTT

    stt = LocalSTT.__new__(LocalSTT)
    stt._moonshine = mock_moonshine
    stt.device = "cpu"
    stt.ncpu = 1
    return stt


class TestLocalSTTCall:
    def test_with_bytes(self, local_stt):
        # Create int16 audio bytes
        audio = np.array([0, 100, -100, 32767], dtype=np.int16)
        audio_bytes = audio.tobytes()
        local_stt._moonshine.transcribe.return_value = "hello world"

        result = local_stt({"query": audio_bytes})
        assert result["output"] == "hello world"
        local_stt._moonshine.transcribe.assert_called_once()

    def test_with_numpy(self, local_stt):
        audio = np.array([0.1, 0.2, -0.1], dtype=np.float32)
        local_stt._moonshine.transcribe.return_value = "test"

        result = local_stt({"query": audio})
        assert result["output"] == "test"

    def test_multidimensional_flattened(self, local_stt):
        audio = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        local_stt._moonshine.transcribe.return_value = "flat"

        result = local_stt({"query": audio})
        assert result["output"] == "flat"
        # Verify the array was flattened
        call_arg = local_stt._moonshine.transcribe.call_args[0][0]
        assert call_arg.ndim == 1

    def test_list_result_joined(self, local_stt):
        audio = np.array([0.0], dtype=np.float32)
        local_stt._moonshine.transcribe.return_value = ["hello", "world"]

        result = local_stt({"query": audio})
        assert result["output"] == "hello world"

    def test_unsupported_type(self, local_stt):
        result = local_stt({"query": 12345})
        assert result["output"] == ""

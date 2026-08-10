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
    stt._sample_rate = 16000
    stt.device = "cpu"
    stt.ncpu = 1
    return stt


def _setup_mock_stream(local_stt, text):
    """Configure the mock recognizer to return a stream with the given text."""
    mock_stream = MagicMock()
    mock_stream.result.text = text
    local_stt._recognizer.create_stream.return_value = mock_stream
    return mock_stream


class TestLocalSTTCall:
    def test_with_bytes(self, local_stt):
        # Create int16 audio bytes
        audio = np.array([0, 100, -100, 32767], dtype=np.int16)
        audio_bytes = audio.tobytes()
        mock_stream = _setup_mock_stream(local_stt, "hello world")

        result = local_stt({"query": audio_bytes})
        assert result["output"] == "hello world"
        mock_stream.accept_waveform.assert_called_once()
        local_stt._recognizer.decode_stream.assert_called_once_with(mock_stream)

    def test_with_numpy(self, local_stt):
        audio = np.array([0.1, 0.2, -0.1], dtype=np.float32)
        _setup_mock_stream(local_stt, "test")

        result = local_stt({"query": audio})
        assert result["output"] == "test"

    def test_multidimensional_flattened(self, local_stt):
        audio = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        mock_stream = _setup_mock_stream(local_stt, "flat")

        result = local_stt({"query": audio})
        assert result["output"] == "flat"
        # Verify the array was flattened
        call_args = mock_stream.accept_waveform.call_args[0]
        assert call_args[1].ndim == 1

    def test_unsupported_type(self, local_stt):
        result = local_stt({"query": 12345})
        assert result["output"] == ""


class TestLocalSTTModelDetection:
    """Family detection from bundle contents — pure filesystem, no sherpa."""

    @staticmethod
    def _make_bundle(tmp_path, files, dirname="bundle"):
        bundle = tmp_path / dirname
        bundle.mkdir()
        for name in files:
            if name.endswith("/"):
                (bundle / name.rstrip("/")).mkdir()
            else:
                (bundle / name).touch()
        return str(bundle)

    def test_parakeet_transducer_detected(self, tmp_path):
        from agents.utils.local_stt import detect_model_family

        bundle = self._make_bundle(
            tmp_path,
            ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"],
            dirname="sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
        )
        family, files = detect_model_family(bundle)
        assert family == "transducer"
        assert files["joiner"].endswith("joiner.int8.onnx")

    def test_whisper_fallback(self, tmp_path):
        from agents.utils.local_stt import detect_model_family

        bundle = self._make_bundle(
            tmp_path,
            [
                "tiny.en-encoder.int8.onnx",
                "tiny.en-decoder.int8.onnx",
                "tiny.en-tokens.txt",
            ],
        )
        family, files = detect_model_family(bundle)
        assert family == "whisper"
        assert "encoder" in files and "decoder" in files

    def test_qwen3_detected(self, tmp_path):
        from agents.utils.local_stt import detect_model_family

        bundle = self._make_bundle(
            tmp_path,
            ["conv_frontend.onnx", "encoder.int8.onnx", "decoder.int8.onnx", "tokenizer/"],
        )
        family, files = detect_model_family(bundle)
        assert family == "qwen3_asr"
        assert files["tokenizer"].endswith("tokenizer")

    def test_moonshine_v1_and_v2(self, tmp_path):
        from agents.utils.local_stt import detect_model_family

        v1 = self._make_bundle(
            tmp_path,
            [
                "preprocess.onnx",
                "encode.int8.onnx",
                "uncached_decode.int8.onnx",
                "cached_decode.int8.onnx",
                "tokens.txt",
            ],
            dirname="moonshine-v1",
        )
        assert detect_model_family(v1)[0] == "moonshine"

        v2 = self._make_bundle(
            tmp_path,
            ["encoder_model.ort", "decoder_model_merged.ort", "tokens.txt"],
            dirname="moonshine-v2",
        )
        assert detect_model_family(v2)[0] == "moonshine_v2"

    def test_sense_voice_needs_name_hint(self, tmp_path):
        from agents.utils.local_stt import detect_model_family

        bundle = self._make_bundle(
            tmp_path,
            ["model.int8.onnx", "tokens.txt"],
            dirname="sherpa-onnx-sense-voice-zh-en-int8",
        )
        assert detect_model_family(bundle)[0] == "sense_voice"

    def test_model_type_override(self, tmp_path):
        from agents.utils.local_stt import detect_model_family

        bundle = self._make_bundle(
            tmp_path, ["model.onnx", "tokens.txt"], dirname="anon"
        )
        family, _ = detect_model_family(bundle, model_type="nemo_ctc")
        assert family == "nemo_ctc"

    def test_undetectable_raises_with_hint(self, tmp_path):
        from agents.utils.local_stt import detect_model_family

        bundle = self._make_bundle(tmp_path, ["README.md"])
        with pytest.raises(ValueError, match="model_type"):
            detect_model_family(bundle)


class TestLocalSTTModelOptions:
    """Options validation against real sherpa factory signatures."""

    @staticmethod
    def _mocked_sherpa_with_real_signature():
        """A mock sherpa module whose from_transducer has the real signature."""
        import inspect

        real_sherpa = pytest.importorskip("sherpa_onnx")
        mock = MagicMock()
        factory_mock = MagicMock()
        factory_mock.__signature__ = inspect.signature(
            real_sherpa.OfflineRecognizer.from_transducer
        )
        mock.OfflineRecognizer.from_transducer = factory_mock
        return mock

    def test_transducer_options_and_nemo_model_type(self, tmp_path):
        mock = self._mocked_sherpa_with_real_signature()
        bundle = tmp_path / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
        bundle.mkdir()
        for f in [
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "joiner.int8.onnx",
            "tokens.txt",
        ]:
            (bundle / f).touch()

        from agents.utils.local_stt import LocalSTT

        with patch.dict(sys.modules, {"sherpa_onnx": mock}):
            stt = LocalSTT(
                str(bundle),
                device="cpu",
                ncpu=2,
                model_options={"decoding_method": "greedy_search"},
            )
        assert stt.model_family == "transducer"
        _, kwargs = mock.OfflineRecognizer.from_transducer.call_args
        assert kwargs["model_type"] == "nemo_transducer"
        assert kwargs["decoding_method"] == "greedy_search"
        assert kwargs["num_threads"] == 2
        # language is not a transducer factory param and must not be passed
        assert "language" not in kwargs

    def test_unknown_option_raises(self, tmp_path):
        mock = self._mocked_sherpa_with_real_signature()
        bundle = tmp_path / "parakeet"
        bundle.mkdir()
        for f in ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]:
            (bundle / f).touch()

        from agents.utils.local_stt import LocalSTT

        with patch.dict(sys.modules, {"sherpa_onnx": mock}):
            with pytest.raises(ValueError, match="Valid options"):
                LocalSTT(str(bundle), model_options={"not_an_option": 1})

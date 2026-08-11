"""Tests for TextToSpeech component — requires rclpy."""

import pytest
from unittest.mock import MagicMock

from agents.config import TextToSpeechConfig
from agents.ros import Topic
from agents.components.texttospeech import TextToSpeech
from tests.conftest import mock_component_internals


class TestTTSConstruction:
    def test_with_model_client(self, rclpy_init, mock_model_client):
        text = Topic(name="text", msg_type="String")
        audio = Topic(name="audio", msg_type="Audio")
        comp = TextToSpeech(
            inputs=[text],
            outputs=[audio],
            model_client=mock_model_client,
            config=TextToSpeechConfig(),
            trigger=text,
            component_name="test_tts",
        )
        assert comp.model_client is mock_model_client

    def test_with_local_model(self, rclpy_init):
        text = Topic(name="text", msg_type="String")
        comp = TextToSpeech(
            inputs=[text],
            config=TextToSpeechConfig(enable_local_model=True, stream=False),
            trigger=text,
            component_name="test_tts_local",
        )
        assert comp.config.enable_local_model is True

    def test_no_client_no_local_raises(self, rclpy_init):
        text = Topic(name="text", msg_type="String")
        with pytest.raises(TypeError):
            TextToSpeech(
                inputs=[text],
                config=TextToSpeechConfig(stream=False),
                trigger=text,
                component_name="test_tts_fail",
            )


class TestTTSCreateInput:
    @pytest.fixture
    def tts(self, rclpy_init, mock_model_client):
        text = Topic(name="text", msg_type="String")
        audio = Topic(name="audio", msg_type="Audio")
        comp = TextToSpeech(
            inputs=[text],
            outputs=[audio],
            model_client=mock_model_client,
            config=TextToSpeechConfig(),
            trigger=text,
            component_name="test_tts_input",
        )
        mock_component_internals(comp)
        return comp

    def test_from_trigger(self, tts):
        trigger = Topic(name="text", msg_type="String")
        mock_cb = MagicMock()
        mock_cb.get_output.return_value = "Hello world"
        tts.trig_callbacks = {"text": mock_cb}

        result = tts._create_input(topic=trigger)
        assert result is not None
        assert result["query"] == "Hello world"

    def test_from_text_kwarg(self, tts):
        result = tts._create_input(text="Direct text")
        assert result is not None
        assert result["query"] == "Direct text"


class TestLocalTTSModelDetection:
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

    def test_pocket_detected(self, tmp_path):
        from agents.utils.local_tts import detect_model_family

        bundle = self._make_bundle(
            tmp_path,
            [
                "lm_main.int8.onnx",
                "lm_flow.int8.onnx",
                "encoder.onnx",
                "decoder.int8.onnx",
                "text_conditioner.onnx",
                "vocab.json",
                "token_scores.json",
            ],
        )
        family, files = detect_model_family(bundle)
        assert family == "pocket"
        assert files["lm_main"].endswith("lm_main.int8.onnx")
        assert files["vocab_json"].endswith("vocab.json")

    def test_kokoro_detected(self, tmp_path):
        from agents.utils.local_tts import detect_model_family

        bundle = self._make_bundle(
            tmp_path,
            ["model.onnx", "voices.bin", "tokens.txt", "espeak-ng-data/"],
        )
        family, files = detect_model_family(bundle)
        assert family == "kokoro"
        assert files["data_dir"].endswith("espeak-ng-data")

    def test_kitten_needs_name_hint(self, tmp_path):
        from agents.utils.local_tts import detect_model_family

        files = ["model.onnx", "voices.bin", "tokens.txt", "espeak-ng-data/"]
        # same layout as kokoro: kitten only wins on a name hint
        bundle = self._make_bundle(tmp_path, files, dirname="kitten-nano-en")
        family, _ = detect_model_family(bundle)
        assert family == "kitten"

    def test_model_type_override(self, tmp_path):
        from agents.utils.local_tts import detect_model_family

        bundle = self._make_bundle(
            tmp_path, ["model.onnx", "voices.bin", "tokens.txt"]
        )
        family, _ = detect_model_family(bundle, model_type="kitten")
        assert family == "kitten"

    def test_vits_detected_without_voices(self, tmp_path):
        from agents.utils.local_tts import detect_model_family

        bundle = self._make_bundle(tmp_path, ["en_US-amy.onnx", "tokens.txt"])
        family, _ = detect_model_family(bundle)
        assert family == "vits"

    def test_undetectable_raises_with_hint(self, tmp_path):
        from agents.utils.local_tts import detect_model_family

        bundle = self._make_bundle(tmp_path, ["README.md"])
        with pytest.raises(ValueError, match="model_type"):
            detect_model_family(bundle)

    def test_unknown_model_type_raises(self, tmp_path):
        from agents.utils.local_tts import detect_model_family

        bundle = self._make_bundle(tmp_path, ["model.onnx", "tokens.txt"])
        with pytest.raises(ValueError, match="Unknown model_type"):
            detect_model_family(bundle, model_type="not_a_family")


class TestLocalTTSModelOptions:
    """Option splitting/validation against sherpa-onnx config fields."""

    def test_options_split_and_unknown_key(self):
        sherpa_onnx = pytest.importorskip("sherpa_onnx")
        from agents.utils.local_tts import split_model_options

        sub, top, generation = split_model_options(
            sherpa_onnx.OfflineTtsVitsModelConfig,
            {
                "noise_scale": 0.5,
                "length_scale": 1.2,
                "silence_scale": 0.1,
                "voice": "loona",
            },
        )
        assert sub == {"noise_scale": 0.5, "length_scale": 1.2}
        assert top == {"silence_scale": 0.1}
        assert generation == {"voice": "loona"}

        with pytest.raises(ValueError, match="Valid family options"):
            split_model_options(
                sherpa_onnx.OfflineTtsVitsModelConfig, {"not_a_real_option": 1}
            )

    def test_voice_resolution(self, tmp_path):
        from agents.utils.local_tts import resolve_voice_wav

        bundle = tmp_path / "bundle"
        (bundle / "test_wavs").mkdir(parents=True)
        (bundle / "test_wavs" / "bria.wav").touch()
        (bundle / "test_wavs" / "loona.wav").touch()

        # default: first bundle voice
        assert resolve_voice_wav(bundle, None).endswith("bria.wav")
        # by name
        assert resolve_voice_wav(bundle, "loona").endswith("loona.wav")
        # unknown name lists available voices
        with pytest.raises(ValueError, match="bria"):
            resolve_voice_wav(bundle, "not_a_voice")
        # no wavs in bundle
        empty = tmp_path / "empty"
        empty.mkdir()
        assert resolve_voice_wav(empty, None) is None

"""Tests for SpeechToText component — requires rclpy."""

import sys

import pytest
from unittest.mock import MagicMock, patch

from agents.config import SpeechToTextConfig
from agents.ros import Topic
from agents.components.speechtotext import SpeechToText
from tests.conftest import mock_component_internals


class TestSTTConstruction:
    def test_with_model_client(self, rclpy_init, mock_model_client):
        audio = Topic(name="audio", msg_type="Audio")
        text = Topic(name="text", msg_type="String")
        comp = SpeechToText(
            inputs=[audio],
            outputs=[text],
            model_client=mock_model_client,
            config=SpeechToTextConfig(),
            trigger=audio,
            component_name="test_stt",
        )
        assert comp.model_client is mock_model_client

    def test_with_local_model(self, rclpy_init):
        audio = Topic(name="audio", msg_type="Audio")
        text = Topic(name="text", msg_type="String")
        comp = SpeechToText(
            inputs=[audio],
            outputs=[text],
            config=SpeechToTextConfig(enable_local_model=True),
            trigger=audio,
            component_name="test_stt_local",
        )
        assert comp.config.enable_local_model is True

    def test_no_client_no_local_raises(self, rclpy_init):
        audio = Topic(name="audio", msg_type="Audio")
        text = Topic(name="text", msg_type="String")
        with pytest.raises(TypeError):
            SpeechToText(
                inputs=[audio],
                outputs=[text],
                config=SpeechToTextConfig(),
                trigger=audio,
                component_name="test_stt_fail",
            )

    def test_float_trigger_raises(self, rclpy_init, mock_model_client):
        audio = Topic(name="audio", msg_type="Audio")
        text = Topic(name="text", msg_type="String")
        with pytest.raises(TypeError):
            SpeechToText(
                inputs=[audio],
                outputs=[text],
                model_client=mock_model_client,
                config=SpeechToTextConfig(),
                trigger=1.0,
                component_name="test_stt_timed",
            )

    def test_stream_without_ws_raises(self, rclpy_init, mock_model_client):
        audio = Topic(name="audio", msg_type="Audio")
        text = Topic(name="text", msg_type="String")
        with pytest.raises(TypeError):
            SpeechToText(
                inputs=[audio],
                outputs=[text],
                model_client=mock_model_client,
                config=SpeechToTextConfig(stream=True, enable_vad=True),
                trigger=audio,
                component_name="test_stt_stream",
            )


class TestSTTCreateInput:
    @pytest.fixture
    def stt(self, rclpy_init, mock_model_client):
        audio = Topic(name="audio", msg_type="Audio")
        text = Topic(name="text", msg_type="String")
        comp = SpeechToText(
            inputs=[audio],
            outputs=[text],
            model_client=mock_model_client,
            config=SpeechToTextConfig(),
            trigger=audio,
            component_name="test_stt_input",
        )
        mock_component_internals(comp)
        return comp

    def test_from_vad_speech(self, stt):
        stt.config.enable_vad = True
        result = stt._create_input(speech=[b"aaa", b"bbb"])
        assert result is not None
        assert result["query"] == b"aaabbb"

    def test_from_trigger(self, stt):
        trigger = Topic(name="audio", msg_type="Audio")
        mock_cb = MagicMock()
        mock_cb.get_output.return_value = b"audio_bytes"
        stt.trig_callbacks = {"audio": mock_cb}

        result = stt._create_input(topic=trigger)
        assert result is not None
        assert result["query"] == b"audio_bytes"

    def test_empty_returns_none(self, stt):
        trigger = Topic(name="audio", msg_type="Audio")
        mock_cb = MagicMock()
        mock_cb.get_output.return_value = None
        stt.trig_callbacks = {"audio": mock_cb}

        result = stt._create_input(topic=trigger)
        assert result is None


class TestWakeWordSpotter:
    """Keyword-file building for the sherpa keyword spotter."""

    @staticmethod
    def _bundle_with_tokens(tmp_path, tokens):
        bundle = tmp_path / "kws-bundle"
        bundle.mkdir()
        (bundle / "bpe.model").touch()
        (bundle / "tokens.txt").write_text(
            "".join(f"{token} {i}\n" for i, token in enumerate(tokens))
        )
        return bundle

    @staticmethod
    def _mock_sentencepiece(pieces_map):
        """Mock sentencepiece whose encode_as_pieces uses a lookup table."""
        spm = MagicMock()
        sp = spm.SentencePieceProcessor.return_value
        sp.encode_as_pieces.side_effect = lambda text: pieces_map.get(text, ["<unk>"])
        return spm

    def test_keywords_file_built_with_casing_fallback(self, tmp_path):
        from agents.utils.voice import WakeWordSpotter

        bundle = self._bundle_with_tokens(tmp_path, ["▁HEY", "▁JARVIS"])
        spm = self._mock_sentencepiece({"HEY JARVIS": ["▁HEY", "▁JARVIS"]})
        with patch.dict(sys.modules, {"sentencepiece": spm}):
            keywords_file = WakeWordSpotter._build_keywords_file(
                bundle, ["hey jarvis"]
            )
        content = (bundle / "agents_keywords.txt").read_text()
        assert keywords_file.endswith("agents_keywords.txt")
        # uppercase encoding matched the vocab; label must be space-free
        assert content == "▁HEY ▁JARVIS @hey_jarvis\n"

    def test_unencodable_phrase_raises(self, tmp_path):
        from agents.utils.voice import WakeWordSpotter

        bundle = self._bundle_with_tokens(tmp_path, ["▁HEY", "▁JARVIS"])
        spm = self._mock_sentencepiece({})  # nothing encodes into the vocab
        with patch.dict(sys.modules, {"sentencepiece": spm}):
            with pytest.raises(ValueError, match="cannot be encoded"):
                WakeWordSpotter._build_keywords_file(bundle, ["bonjour robot"])

    def test_missing_bpe_model_raises(self, tmp_path):
        from agents.utils.voice import WakeWordSpotter

        bundle = tmp_path / "no-bpe"
        bundle.mkdir()
        (bundle / "tokens.txt").touch()
        with pytest.raises(FileNotFoundError, match="BPE"):
            WakeWordSpotter._build_keywords_file(bundle, ["hey jarvis"])


class TestLoadModelArchive:
    def test_local_directory_returned_as_is(self, tmp_path):
        from agents.utils.utils import load_model_archive

        bundle = tmp_path / "kws"
        bundle.mkdir()
        assert load_model_archive("wakeword_kws", str(bundle)) == str(bundle)

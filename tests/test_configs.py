"""Tests for config validation in agents/config.py — no ROS needed."""

import pytest
from agents.config import (
    LLMConfig,
    MLLMConfig,
    SpeechToTextConfig,
    TextToSpeechConfig,
    SemanticRouterConfig,
)


class TestLLMConfig:
    def test_construction(self):
        """LLMConfig can be constructed with defaults."""
        LLMConfig()

    def test_temperature_must_be_positive(self):
        with pytest.raises(ValueError):
            LLMConfig(temperature=0.0)

    def test_max_tokens_must_be_positive(self):
        with pytest.raises(ValueError):
            LLMConfig(max_new_tokens=0)

    def test_history_size_gt_4(self):
        with pytest.raises(ValueError):
            LLMConfig(history_size=4)
        c = LLMConfig(history_size=5)
        assert c.history_size == 5

    def test_empty_response_terminator(self):
        with pytest.raises(ValueError):
            LLMConfig(response_terminator="")

    def test_get_inference_params(self):
        c = LLMConfig()
        params = c._get_inference_params()
        assert "temperature" in params
        assert "max_new_tokens" in params
        assert "stream" in params

    def test_local_model_enabled(self):
        c = LLMConfig(enable_local_model=True)
        assert c.enable_local_model is True


class TestMLLMConfig:
    def test_construction(self):
        """MLLMConfig can be constructed with defaults."""
        MLLMConfig()

    def test_task_with_stream_raises(self):
        with pytest.raises(ValueError):
            MLLMConfig(task="general", stream=True)

    def test_nongeneral_task_with_local_raises(self):
        with pytest.raises(ValueError):
            MLLMConfig(task="pointing", enable_local_model=True)

    def test_general_task_with_local_ok(self):
        c = MLLMConfig(task="general", enable_local_model=True)
        assert c.task == "general"
        assert c.enable_local_model is True

    def test_stream_with_local_raises(self):
        with pytest.raises(ValueError):
            MLLMConfig(enable_local_model=True, stream=True)

    def test_inference_params_with_task(self):
        c = MLLMConfig(task="general")
        params = c._get_inference_params()
        assert "task" in params
        assert params["task"] == "general"

    def test_inference_params_without_task(self):
        c = MLLMConfig()
        params = c._get_inference_params()
        assert "task" not in params


class TestSTTConfig:
    def test_construction(self):
        """SpeechToTextConfig can be constructed with defaults."""
        SpeechToTextConfig()

    def test_wakeword_requires_vad(self):
        with pytest.raises(ValueError):
            SpeechToTextConfig(enable_wakeword=True)

    def test_stream_requires_vad(self):
        with pytest.raises(ValueError):
            SpeechToTextConfig(stream=True)

    def test_stream_with_local_ok(self):
        """Local model + stream is accepted at config time; stream is
        disabled at runtime by _deploy_local_model."""
        c = SpeechToTextConfig(stream=True, enable_vad=True, enable_local_model=True)
        assert c.stream is True  # will be overridden at deploy time

    def test_vad_threshold_range(self):
        with pytest.raises(ValueError):
            SpeechToTextConfig(vad_threshold=1.5)
        with pytest.raises(ValueError):
            SpeechToTextConfig(vad_threshold=-0.1)

    def test_min_chunk_gt_500(self):
        with pytest.raises(ValueError):
            SpeechToTextConfig(min_chunk_size=500)

    def test_post_init_privates(self):
        c = SpeechToTextConfig(stream=False, enable_vad=False)
        assert c._word_timestamps is False
        assert c._vad_filter is True

        c2 = SpeechToTextConfig(stream=True, enable_vad=True)
        assert c2._word_timestamps is True
        assert c2._vad_filter is False


class TestTTSConfig:
    def test_construction(self):
        """TextToSpeechConfig can be constructed with defaults."""
        TextToSpeechConfig()

    def test_stream_with_local_ok(self):
        """Local model + stream is accepted at config time; stream is
        disabled at runtime by _deploy_local_model."""
        c = TextToSpeechConfig(enable_local_model=True)
        assert c.enable_local_model is True
        assert c.stream is True  # will be overridden at deploy time

    def test_local_no_stream_ok(self):
        c = TextToSpeechConfig(enable_local_model=True, stream=False)
        assert c.enable_local_model is True
        assert c.stream is False

    def test_stream_to_ip_without_play_raises(self):
        with pytest.raises(ValueError):
            TextToSpeechConfig(stream_to_ip="192.168.1.1", stream_to_port=1234)

    def test_stream_to_ip_without_port_raises(self):
        with pytest.raises(ValueError):
            TextToSpeechConfig(stream_to_ip="192.168.1.1", play_on_device=True)

    def test_stream_to_port_without_ip_raises(self):
        with pytest.raises(ValueError):
            TextToSpeechConfig(stream_to_port=1234, play_on_device=True)


class TestRouterConfig:
    def test_construction(self):
        c = SemanticRouterConfig(router_name="test_router")
        assert c.router_name == "test_router"

    def test_distance_func_options(self):
        for func in ["l2", "ip", "cosine"]:
            c = SemanticRouterConfig(router_name="r", distance_func=func)
            assert c.distance_func == func

    def test_max_distance_range(self):
        with pytest.raises(ValueError):
            SemanticRouterConfig(router_name="r", maximum_distance=0.05)
        c = SemanticRouterConfig(router_name="r", maximum_distance=0.5)
        assert c.maximum_distance == 0.5
        with pytest.raises(ValueError):
            SemanticRouterConfig(router_name="r", maximum_distance=1.1)

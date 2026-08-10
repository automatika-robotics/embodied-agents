"""Tests for config validation in agents/config.py — no ROS needed."""

import json

import pytest
from agents.config import (
    LLMConfig,
    MLLMConfig,
    MotionDetectorConfig,
    VLAConfig,
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

    def test_wakeword_defaults(self):
        c = SpeechToTextConfig(enable_vad=True, enable_wakeword=True)
        assert c.wakeword_phrase == "ok robot"
        assert c.wakeword_threshold == 0.25
        assert c.wakeword_model_path.endswith(".tar.bz2")
        # the openWakeWord-era model fields are gone
        assert not hasattr(c, "melspectrogram_model_path")
        assert not hasattr(c, "embedding_model_path")

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
        """Local model + stream is a supported combination (chunks are
        yielded by the local model as they are synthesized)."""
        c = TextToSpeechConfig(enable_local_model=True)
        assert c.enable_local_model is True
        assert c.stream is True

    def test_local_no_stream_ok(self):
        c = TextToSpeechConfig(enable_local_model=True, stream=False)
        assert c.enable_local_model is True
        assert c.stream is False

    def test_local_model_defaults(self):
        c = TextToSpeechConfig()
        assert "pocket-tts" in c.local_model_path
        assert c.speaker_id == 0
        assert c.local_model_options == {}

    def test_speaker_id_non_negative(self):
        with pytest.raises(ValueError):
            TextToSpeechConfig(speaker_id=-1)
        c = TextToSpeechConfig(speaker_id=3)
        assert c.speaker_id == 3

    def test_local_model_options_round_trip(self):
        config = TextToSpeechConfig(
            local_model_options={"model_type": "kokoro", "length_scale": 1.2}
        )
        rebuilt = TextToSpeechConfig(**json.loads(config.to_json()))
        assert rebuilt.local_model_options == {
            "model_type": "kokoro",
            "length_scale": 1.2,
        }

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


class TestMotionDetectorConfig:
    def test_construction(self):
        """MotionDetectorConfig can be constructed with defaults."""
        MotionDetectorConfig()

    def test_threshold_range(self):
        with pytest.raises(ValueError):
            MotionDetectorConfig(threshold=0.05)
        with pytest.raises(ValueError):
            MotionDetectorConfig(threshold=5.1)

    def test_voxel_size_positive(self):
        with pytest.raises(ValueError):
            MotionDetectorConfig(voxel_size=0.0)

    def test_cluster_params_positive(self):
        with pytest.raises(ValueError):
            MotionDetectorConfig(changed_voxel_threshold=0)
        with pytest.raises(ValueError):
            MotionDetectorConfig(min_cluster_size=0)
        with pytest.raises(ValueError):
            MotionDetectorConfig(max_clusters=0)

    def test_motion_stop_delay_non_negative(self):
        with pytest.raises(ValueError):
            MotionDetectorConfig(motion_stop_delay=-1)

    def test_device_options(self):
        for device in ["cpu", "cuda"]:
            c = MotionDetectorConfig(device=device)
            assert c.device == device

    def test_flow_kwargs_validated_against_defaults(self):
        with pytest.raises(AttributeError):
            MotionDetectorConfig(flow_kwargs={"not_a_flow_param": 1})

    def test_serialization_round_trip(self):
        """Multiprocess launch path: config survives a JSON round trip."""
        config = MotionDetectorConfig(voxel_size=0.2, base_frame="base_footprint")
        rebuilt = MotionDetectorConfig(**json.loads(config.to_json()))
        assert rebuilt.voxel_size == pytest.approx(0.2)
        assert rebuilt.base_frame == "base_footprint"


class TestVLAConfig:
    _MAPS = {
        "joint_names_map": {"shoulder_pan.pos": "joint1"},
        "camera_inputs_map": {"front": {"name": "camera", "msg_type": "Image"}},
    }

    def test_construction_defaults(self):
        c = VLAConfig(**self._MAPS)
        assert c.aggregate_fn_name == "latest_only"
        # main action loop runs at the observation sending rate
        assert c.loop_rate == c.observation_sending_rate

    def test_aggregate_preset_selectable(self):
        c = VLAConfig(**self._MAPS, aggregate_fn_name="weighted_average")
        assert c.aggregate_fn_name == "weighted_average"

    def test_rates_must_be_positive(self):
        with pytest.raises(ValueError):
            VLAConfig(**self._MAPS, observation_sending_rate=0.0)
        with pytest.raises(ValueError):
            VLAConfig(**self._MAPS, action_sending_rate=0.0)

    def test_policy_action_units_default(self):
        c = VLAConfig(**self._MAPS)
        assert c.policy_action_units == "radians"

    def test_serialization_round_trip(self):
        """Multiprocess launch path: config survives a JSON round trip."""
        config = VLAConfig(
            **self._MAPS,
            aggregate_fn_name="conservative",
            policy_action_units="normalized",
        )
        rebuilt = VLAConfig(**json.loads(config.to_json()))
        assert rebuilt.aggregate_fn_name == "conservative"
        assert rebuilt.policy_action_units == "normalized"
        assert rebuilt.joint_names_map == {"shoulder_pan.pos": "joint1"}

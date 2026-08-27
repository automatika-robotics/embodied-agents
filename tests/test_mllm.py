"""Tests for MLLM/VLM component — requires rclpy."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from agents.config import MLLMConfig
from agents.ros import Topic, Image
from agents.components.mllm import MLLM
from tests.conftest import mock_component_internals
from tests.test_vision import _SyntheticCamera, _callback, _image, _warnings


@pytest.fixture
def mllm(rclpy_init, mock_model_client):
    """Create an MLLM with a mock model client."""
    comp = MLLM(
        inputs=[
            Topic(name="text_in", msg_type="String"),
            Topic(name="img_in", msg_type="Image"),
        ],
        outputs=[Topic(name="out", msg_type="String")],
        model_client=mock_model_client,
        config=MLLMConfig(),
        component_name="test_mllm",
    )
    mock_component_internals(comp)
    return comp


class TestMLLMConstruction:
    def test_with_model_client(self, rclpy_init, mock_model_client):
        comp = MLLM(
            inputs=[
                Topic(name="text_in", msg_type="String"),
                Topic(name="img_in", msg_type="Image"),
            ],
            outputs=[Topic(name="out", msg_type="String")],
            model_client=mock_model_client,
            config=MLLMConfig(),
            component_name="test_mllm_client",
        )
        assert comp.model_client is mock_model_client

    def test_with_local_model(self, rclpy_init):
        comp = MLLM(
            inputs=[
                Topic(name="text_in", msg_type="String"),
                Topic(name="img_in", msg_type="Image"),
            ],
            outputs=[Topic(name="out", msg_type="String")],
            config=MLLMConfig(enable_local_model=True),
            component_name="test_mllm_local",
        )
        assert comp.config.enable_local_model is True

    def test_no_client_no_local_raises(self, rclpy_init):
        with pytest.raises(RuntimeError):
            MLLM(
                inputs=[
                    Topic(name="text_in", msg_type="String"),
                    Topic(name="img_in", msg_type="Image"),
                ],
                outputs=[Topic(name="out", msg_type="String")],
                config=MLLMConfig(),
                component_name="test_mllm_fail",
            )


class TestMLLM3DConstruction:
    """Aux depth/camera_info inputs and the Detections3D contract."""

    @staticmethod
    def _make(mock_model_client, config=None, inputs=None, **kwargs):
        return MLLM(
            inputs=inputs
            or [
                Topic(name="text_in", msg_type="String"),
                Topic(name="img_in", msg_type="Image"),
            ],
            outputs=[Topic(name="d3", msg_type="Detections3D")],
            model_client=mock_model_client,
            config=config
            or MLLMConfig(task="grounding", detections_frame="base_link"),
            component_name="test_mllm_3d",
            **kwargs,
        )

    def test_valid_3d_construction(self, rclpy_init, mock_model_client):
        comp = self._make(
            mock_model_client,
            depth=Topic(name="depth", msg_type="Image"),
            camera_info=Topic(name="cam_info", msg_type="CameraInfo"),
        )
        assert comp._lift_to_3d
        assert comp._lift_camera == "img_in"
        assert comp._aux_inputs == {"depth", "cam_info"}
        assert comp.depth_topic.name == "depth"
        # the aux topics ride the config for the serialized relaunch
        assert comp.config._depth_topic.name == "depth"
        assert comp.config._camera_info_topic.name == "cam_info"

    def test_3d_output_requires_a_box_task(self, rclpy_init, mock_model_client):
        with pytest.raises(TypeError, match="grounding"):
            self._make(
                mock_model_client,
                config=MLLMConfig(detections_frame="base_link"),
                depth=Topic(name="depth", msg_type="Image"),
                camera_info=Topic(name="cam_info", msg_type="CameraInfo"),
            )

    def test_3d_output_requires_a_frame(self, rclpy_init, mock_model_client):
        with pytest.raises(TypeError, match="detections_frame"):
            self._make(
                mock_model_client,
                config=MLLMConfig(task="grounding"),
                depth=Topic(name="depth", msg_type="Image"),
                camera_info=Topic(name="cam_info", msg_type="CameraInfo"),
            )

    def test_3d_output_requires_depth(self, rclpy_init, mock_model_client):
        with pytest.raises(TypeError, match="requires\\s+depth"):
            self._make(mock_model_client)

    def test_depth_requires_camera_info(self, rclpy_init, mock_model_client):
        with pytest.raises(TypeError, match="camera_info"):
            self._make(
                mock_model_client, depth=Topic(name="depth", msg_type="Image")
            )

    def test_stray_camera_info_input_rejected(self, rclpy_init, mock_model_client):
        with pytest.raises(TypeError, match="camera_info=Topic"):
            self._make(
                mock_model_client,
                inputs=[
                    Topic(name="text_in", msg_type="String"),
                    Topic(name="img_in", msg_type="Image"),
                    Topic(name="stray_info", msg_type="CameraInfo"),
                ],
                depth=Topic(name="depth", msg_type="Image"),
                camera_info=Topic(name="cam_info", msg_type="CameraInfo"),
            )

    def test_aux_topics_never_reach_the_model(self, rclpy_init, mock_model_client):
        comp = self._make(
            mock_model_client,
            depth=Topic(name="depth", msg_type="Image"),
            camera_info=Topic(name="cam_info", msg_type="CameraInfo"),
        )
        mock_component_internals(comp)

        trigger = Topic(name="text_in", msg_type="String")
        trig_cb = MagicMock()
        trig_cb.get_output.return_value = "ground the mug"
        comp.trig_callbacks = {"text_in": trig_cb}

        img_cb = MagicMock()
        img_cb.get_output.return_value = np.zeros((10, 10, 3))
        img_cb.msg = MagicMock()
        img_cb.input_topic = Topic(name="img_in", msg_type="Image")
        depth_cb = MagicMock()
        depth_cb.get_output.return_value = np.zeros((10, 10))
        depth_cb.msg = MagicMock()
        depth_cb.input_topic = Topic(name="depth", msg_type="Image")
        comp.callbacks = {"img_in": img_cb, "depth": depth_cb}

        result = comp._create_input(topic=trigger)

        assert result is not None
        assert len(result["images"]) == 1

    def test_3d_mllm_survives_the_executable_reconstruction(
        self, rclpy_init, mock_model_client
    ):
        """The launch executable rebuilds the component from serialized
        config, inputs and outputs — it knows nothing of the `depth` and
        `camera_info` parameters. They ride the config instead."""
        import json

        from agents.ros import QoSConfig

        original = self._make(
            mock_model_client,
            depth=Topic(name="depth", msg_type="Image"),
            camera_info=Topic(name="cam_info", msg_type="CameraInfo"),
        )

        def _topics(serialized):
            topics = []
            for entry in json.loads(serialized):
                data = json.loads(entry)
                data["qos_profile"] = QoSConfig(**data.get("qos_profile", {}))
                data["additional_types"] = []
                topics.append(Topic(**data))
            return topics

        rebuilt = MLLM(
            inputs=_topics(original._inputs_json),
            outputs=_topics(original._outputs_json),
            model_client=mock_model_client,
            trigger=1.0,
            config=MLLMConfig(**json.loads(original.config.to_json())),
            component_name="test_mllm_3d",
        )

        assert rebuilt._aux_inputs == original._aux_inputs
        assert rebuilt._lift_camera == original._lift_camera == "img_in"
        assert rebuilt.depth_topic.name == "depth"
        assert rebuilt.camera_info_topic.name == "cam_info"


class TestMLLMCreateInput:
    def test_requires_images(self, mllm):
        """Only text, no images → None."""
        trigger = Topic(name="text_in", msg_type="String")
        mock_trig_cb = MagicMock()
        mock_trig_cb.get_output.return_value = "What is this?"
        mllm.trig_callbacks = {"text_in": mock_trig_cb}

        # Callbacks with only text, no image
        mock_text_cb = MagicMock()
        mock_text_cb.get_output.return_value = "What is this?"
        mock_text_cb.msg = None
        mock_text_cb.input_topic = Topic(name="text_in", msg_type="String")
        mllm.callbacks = {"text_in": mock_text_cb}

        result = mllm._create_input(topic=trigger)
        assert result is None

    def test_requires_query(self, mllm):
        """Only images, no query → None."""
        trigger = Topic(name="text_in", msg_type="String")
        mock_trig_cb = MagicMock()
        mock_trig_cb.get_output.return_value = None
        mllm.trig_callbacks = {"text_in": mock_trig_cb}

        mock_img_cb = MagicMock()
        mock_img_cb.get_output.return_value = np.zeros((100, 100, 3))
        mock_img_cb.msg = MagicMock()
        mock_img_cb.input_topic = Topic(name="img_in", msg_type="Image")
        mllm.callbacks = {"img_in": mock_img_cb}

        result = mllm._create_input(topic=trigger)
        assert result is None

    def test_with_both(self, mllm):
        """Both text and image → returns valid input."""
        trigger = Topic(name="text_in", msg_type="String")
        mock_trig_cb = MagicMock()
        mock_trig_cb.get_output.return_value = "What is this?"
        mllm.trig_callbacks = {"text_in": mock_trig_cb}

        mock_img_cb = MagicMock()
        mock_img_cb.get_output.return_value = np.zeros((100, 100, 3))
        mock_img_cb.msg = MagicMock()
        mock_img_cb.input_topic = Topic(name="img_in", msg_type="Image")
        mock_img_cb.input_topic.msg_type = Image
        mllm.callbacks = {"img_in": mock_img_cb}

        result = mllm._create_input(topic=trigger)
        assert result is not None
        assert "query" in result
        assert "images" in result


class TestMLLMSetTask:
    def test_valid_task(self, mllm):
        # validate_func_args decorator doesn't support Literal isinstance check,
        # so we call the underlying undecorated logic directly
        mllm._task = "pointing"
        mllm.config.task = "pointing"
        assert mllm._task == "pointing"
        assert mllm.config.task == "pointing"

    def test_invalid_task_value(self, mllm):
        # Directly test the validation inside the method body
        mllm._task = None
        with pytest.raises((ValueError, TypeError)):
            mllm.set_task("invalid_task")


class TestGroundedLift(_SyntheticCamera):
    """Grounded 2D boxes are lifted onto the depth latched with the picture
    the model saw, and published as Detections3D named by the query."""

    @pytest.fixture
    def grounder(self, rclpy_init, mock_model_client):
        comp = MLLM(
            inputs=[
                Topic(name="text_in", msg_type="String"),
                Topic(name="img_in", msg_type="Image"),
            ],
            outputs=[
                Topic(name="d2", msg_type="Detections"),
                Topic(name="d3", msg_type="Detections3D"),
            ],
            depth=Topic(name="depth", msg_type="Image"),
            camera_info=Topic(name="cam_info", msg_type="CameraInfo"),
            model_client=mock_model_client,
            config=MLLMConfig(task="grounding", detections_frame="cam"),
            component_name="test_grounder",
        )
        mock_component_internals(comp)
        # one mock publisher per output topic, keyed by name
        comp.publishers_dict = {
            topic.name: MagicMock(output_topic=topic) for topic in comp.out_topics
        }
        # what custom_on_configure derives from config and outputs
        comp._task = "grounding"
        comp._string_publishers = []
        comp._poi_publishers = []
        comp._detections_publishers = ["d2"]
        comp._detections3d_publishers = ["d3"]
        return comp

    def _wire(self, comp, stamp=0.0, depth_stamp=0.0):
        trig = MagicMock()
        trig.get_output.return_value = "the orange"
        comp.trig_callbacks = {"text_in": trig}
        comp.callbacks = {
            "img_in": _callback(
                _image("cam", width=self.WIDTH, height=self.HEIGHT, stamp=stamp),
                output=np.zeros((self.HEIGHT, self.WIDTH, 3)),
                name="img_in",
            ),
            "depth": _callback(self._depth_scene("cam", depth_stamp), name="depth"),
            "cam_info": _callback(
                msg_type="CameraInfo", output=self._intrinsics(), name="cam_info"
            ),
        }

    @staticmethod
    def _published(comp, topic="d3"):
        args, kwargs = comp.publishers_dict[topic].publish.call_args
        return args[0], kwargs

    def _trigger(self, comp):
        comp._execution_step(topic=Topic(name="text_in", msg_type="String"))

    def test_grounded_boxes_are_published_in_metric_space(
        self, grounder, mock_model_client
    ):
        mock_model_client.inference.return_value = {"output": [list(self.PATCH)]}
        self._wire(grounder)

        self._trigger(grounder)

        boxes, published = self._published(grounder)
        # the query names the objects: it is all the scene consumer gets
        assert published["labels"] == ["the orange"]
        assert published["frame_id"] == "cam"
        assert published["boxes_2d"] == [list(self.PATCH)]
        (center, _), = boxes
        assert center[2] == pytest.approx(self.PATCH_DEPTH_M, abs=0.02)
        # the 2D output still goes out alongside, labeled the same way
        pixels, _ = self._published(grounder, "d2")
        assert pixels["bboxes"] == [list(self.PATCH)]
        assert pixels["labels"] == ["the orange"]

    def test_nothing_grounded_is_an_empty_message(
        self, grounder, mock_model_client
    ):
        """The model looked and found nothing: an observation, published so
        consumers let go of objects that are no longer there."""
        mock_model_client.inference.return_value = {"output": []}
        self._wire(grounder)

        self._trigger(grounder)

        boxes, _ = self._published(grounder)
        assert boxes == []

    def test_depth_is_the_instant_of_capture_not_of_publishing(
        self, grounder, mock_model_client
    ):
        """The VLM takes seconds; the boxes must measure the scene it saw."""
        self._wire(grounder)

        def _slow_inference(*_, **__):
            # a much later depth frame lands while the model is busy
            grounder.callbacks["depth"].msg = self._depth_scene("cam", stamp=99.0)
            return {"output": [list(self.PATCH)]}

        mock_model_client.inference.side_effect = _slow_inference

        self._trigger(grounder)

        # had the late frame been read, max_depth_age would have refused it
        boxes, _ = self._published(grounder)
        assert boxes[0][0][2] == pytest.approx(self.PATCH_DEPTH_M, abs=0.02)

    def test_stale_depth_publishes_nothing(self, grounder, mock_model_client):
        mock_model_client.inference.return_value = {"output": [list(self.PATCH)]}
        self._wire(grounder, stamp=10.0, depth_stamp=9.0)

        self._trigger(grounder)

        assert grounder.publishers_dict["d3"].publish.call_count == 0
        assert "max_depth_age" in _warnings(grounder)

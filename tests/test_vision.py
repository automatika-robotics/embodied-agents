"""Tests for the Vision component — requires rclpy."""

from unittest.mock import MagicMock, PropertyMock

import pytest

from agents.components.vision import Vision
from agents.config import VisionConfig
from agents.ros import Detections, DetectionsMultiSource, Topic
from tests.conftest import mock_component_internals


def _image(frame_id="camera_optical_frame"):
    """A minimal ROS image carrying a frame."""
    from sensor_msgs.msg import Image as ROSImage

    image = ROSImage()
    image.header.frame_id = frame_id
    image.height, image.width = 2, 2
    image.encoding = "rgb8"
    image.step = 6
    image.data = bytes(12)
    return image


def _callback(msg=None, output=None, msg_type="Image", name="topic"):
    """A stand-in for an input callback, carrying the topic it subscribes to."""
    callback = MagicMock()
    callback.input_topic = Topic(name=name, msg_type=msg_type)
    callback.msg = msg
    callback.get_output.return_value = output
    return callback


@pytest.fixture
def vision(rclpy_init, mock_model_client):
    # Vision rejects clients that do not serve a VisionModel
    type(mock_model_client).model_type = PropertyMock(return_value="VisionModel")
    comp = Vision(
        inputs=[Topic(name="image", msg_type="Image")],
        outputs=[Topic(name="detections", msg_type="Detections")],
        model_client=mock_model_client,
        config=VisionConfig(),
        component_name="test_vision",
    )
    mock_component_internals(comp)
    return comp


class TestDetectionFrames:
    """Detections must carry the frame of the camera that made them —
    without it they cannot be placed in the world."""

    def test_publish_passes_source_frame(self, vision, mock_model_client):
        mock_model_client.inference.return_value = {
            "output": [{"bboxes": [[0, 0, 1, 1]], "labels": ["cup"], "scores": [0.9]}]
        }
        callback = MagicMock()
        callback.msg = _image("front_camera_optical")
        callback.get_output.return_value = [[0, 0, 0]]
        vision.callbacks = {"image": callback}
        vision.trig_callbacks = {}
        vision.run_type = vision.run_type  # keep whatever mock_component_internals set

        vision._execution_step()

        publish_kwargs = vision.publishers_dict["out"].publish.call_args[1]
        assert publish_kwargs["frame_id"] == "front_camera_optical"

    def test_source_frame_from_rgbd(self, vision):
        rgbd = MagicMock()
        rgbd.rgb = _image("rgbd_optical")
        vision._images = [rgbd]
        assert vision._source_frame() == "rgbd_optical"

    def test_source_frame_none_without_header(self, vision):
        vision._images = [MagicMock(spec=[])]
        assert vision._source_frame() is None

    def test_multi_camera_leaves_outer_frame_unset(self, vision):
        """A single frame would name only one of the cameras — the nested
        per-source detections carry the frames instead."""
        vision._images = [_image("left_camera"), _image("right_camera")]
        assert vision._source_frame() is None

    def test_no_images_has_no_frame(self, vision):
        vision._images = []
        assert vision._source_frame() is None


class TestDetectionSet:
    """Which image inputs reach the model. The rest stay subscribed for
    take_picture and record_video rather than being inferred on and thrown
    away, which is what used to happen."""

    @staticmethod
    def _build(mock_model_client, inputs, outputs, trigger=1.0):
        type(mock_model_client).model_type = PropertyMock(return_value="VisionModel")
        component = Vision(
            inputs=inputs,
            outputs=outputs,
            model_client=mock_model_client,
            config=VisionConfig(),
            component_name="test_set",
            trigger=trigger,
        )
        return mock_component_internals(component)

    @staticmethod
    def _cameras(*names):
        return [Topic(name=name, msg_type="Image") for name in names]

    def test_timed_single_source_detects_on_one_camera(
        self, rclpy_init, mock_model_client
    ):
        component = self._build(
            mock_model_client,
            self._cameras("front", "rear"),
            [Topic(name="detections", msg_type="Detections")],
        )
        assert component._inference_set == ["front"]
        assert component._spectators == ["rear"]

    def test_a_spectator_camera_never_reaches_the_model(
        self, rclpy_init, mock_model_client
    ):
        """Inferring on it and discarding the result costs a full model pass
        per tick."""
        component = self._build(
            mock_model_client,
            self._cameras("front", "rear"),
            [Topic(name="detections", msg_type="Detections")],
        )
        component.callbacks = {
            "front": _callback(_image("front_optical"), [[1]], name="front"),
            "rear": _callback(_image("rear_optical"), [[2]], name="rear"),
        }
        component.trig_callbacks = {}

        assert component._create_input()["images"] == [[[1]]]

    def test_it_is_named_rather_than_quietly_ignored(
        self, rclpy_init, mock_model_client, monkeypatch
    ):
        component = self._build(
            mock_model_client,
            self._cameras("front", "rear"),
            [Topic(name="detections", msg_type="Detections")],
        )
        monkeypatch.setattr(
            "agents.components.model_component.ModelComponent.custom_on_configure",
            lambda _: None,
        )

        component.custom_on_configure()

        warning = " ".join(
            str(call[0][0]) for call in component.get_logger().warning.call_args_list
        )
        assert "rear" in warning and "take_picture" in warning

    def test_timed_multi_source_detects_on_all_cameras(
        self, rclpy_init, mock_model_client
    ):
        component = self._build(
            mock_model_client,
            self._cameras("front", "rear"),
            [Topic(name="detections", msg_type="DetectionsMultiSource")],
        )
        assert component._inference_set == ["front", "rear"]
        assert component._spectators == []

    def test_single_source_alongside_multi_source_is_refused(
        self, rclpy_init, mock_model_client
    ):
        """The multi source output pulls both cameras into the same pass, and
        a Detections topic cannot describe both."""
        with pytest.raises(TypeError, match="MultiSource"):
            self._build(
                mock_model_client,
                self._cameras("front", "rear"),
                [
                    Topic(name="all", msg_type="DetectionsMultiSource"),
                    Topic(name="one", msg_type="Detections"),
                ],
            )

    def test_triggered_cameras_are_the_detection_set(
        self, rclpy_init, mock_model_client
    ):
        """A tick reads only the topic that fired, so several triggers still
        mean one picture per tick and a single source output stays valid."""
        cameras = self._cameras("front", "rear", "spare")
        component = self._build(
            mock_model_client,
            cameras,
            [Topic(name="detections", msg_type="Detections")],
            trigger=cameras[:2],
        )
        assert component._inference_set == ["front", "rear"]
        assert component._spectators == ["spare"]

    def test_trackers_are_allocated_per_camera_detected_on(
        self, rclpy_init, mock_model_client
    ):
        type(mock_model_client).model_type = PropertyMock(return_value="VisionModel")
        mock_model_client._model = MagicMock(setup_trackers=True)
        Vision(
            inputs=self._cameras("front", "rear"),
            outputs=[Topic(name="detections", msg_type="Detections")],
            model_client=mock_model_client,
            config=VisionConfig(),
            component_name="test_trackers",
        )
        assert mock_model_client._model._num_trackers == 1


class TestPublishRouting:
    """Inference returns one set of detections per image; each output topic
    gets the shape it can carry."""

    def test_single_source_gets_one_cameras_detections(
        self, vision, mock_model_client
    ):
        mock_model_client.inference.return_value = {
            "output": [{"bboxes": [[0, 0, 1, 1]], "labels": ["cup"], "scores": [0.9]}]
        }
        vision.callbacks = {"image": _callback(_image("front"), [[0]], name="image")}
        vision.trig_callbacks = {}

        vision._execution_step()

        published = vision.publishers_dict["out"].publish.call_args[0][0]
        assert published["labels"] == ["cup"]

    def test_multi_source_gets_the_whole_list(self, rclpy_init, mock_model_client):
        type(mock_model_client).model_type = PropertyMock(return_value="VisionModel")
        component = mock_component_internals(
            Vision(
                inputs=[
                    Topic(name="front", msg_type="Image"),
                    Topic(name="rear", msg_type="Image"),
                ],
                outputs=[Topic(name="all", msg_type="DetectionsMultiSource")],
                model_client=mock_model_client,
                config=VisionConfig(),
                component_name="test_multi",
            )
        )
        mock_model_client.inference.return_value = {
            "output": [
                {"bboxes": [[0, 0, 1, 1]], "labels": ["cup"], "scores": [0.9]},
                {"bboxes": [[1, 1, 2, 2]], "labels": ["bowl"], "scores": [0.8]},
            ]
        }
        component.callbacks = {
            "front": _callback(_image("front_optical"), [[1]], name="front"),
            "rear": _callback(_image("rear_optical"), [[2]], name="rear"),
        }
        component.trig_callbacks = {}

        component._execution_step()

        published = component.publishers_dict["out"].publish.call_args[0][0]
        assert [d["labels"] for d in published] == [["cup"], ["bowl"]]


class TestConvertHeaders:
    """The per-source detections nested in a multi-source message need their
    own frames — the outer header can only name one camera."""

    def test_detections_convert_copies_header(self):
        message = Detections.convert(
            {"bboxes": [[0, 0, 2, 2]], "labels": ["cup"], "scores": [0.8]},
            _image("left_camera"),
        )
        assert message.header.frame_id == "left_camera"

    def test_detections_refuses_several_cameras(self):
        """Silently keeping the first camera's detections and dropping the
        rest is what this message type used to do."""
        with pytest.raises(TypeError, match="MultiSource"):
            Detections.convert(
                [{"bboxes": [[0, 0, 2, 2]], "labels": ["cup"], "scores": [0.8]}],
                [_image("left_camera")],
            )

    def test_multi_source_nests_per_camera_frames(self):
        output = [
            {"bboxes": [[0, 0, 1, 1]], "labels": ["cup"], "scores": [0.8]},
            {"bboxes": [[1, 1, 2, 2]], "labels": ["bowl"], "scores": [0.7]},
        ]
        message = DetectionsMultiSource.convert(
            output, [_image("left_camera"), _image("right_camera")]
        )
        assert [d.header.frame_id for d in message.detections] == [
            "left_camera",
            "right_camera",
        ]

    def test_convert_without_image_is_unchanged(self):
        message = Detections.convert(
            {"bboxes": [[0, 0, 1, 1]], "labels": ["cup"], "scores": [0.8]}
        )
        assert message.header.frame_id == ""
        assert message.labels == ["cup"]


class TestDetections3DMessage:
    """The 3D detection message and its conversion."""

    @staticmethod
    def _boxes():
        return [((0.3, 0.0, 0.15), (0.06, 0.06, 0.08))]

    def test_convert_round_trip(self):
        from agents.ros import Detections3D

        message = Detections3D.convert(
            self._boxes(),
            labels=["orange"],
            scores=[0.87],
            depth_validity=[0.64],
            source_frame="front_optical",
        )
        assert message.labels == ["orange"]
        assert list(message.scores) == [0.87]
        assert list(message.depth_validity) == [0.64]
        assert message.source_frame == "front_optical"
        box = message.boxes[0]
        assert (box.center.position.x, box.center.position.z) == (0.3, 0.15)
        assert (box.size.x, box.size.y, box.size.z) == (0.06, 0.06, 0.08)
        # a bare Pose has w=0, which is not a valid rotation
        assert box.center.orientation.w == 1.0

    def test_convert_without_metadata(self):
        from agents.ros import Detections3D

        message = Detections3D.convert(self._boxes())
        assert len(message.boxes) == 1
        assert message.labels == [] and list(message.scores) == []

    def test_box_maps_onto_ros_and_moveit_types(self):
        """The message is shaped so neither perception nor planning needs a
        conversion layer."""
        vision_msgs = pytest.importorskip("vision_msgs.msg")
        shape_msgs = pytest.importorskip("shape_msgs.msg")
        from agents.ros import Detections3D

        box = Detections3D.convert(self._boxes()).boxes[0]

        bounding_box = vision_msgs.BoundingBox3D()
        bounding_box.center = box.center
        bounding_box.size = box.size
        assert bounding_box.size.x == 0.06

        primitive = shape_msgs.SolidPrimitive(
            type=shape_msgs.SolidPrimitive.BOX,
            dimensions=[box.size.x, box.size.y, box.size.z],
        )
        assert list(primitive.dimensions) == [0.06, 0.06, 0.08]

    def test_callback_gives_context_by_default_and_the_message_on_request(self):
        """Prompts and memory want the classes; a planner wants the boxes."""
        from agents.callbacks import Detections3DCallback
        from agents.ros import Detections3D, Topic

        callback = Detections3DCallback(Topic(name="d3", msg_type="Detections3D"))
        assert callback.get_output() is None
        assert callback.get_output(get_msg=True) is None
        assert "No objects" in callback._get_ui_content()

        message = Detections3D.convert(self._boxes(), labels=["orange"])
        message.header.frame_id = "base_link"
        callback.msg = message

        assert callback.get_output() == "1 orange"
        assert callback.get_output(get_msg=True) is message

        content = callback._get_ui_content()
        assert "orange" in content and "base_link" in content

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


class TestConvertHeaders:
    """The per-source detections nested in a multi-source message need their
    own frames — the outer header can only name one camera."""

    def test_detections_convert_copies_header(self):
        message = Detections.convert(
            [{"bboxes": [[0, 0, 2, 2]], "labels": ["cup"], "scores": [0.8]}],
            [_image("left_camera")],
        )
        assert message.header.frame_id == "left_camera"

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
            [{"bboxes": [[0, 0, 1, 1]], "labels": ["cup"], "scores": [0.8]}], []
        )
        assert message.header.frame_id == ""
        assert message.labels == ["cup"]

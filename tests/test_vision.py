"""Tests for the Vision component — requires rclpy."""

from unittest.mock import MagicMock, PropertyMock

import pytest

from agents.components.vision import Vision
from agents.config import VisionConfig
from agents.ros import Detections, DetectionsMultiSource, Topic
from tests.conftest import mock_component_internals


def _image(frame_id="camera_optical_frame", width=2, height=2, stamp=0.0):
    """A minimal ROS image carrying a frame."""
    from sensor_msgs.msg import Image as ROSImage

    image = ROSImage()
    image.header.frame_id = frame_id
    image.header.stamp.sec = int(stamp)
    image.header.stamp.nanosec = int((stamp % 1) * 1e9)
    image.height, image.width = height, width
    image.encoding = "rgb8"
    image.step = width * 3
    image.data = bytes(width * height * 3)
    return image


def _callback(msg=None, output=None, msg_type="Image", name="topic"):
    """A stand-in for an input callback, carrying the topic it subscribes to."""
    callback = MagicMock()
    callback.input_topic = Topic(name=name, msg_type=msg_type)
    callback.msg = msg
    callback.get_output.return_value = output
    return callback


def _warnings(component) -> str:
    return " ".join(
        str(call[0][0]) for call in component.get_logger().warning.call_args_list
    )


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


class Test3DContract:
    """Asking for a Detections3D output is what turns lifting on, so the
    component says up front when it was not given what that takes: one camera,
    depth registered to its pictures, that depth's calibration, and a frame to
    report boxes in."""

    @staticmethod
    def _build(mock_model_client, inputs, config=None, outputs=None, **kwargs):
        type(mock_model_client).model_type = PropertyMock(return_value="VisionModel")
        return Vision(
            inputs=inputs,
            outputs=outputs or [Topic(name="d3", msg_type="Detections3D")],
            model_client=mock_model_client,
            config=config or VisionConfig(detections_frame="base_link"),
            component_name="test_3d",
            **kwargs,
        )

    def test_an_rgbd_input_carries_everything(self, rclpy_init, mock_model_client):
        """Depth registered to the picture, plus both calibrations, in one
        message — so it needs no other topic."""
        component = self._build(
            mock_model_client, [Topic(name="rgbd", msg_type="RGBD")]
        )
        assert component._lift_to_3d and component._inference_set == ["rgbd"]

    def test_a_named_depth_topic_is_accepted(self, rclpy_init, mock_model_client):
        """Every stereo camera other than RealSense publishes its registered
        depth on its own topic."""
        component = self._build(
            mock_model_client,
            [Topic(name="image", msg_type="Image")],
            depth=Topic(name="depth", msg_type="Image"),
            camera_info=Topic(name="camera_info", msg_type="CameraInfo"),
        )
        # both become inputs, but neither is a picture to detect on
        assert component._aux_inputs == {"depth", "camera_info"}
        assert component._inference_set == ["image"]
        assert component._spectators == []
        assert {"depth", "camera_info"} <= {t.name for t in component.in_topics}

    def test_depth_is_required(self, rclpy_init, mock_model_client):
        """An image topic cannot be guessed to be depth."""
        with pytest.raises(TypeError, match="requires depth"):
            self._build(mock_model_client, [Topic(name="image", msg_type="Image")])

    def test_depth_needs_its_calibration(self, rclpy_init, mock_model_client):
        with pytest.raises(TypeError, match="camera_info"):
            self._build(
                mock_model_client,
                [Topic(name="image", msg_type="Image")],
                depth=Topic(name="depth", msg_type="Image"),
            )

    def test_depth_cannot_be_compressed(self, rclpy_init, mock_model_client):
        with pytest.raises(TypeError, match="uncompressed"):
            self._build(
                mock_model_client,
                [Topic(name="image", msg_type="Image")],
                depth=Topic(name="depth", msg_type="CompressedImage"),
                camera_info=Topic(name="camera_info", msg_type="CameraInfo"),
            )

    def test_a_frame_to_report_in_is_required(self, rclpy_init, mock_model_client):
        """Boxes are axis aligned in the frame they are measured in, so it is
        chosen before they exist."""
        with pytest.raises(TypeError, match="detections_frame"):
            self._build(
                mock_model_client,
                [Topic(name="rgbd", msg_type="RGBD")],
                config=VisionConfig(),
            )

    def test_the_first_of_several_cameras_is_lifted(
        self, rclpy_init, mock_model_client
    ):
        """Only one camera's depth and calibration are given, so the rest are
        left for the component actions, as with 2D outputs."""
        component = self._build(
            mock_model_client,
            [
                Topic(name="left", msg_type="RGBD"),
                Topic(name="right", msg_type="RGBD"),
            ],
        )
        assert component._lift_camera == "left"
        assert component._inference_set == ["left"]
        assert component._spectators == ["right"]

    def test_an_rgbd_wins_over_a_plain_camera_whatever_the_order(
        self, rclpy_init, mock_model_client
    ):
        """An RGBD frame identifies its own camera, so it needs no ordering
        convention to be chosen."""
        component = self._build(
            mock_model_client,
            [
                Topic(name="spare", msg_type="Image"),
                Topic(name="rgbd", msg_type="RGBD"),
            ],
        )
        assert component._lift_camera == "rgbd"
        assert component._inference_set == ["rgbd"]
        assert component._spectators == ["spare"]

    def test_the_lifted_camera_is_named_when_others_are_inferenced_on(
        self, rclpy_init, mock_model_client, monkeypatch
    ):
        """A multi source output pulls every camera into the pass, but only one
        of them can be placed in space."""
        component = self._build(
            mock_model_client,
            [
                Topic(name="rgbd", msg_type="RGBD"),
                Topic(name="spare", msg_type="Image"),
            ],
            outputs=[
                Topic(name="d3", msg_type="Detections3D"),
                Topic(name="all", msg_type="DetectionsMultiSource"),
            ],
        )
        mock_component_internals(component)
        monkeypatch.setattr(
            "agents.components.model_component.ModelComponent.custom_on_configure",
            lambda _: None,
        )
        assert component._inference_set == ["rgbd", "spare"]

        component.custom_on_configure()

        warning = " ".join(
            str(call[0][0]) for call in component.get_logger().warning.call_args_list
        )
        assert "rgbd" in warning and "spare" in warning and "3D" in warning

    def test_camera_info_may_override_what_an_rgbd_carries(
        self, rclpy_init, mock_model_client
    ):
        component = self._build(
            mock_model_client,
            [Topic(name="rgbd", msg_type="RGBD")],
            camera_info=Topic(name="camera_info", msg_type="CameraInfo"),
        )
        assert component.camera_info_topic.name == "camera_info"
        assert component._inference_set == ["rgbd"]

    def test_camera_info_in_inputs_is_refused(self, rclpy_init, mock_model_client):
        """Depth cannot be told from a camera in `inputs`, so neither is looked
        for there: both have exactly one way in."""
        with pytest.raises(TypeError, match="camera_info=Topic"):
            self._build(
                mock_model_client,
                [
                    Topic(name="rgbd", msg_type="RGBD"),
                    Topic(name="camera_info", msg_type="CameraInfo"),
                ],
            )

    def test_depth_cannot_be_the_trigger(self, rclpy_init, mock_model_client):
        depth = Topic(name="depth", msg_type="Image")
        with pytest.raises(TypeError, match="trigger"):
            self._build(
                mock_model_client,
                [Topic(name="image", msg_type="Image")],
                depth=depth,
                camera_info=Topic(name="camera_info", msg_type="CameraInfo"),
                trigger=depth,
            )

    def test_depth_without_a_3d_output_is_ignored_and_said_so(
        self, rclpy_init, mock_model_client, monkeypatch
    ):
        """Unused config rather than an impossibility, so the component builds
        and says what it is leaving out."""
        component = self._build(
            mock_model_client,
            [Topic(name="image", msg_type="Image")],
            outputs=[Topic(name="detections", msg_type="Detections")],
            depth=Topic(name="depth", msg_type="Image"),
            camera_info=Topic(name="camera_info", msg_type="CameraInfo"),
        )
        # nothing reads them, so nothing subscribes to them either
        assert {t.name for t in component.in_topics} == {"image"}

        mock_component_internals(component)
        monkeypatch.setattr(
            "agents.components.model_component.ModelComponent.custom_on_configure",
            lambda _: None,
        )
        component.custom_on_configure()

        warning = " ".join(
            str(call[0][0]) for call in component.get_logger().warning.call_args_list
        )
        assert "depth" in warning and "camera_info" in warning

    def test_2d_only_components_are_untouched(self, rclpy_init, mock_model_client):
        component = self._build(
            mock_model_client,
            [Topic(name="image", msg_type="Image")],
            config=VisionConfig(),
            outputs=[Topic(name="detections", msg_type="Detections")],
        )
        assert not component._lift_to_3d and component._aux_inputs == set()


class _SyntheticCamera:
    """A camera looking at a patch half a meter away on a far wall. Shared by
    the two ways depth reaches the component."""

    FX = FY = 500.0
    WIDTH, HEIGHT = 200, 120
    PATCH = (100, 40, 140, 60)
    PATCH_DEPTH_M = 0.5

    @pytest.fixture(autouse=True)
    def _needs_kompass_core(self):
        pytest.importorskip("kompass_core.vision")

    @classmethod
    def _depth_scene(cls, frame_id="cam", stamp=0.0, background_m=3.0):
        import numpy as np
        from sensor_msgs.msg import Image as ROSImage

        pixels = np.full((cls.HEIGHT, cls.WIDTH), int(background_m * 1000), np.uint16)
        x1, y1, x2, y2 = cls.PATCH
        pixels[y1:y2, x1:x2] = int(cls.PATCH_DEPTH_M * 1000)

        msg = ROSImage()
        msg.header.frame_id = frame_id
        msg.header.stamp.sec = int(stamp)
        msg.header.stamp.nanosec = int((stamp % 1) * 1e9)
        msg.height, msg.width = cls.HEIGHT, cls.WIDTH
        msg.encoding = "16UC1"
        msg.step = cls.WIDTH * 2
        msg.data = pixels.tobytes()
        return msg

    @classmethod
    def _intrinsics(cls, frame_id="cam", width=None, height=None):
        from ros_sugar.io import CameraIntrinsics

        return CameraIntrinsics(
            fx=cls.FX,
            fy=cls.FY,
            cx=cls.WIDTH / 2,
            cy=cls.HEIGHT / 2,
            width=width or cls.WIDTH,
            height=height or cls.HEIGHT,
            frame_id=frame_id,
        )

    @classmethod
    def _detections(cls, boxes=None):
        return {
            "output": [
                {
                    "bboxes": boxes if boxes is not None else [list(cls.PATCH)],
                    "labels": ["orange"],
                    "scores": [0.9],
                }
            ]
        }

    @staticmethod
    def _published(component):
        """The boxes and the fields they were published with."""
        args, kwargs = component.publishers_dict["out"].publish.call_args
        return args[0], kwargs


class TestLiftingWithADepthTopic(_SyntheticCamera):
    """Depth on its own topic, as ZED, Orbbec and OAK publish it."""

    @pytest.fixture
    def vision_3d(self, rclpy_init, mock_model_client):
        type(mock_model_client).model_type = PropertyMock(return_value="VisionModel")
        component = Vision(
            inputs=[Topic(name="image", msg_type="Image")],
            outputs=[Topic(name="d3", msg_type="Detections3D")],
            depth=Topic(name="depth", msg_type="Image"),
            camera_info=Topic(name="camera_info", msg_type="CameraInfo"),
            model_client=mock_model_client,
            config=VisionConfig(detections_frame="cam"),
            component_name="test_lift",
        )
        return mock_component_internals(component)

    def _wire(self, component, stamp=0.0, depth_stamp=0.0, intrinsics=None):
        component.callbacks = {
            "image": _callback(
                _image("cam", width=self.WIDTH, height=self.HEIGHT, stamp=stamp),
                [[0]],
                name="image",
            ),
            "depth": _callback(self._depth_scene("cam", depth_stamp), name="depth"),
            "camera_info": _callback(
                msg_type="CameraInfo",
                output=intrinsics or self._intrinsics(),
                name="camera_info",
            ),
        }
        component.trig_callbacks = {}

    def test_detections_are_published_in_metric_space(
        self, vision_3d, mock_model_client
    ):
        mock_model_client.inference.return_value = self._detections()
        self._wire(vision_3d)

        vision_3d._execution_step()

        boxes, published = self._published(vision_3d)
        assert published["labels"] == ["orange"] and published["scores"] == [0.9]
        assert published["frame_id"] == "cam"
        # published in the camera's own frame, where distance runs along z
        (center, size), = boxes
        assert center[2] == pytest.approx(self.PATCH_DEPTH_M, abs=0.02)
        assert size[0] == pytest.approx(40 * self.PATCH_DEPTH_M / self.FX, abs=0.01)
        assert published["boxes_2d"] == [list(self.PATCH)]

    def test_an_empty_scene_is_still_reported(self, vision_3d, mock_model_client):
        """Seeing nothing is an observation, and consumers need it to let go of
        objects that are no longer there."""
        mock_model_client.inference.return_value = self._detections(boxes=[])
        self._wire(vision_3d)

        vision_3d._execution_step()

        boxes, _ = self._published(vision_3d)
        assert boxes == []

    def test_depth_is_paired_when_the_picture_is_taken(
        self, vision_3d, mock_model_client
    ):
        """Which depth frame pairs with the picture must not depend on how
        long inference takes."""
        self._wire(vision_3d)

        def _slow_inference(*_, **__):
            # a much later depth frame lands while the model is busy
            vision_3d.callbacks["depth"].msg = self._depth_scene("cam", stamp=99.0)
            return self._detections()

        mock_model_client.inference.side_effect = _slow_inference

        vision_3d._execution_step()

        # had the late frame been read, max_depth_age would have refused it
        boxes, _ = self._published(vision_3d)
        assert boxes[0][0][2] == pytest.approx(self.PATCH_DEPTH_M, abs=0.02)

    def test_stale_depth_publishes_nothing(self, vision_3d, mock_model_client):
        """Silence differs from an empty message: one says the camera cannot be
        used, the other that the scene is clear."""
        mock_model_client.inference.return_value = self._detections()
        self._wire(vision_3d, stamp=10.0, depth_stamp=9.0)

        vision_3d._execution_step()

        assert vision_3d.publishers_dict["out"].publish.call_count == 0
        assert "max_depth_age" in _warnings(vision_3d)

    def test_intrinsics_for_another_resolution_are_refused(
        self, vision_3d, mock_model_client
    ):
        mock_model_client.inference.return_value = self._detections()
        self._wire(vision_3d, intrinsics=self._intrinsics(width=1280, height=720))

        vision_3d._execution_step()

        assert vision_3d.publishers_dict["out"].publish.call_count == 0
        assert "1280x720" in vision_3d.get_logger().error.call_args[0][0]

    def test_boxes_built_from_too_little_depth_are_dropped(
        self, vision_3d, mock_model_client
    ):
        vision_3d.config.min_depth_validity = 0.9
        vision_3d.config.max_depth = 5.0
        mock_model_client.inference.return_value = self._detections(
            boxes=[[80, 20, 160, 80]]
        )
        self._wire(vision_3d)
        # the wall is beyond max_depth, so only the patch itself reads back
        vision_3d.callbacks["depth"].msg = self._depth_scene("cam", background_m=9.0)

        vision_3d._execution_step()

        boxes, _ = self._published(vision_3d)
        assert boxes == []

    def test_boxes_are_transformed_into_the_configured_frame(
        self, vision_3d, mock_model_client
    ):
        """A planner works in the robot's frame, not the camera's."""
        vision_3d.config.detections_frame = "base_link"
        mock_model_client.inference.return_value = self._detections()
        self._wire(vision_3d)
        # a camera looking straight ahead, a meter up on the robot
        vision_3d.get_transform_listener = MagicMock(
            return_value=MagicMock(
                got_transform=True,
                translation=[0.0, 0.0, 1.0],
                rotation=[-0.5, 0.5, -0.5, 0.5],
            )
        )

        vision_3d._execution_step()

        boxes, published = self._published(vision_3d)
        assert published["frame_id"] == "base_link"
        (center, _), = boxes
        assert center[0] == pytest.approx(self.PATCH_DEPTH_M, abs=0.02)
        assert center[2] == pytest.approx(1.0, abs=0.02)

    def test_nothing_is_published_before_the_transform_resolves(
        self, vision_3d, mock_model_client
    ):
        vision_3d.config.detections_frame = "base_link"
        mock_model_client.inference.return_value = self._detections()
        self._wire(vision_3d)
        vision_3d.get_transform_listener = MagicMock(
            return_value=MagicMock(got_transform=False)
        )

        vision_3d._execution_step()

        assert vision_3d.publishers_dict["out"].publish.call_count == 0
        assert "has not been resolved" in _warnings(vision_3d)


class TestLiftingFromRGBD(_SyntheticCamera):
    """A RealSense publishes everything lifting needs in one message."""

    @pytest.fixture(autouse=True)
    def _needs_realsense_msgs(self):
        pytest.importorskip("realsense2_camera_msgs.msg")

    @classmethod
    def _camera_info_msg(cls, frame_id="cam"):
        from sensor_msgs.msg import CameraInfo as ROSCameraInfo

        info = ROSCameraInfo()
        info.header.frame_id = frame_id
        info.width, info.height = cls.WIDTH, cls.HEIGHT
        info.k = [cls.FX, 0.0, cls.WIDTH / 2, 0.0, cls.FY, cls.HEIGHT / 2, 0.0, 0.0, 1.0]
        return info

    @classmethod
    def _rgbd_frame(cls, frame_id="cam", stamp=0.0):
        from realsense2_camera_msgs.msg import RGBD as ROSRGBD

        frame = ROSRGBD()
        frame.header.frame_id = frame_id
        frame.rgb = _image(frame_id, width=cls.WIDTH, height=cls.HEIGHT, stamp=stamp)
        frame.depth = cls._depth_scene(frame_id, stamp)
        frame.rgb_camera_info = cls._camera_info_msg(frame_id)
        frame.depth_camera_info = cls._camera_info_msg(frame_id)
        return frame

    @pytest.fixture
    def vision_rgbd(self, rclpy_init, mock_model_client):
        type(mock_model_client).model_type = PropertyMock(return_value="VisionModel")
        component = Vision(
            inputs=[Topic(name="rgbd", msg_type="RGBD")],
            outputs=[Topic(name="d3", msg_type="Detections3D")],
            model_client=mock_model_client,
            config=VisionConfig(detections_frame="cam"),
            component_name="test_rgbd",
        )
        return mock_component_internals(component)

    def test_intrinsics_are_parsed_once_and_reused(self, vision_rgbd):
        """Intrinsics do not change in ordinary operation, so they are read
        from the captured frame whenever needed and parsed only once."""
        vision_rgbd._lift_msg = self._rgbd_frame()
        parsed = vision_rgbd._camera_intrinsics()
        assert parsed.fx == self.FX and parsed.frame_id == "cam"

        vision_rgbd._lift_msg = self._rgbd_frame(stamp=1.0)
        assert vision_rgbd._camera_intrinsics() is parsed

    def test_a_new_calibration_is_picked_up(self, vision_rgbd):
        vision_rgbd._lift_msg = self._rgbd_frame()
        vision_rgbd._camera_intrinsics()

        changed = self._rgbd_frame()
        changed.depth_camera_info.k[0] = 999.0
        vision_rgbd._lift_msg = changed
        assert vision_rgbd._camera_intrinsics().fx == 999.0

    def test_an_rgbd_frame_lifts_with_no_other_topic(
        self, vision_rgbd, mock_model_client
    ):
        mock_model_client.inference.return_value = self._detections()
        vision_rgbd.callbacks = {
            "rgbd": _callback(self._rgbd_frame(), [[0]], msg_type="RGBD", name="rgbd")
        }
        vision_rgbd.trig_callbacks = {}

        vision_rgbd._execution_step()

        boxes, published = self._published(vision_rgbd)
        assert published["frame_id"] == "cam"
        (center, _), = boxes
        assert center[2] == pytest.approx(self.PATCH_DEPTH_M, abs=0.02)


class TestSerializedRelaunch:
    """The launch executable reconstructs a component in a child process from
    its serialized config, inputs and outputs — it knows nothing of Vision's
    `depth` and `camera_info` parameters. They ride the config instead."""

    def test_3d_vision_survives_the_executable_reconstruction(
        self, rclpy_init, mock_model_client
    ):
        import json

        from agents.ros import QoSConfig

        type(mock_model_client).model_type = PropertyMock(return_value="VisionModel")
        original = Vision(
            inputs=[Topic(name="image", msg_type="Image")],
            outputs=[Topic(name="d3", msg_type="Detections3D")],
            depth=Topic(name="depth", msg_type="Image"),
            camera_info=Topic(name="cam_info", msg_type="CameraInfo"),
            model_client=mock_model_client,
            config=VisionConfig(detections_frame="base_link"),
            component_name="v",
        )

        # what scripts/executable does in the child process: config from its
        # JSON, topics from theirs, and no depth/camera_info parameters
        def _topics(serialized):
            topics = []
            for entry in json.loads(serialized):
                data = json.loads(entry)
                data["qos_profile"] = QoSConfig(**data.get("qos_profile", {}))
                data["additional_types"] = []
                topics.append(Topic(**data))
            return topics

        rebuilt = Vision(
            inputs=_topics(original._inputs_json),
            outputs=_topics(original._outputs_json),
            model_client=mock_model_client,
            trigger=1.0,
            config=VisionConfig(**json.loads(original.config.to_json())),
            component_name="v",
        )

        assert rebuilt._aux_inputs == original._aux_inputs
        assert rebuilt._lift_camera == original._lift_camera == "image"
        assert rebuilt._inference_set == original._inference_set
        assert rebuilt.depth_topic.name == "depth"
        assert rebuilt.camera_info_topic.name == "cam_info"


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

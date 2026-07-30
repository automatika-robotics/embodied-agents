"""Tests for the MotionDetector component and its motion algorithms.

Algorithm tests exercise ``agents.utils.motion`` directly on synthetic
frames and clouds (no ROS needed). Component tests cover the input/output
modality validation matrix, the motion state machine, and the image/cloud
processing paths with mocked publishers.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from agents.components.imagestovideo import VideoMessageMaker
from agents.components.motion_detection import MotionDetector
from agents.config import MotionDetectorConfig, VideoMessageMakerConfig
from agents.ros import Topic
from agents.utils import motion


# ---------------------------------------------------------------------------
# Algorithm tests: images
# ---------------------------------------------------------------------------


def _textured_frame(seed: int = 7, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size), dtype=np.uint8)


def test_frame_difference_static_vs_motion():
    frame = _textured_frame()
    assert not motion.frame_difference(frame, frame, threshold=0.3)
    moved = np.roll(frame, 8, axis=1)
    assert motion.frame_difference(frame, moved, threshold=0.3)


def test_optical_flow_static_vs_motion():
    flow_kwargs = MotionDetectorConfig().flow_kwargs
    frame = _textured_frame()
    assert not motion.optical_flow(frame, frame, 0.3, flow_kwargs)
    moved = np.roll(frame, 8, axis=1)
    assert motion.optical_flow(frame, moved, 0.3, flow_kwargs)


def test_roi_mask():
    mask = motion.roi_mask((10, 10), [(0, 0), (0, 4), (4, 4), (4, 0)])
    assert mask.shape == (10, 10)
    assert mask[2, 2] == 0  # inside the ignored polygon
    assert mask[8, 8] == 1  # outside


# ---------------------------------------------------------------------------
# Algorithm tests: point clouds
# ---------------------------------------------------------------------------


def test_crop_cloud():
    points = np.array(
        [
            [1.0, 0.0, 0.0],  # kept
            [30.0, 0.0, 0.0],  # beyond max_range
            [0.05, 0.05, 0.0],  # below min_range
            [1.0, 1.0, 5.0],  # above z_max
        ],
        dtype=np.float32,
    )
    cropped = motion.crop_cloud(
        points, min_range=0.2, max_range=20.0, z_min=-1.0, z_max=2.0
    )
    np.testing.assert_allclose(cropped, [[1.0, 0.0, 0.0]])


def test_transform_cloud():
    points = np.array([[1.0, 0.0, 0.5]], dtype=np.float32)
    position = np.array([1.0, 2.0, 3.0, np.pi / 2, 0.0])
    transformed = motion.transform_cloud(points, position)
    np.testing.assert_allclose(transformed, [[1.0, 3.0, 3.5]], atol=1e-6)


def test_apply_transform():
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    # 90 degree rotation around z + translation
    yaw_quat = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    transformed = motion.apply_transform(points, np.array([1.0, 2.0, 3.0]), yaw_quat)
    np.testing.assert_allclose(
        transformed, [[1.0, 3.0, 3.0], [0.0, 2.0, 3.0]], atol=1e-6
    )
    # full 3D rotation (90 degrees around x, e.g. a tilted sensor mount)
    roll_quat = np.array([np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)])
    transformed = motion.apply_transform(
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32), np.zeros(3), roll_quat
    )
    np.testing.assert_allclose(transformed, [[0.0, 0.0, 1.0]], atol=1e-6)


def test_unique_voxels_dedup_and_negatives():
    points = np.array(
        [[0.01, 0.02, 0.03], [0.05, 0.05, 0.05], [-0.3, -0.3, -0.3]],
        dtype=np.float32,
    )
    keys = motion.unique_voxels(points, voxel_size=0.1)
    assert keys.size == 2  # first two points share a voxel
    assert motion.unique_voxels(np.empty((0, 3), dtype=np.float32), 0.1).size == 0


def test_voxel_set_difference():
    first = motion.unique_voxels(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32), 0.1
    )
    second = motion.unique_voxels(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32), 0.1
    )
    # one voxel appeared and one disappeared
    assert motion.voxel_set_difference(second, first).size == 2
    assert motion.voxel_set_difference(first, first).size == 0


def _blob(center, n=30, spread=0.15, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3)) - 0.5) * spread + np.asarray(center)


def test_cluster_centers_two_blobs():
    # the first blob is larger in extent so it occupies clearly more voxels
    cloud = np.vstack(
        [
            _blob((2.0, 0.0, 0.5), n=200, spread=0.4),
            _blob((-3.0, 1.0, 0.5), n=30, seed=1),
        ]
    ).astype(np.float32)
    keys = motion.unique_voxels(cloud, voxel_size=0.1)
    centers = motion.cluster_centers(
        keys, voxel_size=0.1, min_cluster_size=2, max_clusters=5
    )
    assert len(centers) == 2
    # largest cluster first, centers close to the blob centers
    np.testing.assert_allclose(centers[0], (2.0, 0.0, 0.5), atol=0.2)
    np.testing.assert_allclose(centers[1], (-3.0, 1.0, 0.5), atol=0.2)
    # max_clusters caps the list
    assert (
        len(
            motion.cluster_centers(
                keys, voxel_size=0.1, min_cluster_size=2, max_clusters=1
            )
        )
        == 1
    )


def test_cuda_device_torch_guard():
    from importlib.util import find_spec

    if find_spec("torch") is None:
        # helpful installation message when torch is missing
        with pytest.raises(ModuleNotFoundError, match="pip install torch"):
            motion.ensure_torch()
        with pytest.raises(ModuleNotFoundError, match="pip install torch"):
            motion.unique_voxels(
                np.zeros((1, 3), dtype=np.float32), 0.1, device="cuda"
            )
    else:
        import torch

        assert motion.ensure_torch() is torch
        if not torch.cuda.is_available():
            # no CUDA device: numpy fallback with a warning
            with pytest.warns(UserWarning, match="no available CUDA device"):
                keys = motion.unique_voxels(
                    np.zeros((1, 3), dtype=np.float32), 0.1, device="cuda"
                )
            assert keys.size == 1


def test_cluster_min_size_filters_noise():
    lone_point = np.array([[5.0, 5.0, 0.0]], dtype=np.float32)
    keys = motion.unique_voxels(lone_point, voxel_size=0.1)
    assert (
        motion.cluster_centers(keys, voxel_size=0.1, min_cluster_size=2, max_clusters=5)
        == []
    )


def test_ego_motion_compensation_cancels_static_scene():
    # the same static world scene observed from two different robot poses
    world = np.vstack([_blob((3.0, 1.0, 0.2), n=50), _blob((-1.0, -2.0, 0.4), n=50)])

    def observe(pose):
        # world -> sensor frame (inverse of transform_cloud)
        x, y, z, heading = pose
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        rotation = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]])
        return (world - np.array([x, y, z])) @ rotation

    pose_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.5])
    pose_2 = np.array([0.5, -0.2, 0.0, np.pi / 6, 0.5])
    compensated_1 = motion.transform_cloud(observe(pose_1[:4]), pose_1)
    compensated_2 = motion.transform_cloud(observe(pose_2[:4]), pose_2)

    keys_1 = motion.unique_voxels(compensated_1, voxel_size=0.5)
    keys_2 = motion.unique_voxels(compensated_2, voxel_size=0.5)
    # static geometry cancels out up to voxel quantization noise
    changed = motion.voxel_set_difference(keys_2, keys_1)
    assert changed.size <= 2


# ---------------------------------------------------------------------------
# Component construction / validation matrix
# ---------------------------------------------------------------------------

IMAGE = Topic(name="image", msg_type="Image")
CLOUD = Topic(name="cloud", msg_type="PointCloud2")
ODOM = Topic(name="odom", msg_type="Odometry")
BOOL_OUT = Topic(name="motion", msg_type="Bool")
VIDEO_OUT = Topic(name="video", msg_type="Video")
CENTERS_OUT = Topic(name="centers", msg_type="PoseArray")


def _image_detector(rclpy_init, **kwargs) -> MotionDetector:
    defaults = dict(
        inputs=[IMAGE],
        outputs=[BOOL_OUT, VIDEO_OUT],
        trigger=IMAGE,
        component_name="test_motion",
    )
    defaults.update(kwargs)
    return MotionDetector(**defaults)


class TestValidationMatrix:
    def test_image_modality(self, rclpy_init):
        component = _image_detector(rclpy_init)
        assert not component._cloud_modality

    def test_cloud_modality_with_position(self, rclpy_init):
        component = MotionDetector(
            inputs=[CLOUD],
            outputs=[BOOL_OUT, CENTERS_OUT],
            trigger=CLOUD,
            position=ODOM,
            component_name="test_motion",
        )
        assert component._cloud_modality

    def test_mixed_modalities_rejected(self, rclpy_init):
        with pytest.raises(TypeError, match="not both"):
            _image_detector(rclpy_init, inputs=[IMAGE, CLOUD])

    def test_image_inputs_reject_pose_array_output(self, rclpy_init):
        with pytest.raises(TypeError):
            _image_detector(rclpy_init, outputs=[BOOL_OUT, CENTERS_OUT])

    def test_cloud_inputs_reject_video_output(self, rclpy_init):
        with pytest.raises(TypeError):
            MotionDetector(
                inputs=[CLOUD],
                outputs=[VIDEO_OUT],
                trigger=CLOUD,
                component_name="test_motion",
            )

    def test_timed_trigger_rejected(self, rclpy_init):
        with pytest.raises(TypeError):
            _image_detector(rclpy_init, trigger=1.0)

    def test_position_must_be_odometry(self, rclpy_init):
        with pytest.raises(TypeError, match="Odometry"):
            _image_detector(
                rclpy_init, position=Topic(name="pos", msg_type="String")
            )

    def test_position_cannot_be_trigger(self, rclpy_init):
        with pytest.raises(TypeError, match="trigger"):
            MotionDetector(
                inputs=[CLOUD],
                outputs=[BOOL_OUT],
                trigger=ODOM,
                position=ODOM,
                component_name="test_motion",
            )

    def test_deprecated_alias_warns(self, rclpy_init):
        with pytest.warns(DeprecationWarning, match="MotionDetector"):
            component = VideoMessageMaker(
                inputs=[IMAGE],
                outputs=[VIDEO_OUT],
                trigger=IMAGE,
                component_name="test_video_maker",
            )
        assert isinstance(component, MotionDetector)

    def test_deprecated_config_alias(self, rclpy_init):
        assert VideoMessageMakerConfig is MotionDetectorConfig

    def test_position_recovered_from_inputs(self, rclpy_init):
        # multiprocess re-instantiation path: the position topic arrives as a
        # plain input and is identified by its Odometry type
        component = MotionDetector(
            inputs=[CLOUD, ODOM],
            outputs=[BOOL_OUT, CENTERS_OUT],
            trigger=CLOUD,
            component_name="test_motion",
        )
        assert component.position is ODOM
        # not duplicated when also given explicitly
        component = MotionDetector(
            inputs=[CLOUD, ODOM],
            outputs=[BOOL_OUT],
            trigger=CLOUD,
            position=ODOM,
            component_name="test_motion",
        )
        assert sum(t.name == "odom" for t in component.in_topics) == 1



# ---------------------------------------------------------------------------
# Runtime behavior with mocked publishers
# ---------------------------------------------------------------------------


def _prep(component) -> MotionDetector:
    component.get_logger = MagicMock()
    component._bool_publishers = []
    component._video_publishers = []
    component._centers_publishers = []
    return component


class TestStateMachine:
    def test_debounce_and_bool_publishing(self, rclpy_init):
        component = _prep(
            _image_detector(
                rclpy_init, config=MotionDetectorConfig(motion_stop_delay=2)
            )
        )
        bool_publisher = MagicMock()
        component._bool_publishers = [bool_publisher]

        assert not component._step_motion_state(True)  # motion starts
        assert not component._step_motion_state(False)  # 1 still input: debounced
        assert component._motion_active
        assert component._step_motion_state(False)  # 2nd still input: episode ends
        assert not component._motion_active
        published = [c.kwargs["output"] for c in bool_publisher.publish.call_args_list]
        assert published == [True, True, False]

    def test_bool_published_on_change_only(self, rclpy_init):
        component = _prep(
            _image_detector(
                rclpy_init,
                config=MotionDetectorConfig(
                    motion_stop_delay=1, publish_bool_on_change_only=True
                ),
            )
        )
        bool_publisher = MagicMock()
        component._bool_publishers = [bool_publisher]

        for detected in (True, True, True, False):
            component._step_motion_state(detected)
        published = [c.kwargs["output"] for c in bool_publisher.publish.call_args_list]
        assert published == [True, False]


class TestImagePath:
    def test_video_published_at_max_frames(self, rclpy_init):
        # with no motion_estimation_func every frame after the first counts
        # as motion; a full buffer ends the episode and publishes the video
        component = _prep(
            _image_detector(
                rclpy_init,
                config=MotionDetectorConfig(min_video_frames=2, max_video_frames=4),
            )
        )
        video_publisher = MagicMock()
        component._video_publishers = [video_publisher]

        frame = np.dstack([_textured_frame()] * 3)
        for i in range(6):
            component._process_image(SimpleNamespace(number=i), frame)

        video_publisher.publish.assert_called_once()
        frames = video_publisher.publish.call_args.kwargs["output"]
        assert len(frames) == 4
        # motion continues after the flush: a new episode starts buffering
        assert len(component._frames) == 1


class TestCloudPath:
    def _cloud(self, points) -> SimpleNamespace:
        return SimpleNamespace(
            xyz=np.asarray(points, dtype=np.float32), frame_id="lidar"
        )

    def test_motion_centers_published(self, rclpy_init):
        component = _prep(
            MotionDetector(
                inputs=[CLOUD],
                outputs=[BOOL_OUT, CENTERS_OUT],
                trigger=CLOUD,
                component_name="test_motion",
                config=MotionDetectorConfig(
                    voxel_size=0.1,
                    changed_voxel_threshold=2,
                    min_cluster_size=2,
                    motion_stop_delay=1,
                ),
            )
        )
        bool_publisher, centers_publisher = MagicMock(), MagicMock()
        component._bool_publishers = [bool_publisher]
        component._centers_publishers = [centers_publisher]

        static_scene = _blob((2.0, 0.0, 0.5), n=60)
        moving_object = _blob((-2.0, 1.0, 0.5), n=60, seed=3)

        component._process_cloud(self._cloud(static_scene))  # first cloud: baseline
        component._process_cloud(
            self._cloud(np.vstack([static_scene, moving_object]))
        )

        assert bool_publisher.publish.call_args.kwargs["output"] is True
        centers_call = centers_publisher.publish.call_args
        assert centers_call.kwargs["frame_id"] == "lidar"
        np.testing.assert_allclose(
            centers_call.kwargs["output"][0], (-2.0, 1.0, 0.5), atol=0.2
        )

    def test_sensor_extrinsic_from_tf(self, rclpy_init):
        component = _prep(
            MotionDetector(
                inputs=[CLOUD],
                outputs=[BOOL_OUT],
                trigger=CLOUD,
                position=ODOM,
                component_name="test_motion",
            )
        )
        # TF lookup succeeded: sensor mounted 1m forward on the base
        listener = SimpleNamespace(
            got_transform=True,
            translation=np.array([1.0, 0.0, 0.0]),
            rotation=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        component.create_tf_listener = MagicMock(return_value=listener)

        points = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(
            component._apply_sensor_extrinsic(points, "lidar"), [[2.0, 1.0, 1.0]]
        )
        # listener is created once and reused
        component._apply_sensor_extrinsic(points, "lidar")
        component.create_tf_listener.assert_called_once()
        tf_config = component.create_tf_listener.call_args.args[0]
        assert tf_config.source_frame == "lidar"
        assert tf_config.goal_frame == "base_link"

        # cloud already in the base frame: no lookup needed
        np.testing.assert_allclose(
            component._apply_sensor_extrinsic(points, "base_link"), points
        )

    def test_sensor_extrinsic_unavailable_warns_once(self, rclpy_init):
        component = _prep(
            MotionDetector(
                inputs=[CLOUD],
                outputs=[BOOL_OUT],
                trigger=CLOUD,
                position=ODOM,
                component_name="test_motion",
            )
        )
        component.create_tf_listener = MagicMock(
            return_value=SimpleNamespace(got_transform=False)
        )
        points = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        # points pass through unchanged and the warning is emitted only once
        np.testing.assert_allclose(
            component._apply_sensor_extrinsic(points, "lidar"), points
        )
        np.testing.assert_allclose(
            component._apply_sensor_extrinsic(points, "lidar"), points
        )
        assert component.get_logger().warning.call_count == 1

    def test_static_scene_publishes_no_motion(self, rclpy_init):
        component = _prep(
            MotionDetector(
                inputs=[CLOUD],
                outputs=[BOOL_OUT],
                trigger=CLOUD,
                component_name="test_motion",
                config=MotionDetectorConfig(changed_voxel_threshold=2),
            )
        )
        bool_publisher = MagicMock()
        component._bool_publishers = [bool_publisher]

        scene = _blob((2.0, 0.0, 0.5), n=60)
        component._process_cloud(self._cloud(scene))
        component._process_cloud(self._cloud(scene))

        assert bool_publisher.publish.call_args.kwargs["output"] is False

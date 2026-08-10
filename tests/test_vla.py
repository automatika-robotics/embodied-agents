"""Tests for the VLA component, LeRobotPolicy model, LeRobot transport shim,
feature building from dataset info and joint data converters."""

import json
import pickle
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from agents.clients.lerobot import LeRobotClient, SERVER_SUPPORTED_POLICIES
from agents.clients.lerobot_transport.utils import (
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
)
from agents.components.vla import VLA
from agents.config import VLAConfig
from agents.models import LeRobotPolicy
from agents.ros import JointJog, JointTrajectory, JointTrajectoryPoint, Topic
from agents.utils.actions import (
    AGGREGATE_FUNCTIONS,
    JointsData,
    _as_depth_frame,
    convert_joint_limits_units,
    resolve_feature_channels,
)
from agents.utils.utils import build_lerobot_features_from_dataset_info

SAMPLE_DATASET_INFO = {
    "features": {
        "observation.state": {
            "dtype": "float64",
            "shape": [2],
            "names": ["shoulder_pan.pos", "elbow_flex.pos"],
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        },
        "observation.images.depth_front": {
            "dtype": "video",
            "shape": [480, 640, 1],
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": [2],
            "names": ["shoulder_pan.pos", "elbow_flex.pos"],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    }
}


@pytest.fixture
def dataset_info_file(tmp_path):
    """Write the sample dataset info.json to a temp file."""
    path = tmp_path / "info.json"
    path.write_text(json.dumps(SAMPLE_DATASET_INFO))
    return str(path)


class TestLeRobotTransportShim:
    """The shim dataclasses must match the LeRobot async_inference contract
    (verified against lerobot v0.6.0) for pickling to work across the wire."""

    def test_remote_policy_config_fields(self):
        assert set(RemotePolicyConfig.__dataclass_fields__) == {
            "policy_type",
            "pretrained_name_or_path",
            "lerobot_features",
            "actions_per_chunk",
            "device",
            "rename_map",
        }

    def test_timed_observation_fields(self):
        assert set(TimedObservation.__dataclass_fields__) == {
            "timestamp",
            "timestep",
            "observation",
            "must_go",
        }

    def test_timed_action_fields(self):
        assert set(TimedAction.__dataclass_fields__) == {
            "timestamp",
            "timestep",
            "action",
        }

    def test_module_spoofing_and_pickle_round_trip(self):
        assert RemotePolicyConfig.__module__ == "lerobot.async_inference.helpers"
        obs = TimedObservation(
            timestamp=1.0, timestep=3, observation={"a": 1}, must_go=True
        )
        restored = pickle.loads(pickle.dumps(obs))
        assert restored.timestep == 3
        assert restored.observation == {"a": 1}
        assert restored.must_go is True


class TestLeRobotPolicy:
    def test_init_params_contain_rename_map(self):
        model = LeRobotPolicy(name="policy")
        params = model._get_init_params()
        assert params["rename_map"] == {}
        assert set(params) == {
            "checkpoint",
            "policy_type",
            "features",
            "actions_per_chunk",
            "device",
            "rename_map",
        }

    def test_rename_map_passthrough(self):
        model = LeRobotPolicy(
            name="policy",
            rename_map={"observation.images.front": "observation.images.top"},
        )
        assert model._get_init_params()["rename_map"] == {
            "observation.images.front": "observation.images.top"
        }

    def test_server_supported_policies_derived_from_literal(self):
        assert set(SERVER_SUPPORTED_POLICIES) == {
            "act",
            "smolvla",
            "diffusion",
            "tdmpc",
            "vqbet",
            "pi0",
            "pi05",
            "groot",
        }

    def test_dataset_info_parsing(self, dataset_info_file):
        model = LeRobotPolicy(name="policy", dataset_info_file=dataset_info_file)
        assert model._joint_keys == ["shoulder_pan.pos", "elbow_flex.pos"]
        assert sorted(model._image_keys) == ["depth_front", "front"]
        assert model._actions["names"] == ["shoulder_pan.pos", "elbow_flex.pos"]


class TestFeatureBuilder:
    def test_float64_state_coerced_to_float32(self, dataset_info_file):
        result = build_lerobot_features_from_dataset_info(dataset_info_file)
        # the policy server only builds state features with dtype 'float32'
        assert result["features"]["observation.state"]["dtype"] == "float32"

    def test_depth_feature_shape_preserved(self, dataset_info_file):
        result = build_lerobot_features_from_dataset_info(dataset_info_file)
        depth = result["features"]["observation.images.depth_front"]
        assert depth["shape"] == (480, 640, 1)
        assert depth["dtype"] == "video"

    def test_image_keys_stripped_of_prefix(self, dataset_info_file):
        result = build_lerobot_features_from_dataset_info(dataset_info_file)
        assert sorted(result["image_keys"]) == ["depth_front", "front"]

    def test_nested_names_flattened(self, tmp_path):
        info = {
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": {"motors": ["pan", "lift"]},
                },
                "action": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": {"motors": ["pan", "lift"]},
                },
            }
        }
        path = tmp_path / "info.json"
        path.write_text(json.dumps(info))
        result = build_lerobot_features_from_dataset_info(str(path))
        assert result["features"]["observation.state"]["names"] == [
            "motors.pan",
            "motors.lift",
        ]


def _queue_harness(preset: str) -> VLA:
    """Bare VLA instance with just the state used by _update_actions_queue."""
    comp = VLA.__new__(VLA)
    comp._actions_received = queue.Queue()
    comp._last_executed_timestep_lock = threading.Lock()
    comp._last_executed_timestep = -1
    comp._aggregator_function = AGGREGATE_FUNCTIONS[preset]
    return comp


def _timed_action(timestep: int, value: float) -> TimedAction:
    return TimedAction(
        timestamp=float(timestep), timestep=timestep, action=np.array([value])
    )


class TestActionsQueueAggregation:
    def test_new_actions_inserted_in_timestep_order(self):
        comp = _queue_harness("latest_only")
        comp._update_actions_queue([_timed_action(5, 5.0), _timed_action(2, 2.0)])
        timesteps = [a.timestep for a in comp._actions_received.queue]
        assert timesteps == [2, 5]

    def test_stale_timesteps_dropped(self):
        comp = _queue_harness("latest_only")
        comp._last_executed_timestep = 3
        comp._update_actions_queue([_timed_action(2, 2.0), _timed_action(4, 4.0)])
        timesteps = [a.timestep for a in comp._actions_received.queue]
        assert timesteps == [4]

    def test_overlap_latest_only(self):
        comp = _queue_harness("latest_only")
        comp._update_actions_queue([_timed_action(1, 1.0)])
        comp._update_actions_queue([_timed_action(1, 9.0)])
        actions = list(comp._actions_received.queue)
        assert len(actions) == 1
        assert np.allclose(actions[0].action, [9.0])

    def test_overlap_weighted_average(self):
        comp = _queue_harness("weighted_average")
        comp._update_actions_queue([_timed_action(1, 1.0)])
        comp._update_actions_queue([_timed_action(1, 9.0)])
        actions = list(comp._actions_received.queue)
        # 0.3 * old + 0.7 * new
        assert np.allclose(actions[0].action, [0.3 * 1.0 + 0.7 * 9.0])

    def test_overlap_conservative(self):
        comp = _queue_harness("conservative")
        comp._update_actions_queue([_timed_action(1, 1.0)])
        comp._update_actions_queue([_timed_action(1, 9.0)])
        actions = list(comp._actions_received.queue)
        # 0.7 * old + 0.3 * new
        assert np.allclose(actions[0].action, [0.7 * 1.0 + 0.3 * 9.0])


class TestJointLimitsUnitConversion:
    def test_radians_is_passthrough(self):
        limits = {"joint1": {"lower": -1.0, "upper": 1.0, "velocity": 2.0}}
        assert convert_joint_limits_units(limits, "radians") == limits

    def test_degrees_conversion(self):
        limits = {
            "joint1": {
                "lower": -np.pi,
                "upper": np.pi / 2,
                "effort": 5.0,
                "velocity": np.pi,
            }
        }
        out = convert_joint_limits_units(limits, "degrees")
        assert out["joint1"]["lower"] == pytest.approx(-180.0)
        assert out["joint1"]["upper"] == pytest.approx(90.0)
        assert out["joint1"]["velocity"] == pytest.approx(180.0)
        # effort is not an angular quantity, must stay untouched
        assert out["joint1"]["effort"] == 5.0

    def test_normalized_maps_range_and_scales_velocity(self):
        limits = {"joint1": {"lower": -2.0, "upper": 2.0, "velocity": 1.0}}
        out = convert_joint_limits_units(limits, "normalized")
        assert out["joint1"]["lower"] == -100.0
        assert out["joint1"]["upper"] == 100.0
        # velocity scaled by the same per-joint factor: 200 / 4 rad
        assert out["joint1"]["velocity"] == pytest.approx(50.0)

    def test_normalized_gripper_maps_to_0_100(self):
        limits = {"gripper_joint": {"lower": -0.17, "upper": 1.75, "velocity": None}}
        out = convert_joint_limits_units(limits, "normalized")
        assert out["gripper_joint"]["lower"] == 0.0
        assert out["gripper_joint"]["upper"] == 100.0

    def test_normalized_unusable_limits_become_none(self):
        limits = {
            "no_bounds": {"lower": None, "upper": 1.0},
            "inverted": {"lower": 1.0, "upper": 1.0},
        }
        out = convert_joint_limits_units(limits, "normalized")
        assert out["no_bounds"] is None
        assert out["inverted"] is None


class TestResolveFeatureChannels:
    def test_channel_last_with_names(self):
        feature = {"shape": [480, 640, 1], "names": ["height", "width", "channel"]}
        assert resolve_feature_channels(feature) == 1

    def test_channel_first_with_names(self):
        feature = {"shape": [1, 480, 640], "names": ["channels", "height", "width"]}
        assert resolve_feature_channels(feature) == 1

    def test_2d_shape_is_single_channel(self):
        assert resolve_feature_channels({"shape": [480, 640]}) == 1

    def test_3d_shape_without_names_is_channel_last(self):
        assert resolve_feature_channels({"shape": [480, 640, 3]}) == 3

    def test_missing_feature_defaults_to_rgb(self):
        assert resolve_feature_channels({}) == 3

    def test_names_not_parallel_to_shape_ignored(self):
        feature = {"shape": [480, 640, 3], "names": ["channel"]}
        assert resolve_feature_channels(feature) == 3


class TestDepthFrame:
    def test_2d_expanded_to_single_channel(self):
        out = _as_depth_frame(np.zeros((4, 5), dtype=np.float32))
        assert out.shape == (4, 5, 1)

    def test_uint16_cast_to_float32(self):
        out = _as_depth_frame(np.zeros((4, 5), dtype=np.uint16))
        assert out.dtype == np.float32

    def test_single_channel_float_passthrough(self):
        img = np.zeros((4, 5, 1), dtype=np.float32)
        out = _as_depth_frame(img)
        assert out.shape == (4, 5, 1)
        assert out.dtype == np.float32


def _ros_image(arr: np.ndarray, encoding: str):
    from sensor_msgs.msg import Image as ImageROS

    msg = ImageROS()
    msg.height = arr.shape[0]
    msg.width = arr.shape[1]
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    msg.encoding = encoding
    msg.step = arr.shape[1] * channels * arr.itemsize
    msg.is_bigendian = False
    msg.data = arr.tobytes()
    return msg


class TestRGBDCallbackDepthOnly:
    @pytest.fixture
    def rgbd_callback(self):
        # RGBD is an optional msg type in agents.ros
        pytest.importorskip("realsense2_camera_msgs")
        from agents.callbacks import RGBDCallback

        callback = RGBDCallback(Topic(name="cam_rgbd", msg_type="RGBD"))
        rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        depth = np.arange(4 * 5, dtype=np.uint16).reshape(4, 5)
        callback.msg = SimpleNamespace(
            rgb=_ros_image(rgb, "rgb8"), depth=_ros_image(depth, "16UC1")
        )
        return callback

    def test_default_returns_rgb(self, rgbd_callback):
        out = rgbd_callback._get_output()
        assert out.shape == (4, 5, 3)

    def test_get_depth_returns_fused_rgbd(self, rgbd_callback):
        out = rgbd_callback._get_output(get_depth=True)
        assert out.shape == (4, 5, 4)

    def test_depth_only_returns_single_channel(self, rgbd_callback):
        out = rgbd_callback._get_output(depth_only=True)
        assert out.shape == (4, 5, 1)
        assert out.dtype == np.uint16


class TestJointConverters:
    @pytest.fixture
    def joints_data(self):
        return JointsData(
            joints_names=["joint1", "joint2"],
            positions=np.array([1.0, 2.0]),
            velocities=np.array([0.1, 0.2]),
            accelerations=np.array([0.01, 0.02]),
            efforts=np.array([0.5, 0.6]),
            duration=0.25,
            delay=1.5,
        )

    def test_trajectory_point_uses_accelerations(self, joints_data):
        msg = JointTrajectoryPoint.convert(joints_data)
        assert list(msg.accelerations) == [0.01, 0.02]
        assert list(msg.velocities) == [0.1, 0.2]

    def test_trajectory_point_time_from_start_is_duration_msg(self, joints_data):
        from builtin_interfaces.msg import Duration

        msg = JointTrajectoryPoint.convert(joints_data)
        assert isinstance(msg.time_from_start, Duration)
        # delay + duration = 1.75s
        assert msg.time_from_start.sec == 1
        assert msg.time_from_start.nanosec == 750000000

    def test_trajectory_points_monotonically_timed(self):
        # 2D (multi point) joints data bypasses JointsData size validation,
        # so a duck typed stub is used to exercise the indexed conversion
        stub = SimpleNamespace(
            joints_names=["joint1", "joint2"],
            positions=np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]]),
            velocities=np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]),
            accelerations=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            efforts=np.array([[0.5, 0.6], [0.5, 0.6], [0.5, 0.6]]),
            duration=0.1,
            delay=0.0,
        )
        msg = JointTrajectory.convert(stub)
        assert len(msg.points) == 3
        times = [
            p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
            for p in msg.points
        ]
        assert times == sorted(times)
        assert times[0] > 0.0
        assert np.allclose(times, [0.1, 0.2, 0.3], atol=1e-6)

    def test_joint_jog_duration_is_float(self, joints_data):
        # JointJog is an optional msg type in agents.ros
        pytest.importorskip("control_msgs")
        msg = JointJog.convert(joints_data)
        assert msg.duration == 0.25
        assert list(msg.displacements) == [1.0, 2.0]


def _mock_lerobot_client(model: LeRobotPolicy) -> MagicMock:
    client = MagicMock(spec=LeRobotClient)
    client._model = model
    client.model_init_params = model._get_init_params()
    return client


class TestVLAComponent:
    @pytest.fixture
    def vla_topics(self):
        return {
            "state": Topic(name="joint_states", msg_type="JointState"),
            "camera": Topic(name="camera_rgb", msg_type="Image"),
            "depth": Topic(name="camera_depth", msg_type="Image"),
            "out": Topic(name="joint_cmd", msg_type="JointState"),
        }

    def test_auto_spec_refreshes_client_init_params(self, rclpy_init, vla_topics):
        """Auto-generated features must reach the client's init params,
        otherwise the policy server receives an empty feature spec."""
        model = LeRobotPolicy(name="policy")
        client = _mock_lerobot_client(model)
        assert not client.model_init_params["features"]
        VLA(
            inputs=[vla_topics["state"], vla_topics["camera"]],
            outputs=[vla_topics["out"]],
            model_client=client,
            config=VLAConfig(
                joint_names_map={"shoulder_pan.pos": "joint1"},
                camera_inputs_map={"front": vla_topics["camera"]},
            ),
            component_name="test_vla_auto_spec",
        )
        features = client.model_init_params["features"]
        assert "observation.state" in features
        assert "observation.images.front" in features

    def test_dataset_verification_and_camera_prefix_strip(
        self, rclpy_init, vla_topics, dataset_info_file
    ):
        model = LeRobotPolicy(name="policy", dataset_info_file=dataset_info_file)
        client = _mock_lerobot_client(model)
        config = VLAConfig(
            joint_names_map={
                "shoulder_pan.pos": "joint1",
                "elbow_flex.pos": "joint2",
            },
            camera_inputs_map={
                "observation.images.front": vla_topics["camera"],
                "depth_front": vla_topics["depth"],
            },
        )
        comp = VLA(
            inputs=[vla_topics["state"], vla_topics["camera"], vla_topics["depth"]],
            outputs=[vla_topics["out"]],
            model_client=client,
            config=config,
            component_name="test_vla_dataset",
        )
        # dataset action names mapped to robot joint names, in dataset order
        assert comp._dataset_sorted_joint_names == ["joint1", "joint2"]
        # LeRobot prefix stripped from user provided camera keys
        assert sorted(config.camera_inputs_map.keys()) == ["depth_front", "front"]

    def test_incomplete_joint_map_raises(
        self, rclpy_init, vla_topics, dataset_info_file
    ):
        model = LeRobotPolicy(name="policy", dataset_info_file=dataset_info_file)
        client = _mock_lerobot_client(model)
        with pytest.raises(ValueError):
            VLA(
                inputs=[vla_topics["state"], vla_topics["camera"]],
                outputs=[vla_topics["out"]],
                model_client=client,
                config=VLAConfig(
                    joint_names_map={"shoulder_pan.pos": "joint1"},
                    camera_inputs_map={
                        "front": vla_topics["camera"],
                        "depth_front": vla_topics["depth"],
                    },
                ),
                component_name="test_vla_bad_joints",
            )

    def test_incomplete_camera_map_raises(
        self, rclpy_init, vla_topics, dataset_info_file
    ):
        model = LeRobotPolicy(name="policy", dataset_info_file=dataset_info_file)
        client = _mock_lerobot_client(model)
        with pytest.raises(ValueError):
            VLA(
                inputs=[vla_topics["state"], vla_topics["camera"]],
                outputs=[vla_topics["out"]],
                model_client=client,
                config=VLAConfig(
                    joint_names_map={
                        "shoulder_pan.pos": "joint1",
                        "elbow_flex.pos": "joint2",
                    },
                    camera_inputs_map={"front": vla_topics["camera"]},
                ),
                component_name="test_vla_bad_cameras",
            )

    @pytest.fixture
    def urdf_file(self, tmp_path):
        path = tmp_path / "so101.urdf"
        path.write_text(
            '<robot name="so101">'
            '<joint name="joint1" type="revolute">'
            '<limit lower="-2.0" upper="2.0" effort="10.0" velocity="1.0"/>'
            "</joint>"
            '<joint name="joint2" type="revolute">'
            '<limit lower="-1.0" upper="1.0" effort="10.0" velocity="1.0"/>'
            "</joint>"
            "</robot>"
        )
        return str(path)

    def _make_vla_with_limits(self, vla_topics, dataset_info_file, config_kwargs, name):
        model = LeRobotPolicy(name="policy", dataset_info_file=dataset_info_file)
        client = _mock_lerobot_client(model)
        config = VLAConfig(
            joint_names_map={
                "shoulder_pan.pos": "joint1",
                "elbow_flex.pos": "joint2",
            },
            camera_inputs_map={
                "front": vla_topics["camera"],
                "depth_front": vla_topics["depth"],
            },
            **config_kwargs,
        )
        return VLA(
            inputs=[vla_topics["state"], vla_topics["camera"], vla_topics["depth"]],
            outputs=[vla_topics["out"]],
            model_client=client,
            config=config,
            component_name=name,
        )

    def test_urdf_limits_converted_to_policy_units(
        self, rclpy_init, vla_topics, dataset_info_file, urdf_file
    ):
        comp = self._make_vla_with_limits(
            vla_topics,
            dataset_info_file,
            {"robot_urdf_file": urdf_file, "policy_action_units": "normalized"},
            "test_vla_urdf_units",
        )
        assert comp.robot_joints_limits["joint1"]["lower"] == -100.0
        assert comp.robot_joints_limits["joint1"]["upper"] == 100.0
        assert comp.robot_joints_limits["joint2"]["upper"] == 100.0

    def test_urdf_limits_default_stay_radians(
        self, rclpy_init, vla_topics, dataset_info_file, urdf_file
    ):
        comp = self._make_vla_with_limits(
            vla_topics,
            dataset_info_file,
            {"robot_urdf_file": urdf_file},
            "test_vla_urdf_radians",
        )
        assert comp.robot_joints_limits["joint1"]["lower"] == -2.0
        assert comp.robot_joints_limits["joint1"]["upper"] == 2.0

    def test_manual_joint_limits_override_urdf(
        self, rclpy_init, vla_topics, dataset_info_file, urdf_file
    ):
        manual = {"joint2": {"lower": -5.0, "upper": 5.0}}
        comp = self._make_vla_with_limits(
            vla_topics,
            dataset_info_file,
            {
                "robot_urdf_file": urdf_file,
                "policy_action_units": "normalized",
                "joint_limits": manual,
            },
            "test_vla_manual_override",
        )
        # manual entry wins verbatim for joint2; joint1 stays URDF-derived
        assert comp.robot_joints_limits["joint2"] == {"lower": -5.0, "upper": 5.0}
        assert comp.robot_joints_limits["joint1"]["lower"] == -100.0

    def test_sent_actions_carry_one_tick_duration(self, rclpy_init, vla_topics):
        """Published JointsData must carry a nonzero duration (one action
        tick) — a zero time_from_start makes joint_trajectory_controller
        reject the trajectory and gives JointJog a zero jog duration."""
        model = LeRobotPolicy(name="policy")
        client = _mock_lerobot_client(model)
        comp = VLA(
            inputs=[vla_topics["state"], vla_topics["camera"]],
            outputs=[vla_topics["out"]],
            model_client=client,
            config=VLAConfig(
                joint_names_map={"shoulder_pan.pos": "joint1"},
                camera_inputs_map={"front": vla_topics["camera"]},
            ),
            component_name="test_vla_action_duration",
        )
        # action queue state is normally created in custom_on_activate
        comp._actions_received = queue.Queue()
        comp._action_queue_lock = threading.Lock()
        comp._last_executed_timestep_lock = threading.Lock()
        comp._last_executed_timestep = -1
        comp._actions_received.put(
            SimpleNamespace(
                action=SimpleNamespace(numpy=lambda: np.array([0.1])), timestep=3
            )
        )
        published = {}
        comp._publish = lambda result: published.update(result)

        comp._send_action_commands()

        data = published["output"]
        assert data.duration == pytest.approx(1 / comp.config.action_sending_rate)
        assert data.duration > 0

    def _prepare_goal_execution(self, vla_topics, name):
        """Build a VLA and patch the internals main_action_callback needs."""
        model = LeRobotPolicy(name="policy")
        client = _mock_lerobot_client(model)
        comp = VLA(
            inputs=[vla_topics["state"], vla_topics["camera"]],
            outputs=[vla_topics["out"]],
            model_client=client,
            config=VLAConfig(
                joint_names_map={"shoulder_pan.pos": "joint1"},
                camera_inputs_map={"front": vla_topics["camera"]},
            ),
            component_name=name,
        )
        comp._actions_received = queue.Queue()
        comp._last_executed_timestep_lock = threading.Lock()
        comp._last_executed_timestep = -1
        comp._task_completed = False
        comp._main_goal_lock = threading.Lock()
        comp.create_timer = MagicMock()
        comp.get_logger = MagicMock()
        comp.got_all_inputs = MagicMock(return_value=True)
        comp._action_done = MagicMock(return_value=False)
        comp._action_cleanup = MagicMock()
        return comp

    def test_cancel_requested_goal_transitions_to_canceled(
        self, rclpy_init, vla_topics
    ):
        """A canceling goal must reach STATUS_CANCELED — returning without a
        transition makes rclpy force-abort it, so clients cannot distinguish
        a clean preempt from an execution failure."""
        comp = self._prepare_goal_execution(vla_topics, "test_vla_cancel")
        goal_handle = MagicMock()
        goal_handle.request.task = "pick"
        goal_handle.is_active = True
        goal_handle.is_cancel_requested = True

        comp.main_action_callback(goal_handle)

        goal_handle.canceled.assert_called_once()
        goal_handle.abort.assert_not_called()
        comp._action_cleanup.assert_called_once()

    def test_preempted_goal_not_transitioned_again(self, rclpy_init, vla_topics):
        """A goal already aborted by preemption is terminal — transitioning
        it again would be an invalid state transition."""
        comp = self._prepare_goal_execution(vla_topics, "test_vla_preempt")
        goal_handle = MagicMock()
        goal_handle.request.task = "pick"
        goal_handle.is_active = False
        goal_handle.is_cancel_requested = False

        comp.main_action_callback(goal_handle)

        goal_handle.canceled.assert_not_called()
        goal_handle.abort.assert_not_called()
        comp._action_cleanup.assert_called_once()

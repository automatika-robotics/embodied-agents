"""Tests for MoveIt request-building utilities — no ROS node needed."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agents.utils.moveit import (
    MoveMode,
    ensure_moveit_msgs,
    parse_planner_interfaces,
    parse_srdf_group_states,
)


def _trajectory():
    """A real RobotTrajectory for mocked cartesian responses: the component
    assigns it to a message field, whose setter checks the type."""
    from moveit_msgs.msg import RobotTrajectory

    return RobotTrajectory()

SAMPLE_SRDF = """
<robot name="panda">
  <group name="panda_arm">
    <chain base_link="panda_link0" tip_link="panda_link8"/>
  </group>
  <group name="hand">
    <joint name="panda_finger_joint1"/>
  </group>
  <group_state name="ready" group="panda_arm">
    <joint name="panda_joint1" value="0"/>
    <joint name="panda_joint2" value="-0.785"/>
  </group_state>
  <group_state name="extended" group="panda_arm">
    <joint name="panda_joint1" value="0"/>
    <joint name="panda_joint2" value="0"/>
  </group_state>
  <group_state name="open" group="hand">
    <joint name="panda_finger_joint1" value="0.035"/>
  </group_state>
</robot>
"""


class TestSrdfParsing:
    def test_group_states_extracted_per_group(self):
        states = parse_srdf_group_states(SAMPLE_SRDF)
        assert set(states.keys()) == {"panda_arm", "hand"}
        assert states["panda_arm"]["ready"] == {
            "panda_joint1": 0.0,
            "panda_joint2": -0.785,
        }
        assert set(states["panda_arm"].keys()) == {"ready", "extended"}
        assert states["hand"]["open"] == {"panda_finger_joint1": 0.035}

    def test_multi_dof_value_takes_first(self):
        srdf = (
            '<robot><group_state name="s" group="g">'
            '<joint name="j" value="1.5 0.5"/></group_state></robot>'
        )
        assert parse_srdf_group_states(srdf) == {"g": {"s": {"j": 1.5}}}

    def test_no_states_empty(self):
        assert parse_srdf_group_states("<robot/>") == {}


class TestInterfaceNames:
    """Interface naming on the component — requires rclpy and moveit_msgs."""

    @pytest.fixture
    def component(self, rclpy_init):
        pytest.importorskip("moveit_msgs.msg")
        from agents.components.moveit import MoveIt
        from agents.config import MoveItConfig

        return MoveIt(
            config=MoveItConfig(arm_group_name="panda_arm"),
            component_name="test_moveit_names",
        )

    def test_no_namespace(self, component):
        assert component._resolve_name("move_action") == "/move_action"

    def test_namespace_forms_normalized(self, component):
        for namespace in ("robot_a", "/robot_a", "/robot_a/", " robot_a "):
            component.config.move_group_namespace = namespace
            assert component._resolve_name("move_action") == "/robot_a/move_action"

    def test_nested_name_keeps_namespace(self, component):
        component.config.move_group_namespace = "/robot_a"
        assert (
            component._resolve_name("move_group/get_parameters")
            == "/robot_a/move_group/get_parameters"
        )


class TestComponentNamedTargets:
    """The SRDF is read from the running move_group node by the component."""

    @pytest.fixture
    def component(self, rclpy_init):
        pytest.importorskip("moveit_msgs.msg")
        from agents.components.moveit import MoveIt
        from agents.config import MoveItConfig

        comp = MoveIt(
            config=MoveItConfig(arm_group_name="panda_arm", gripper_group_name="hand"),
            component_name="test_moveit_targets",
        )
        comp.get_logger = MagicMock()
        return comp

    @staticmethod
    def _param_client(response):
        client = MagicMock()
        client.send_request_from_dict.return_value = response
        return client

    def test_targets_read_from_move_group(self, component):
        from agents.utils.moveit import SRDF_PARAMETER

        component._param_client = self._param_client(
            SimpleNamespace(values=[SimpleNamespace(string_value=SAMPLE_SRDF)])
        )
        states = component._named_target_states()
        # parsed into targets, not returned as raw SRDF
        assert states["panda_arm"]["ready"]["panda_joint2"] == -0.785
        assert states["hand"]["open"] == {"panda_finger_joint1": 0.035}
        component._param_client.send_request_from_dict.assert_called_once_with({
            "names": [SRDF_PARAMETER]
        })

    def test_targets_cached_after_first_read(self, component):
        component._param_client = self._param_client(
            SimpleNamespace(values=[SimpleNamespace(string_value=SAMPLE_SRDF)])
        )
        first = component._named_target_states()
        assert component._named_target_states() is first
        assert component._param_client.send_request_from_dict.call_count == 1

    def test_falls_back_to_srdf_file(self, component, tmp_path):
        srdf_file = tmp_path / "robot.srdf"
        srdf_file.write_text(SAMPLE_SRDF)
        component.config.srdf_file = str(srdf_file)
        component._param_client = self._param_client(None)
        assert component._named_target_states()["panda_arm"]["ready"]

    def test_no_sources_returns_empty_dict(self, component):
        component._param_client = self._param_client(None)
        # must be a dict, callers index into it
        assert component._named_target_states() == {}
        assert component.get_named_targets.__wrapped__(component) == (
            "No named targets are defined for this robot"
        )


class TestNamedTargets:
    def test_targets_read_from_srdf_content(self):
        from agents.utils.moveit import load_named_targets

        states = load_named_targets(srdf=SAMPLE_SRDF)
        assert states["panda_arm"]["ready"]["panda_joint2"] == -0.785
        assert states["hand"]["open"] == {"panda_finger_joint1": 0.035}

    def test_overrides_take_precedence(self):
        from agents.utils.moveit import load_named_targets

        states = load_named_targets(
            srdf=SAMPLE_SRDF,
            overrides={
                "panda_arm": {"ready": {"panda_joint1": 1.57}},
                "extra": {"parked": {"j": 0.0}},
            },
        )
        # overridden target replaces the SRDF one, others are kept
        assert states["panda_arm"]["ready"] == {"panda_joint1": 1.57}
        assert "extended" in states["panda_arm"]
        assert states["extra"]["parked"] == {"j": 0.0}

    def test_file_used_when_no_content_given(self, tmp_path):
        from agents.utils.moveit import load_named_targets

        srdf_file = tmp_path / "robot.srdf"
        srdf_file.write_text(SAMPLE_SRDF)
        states = load_named_targets(srdf_file=str(srdf_file))
        assert states["panda_arm"]["ready"]["panda_joint2"] == -0.785

    def test_content_preferred_over_file(self, tmp_path):
        from agents.utils.moveit import load_named_targets

        srdf_file = tmp_path / "robot.srdf"
        srdf_file.write_text(SAMPLE_SRDF)
        states = load_named_targets(
            srdf='<robot><group_state name="parked" group="arm">'
            '<joint name="j" value="1.0"/></group_state></robot>',
            srdf_file=str(srdf_file),
        )
        assert states == {"arm": {"parked": {"j": 1.0}}}

    def test_no_sources_returns_empty(self):
        from agents.utils.moveit import load_named_targets

        assert load_named_targets() == {}

    def test_unreadable_file_returns_empty(self):
        from agents.utils.moveit import load_named_targets

        assert load_named_targets(srdf_file="/nonexistent/robot.srdf") == {}

    def test_malformed_srdf_does_not_raise(self):
        from agents.utils.moveit import load_named_targets

        assert load_named_targets(srdf="<robot><not-xml") == {}


class TestPlannerInterfaces:
    def test_pipelines_mapped_to_planner_ids(self):
        response = SimpleNamespace(
            planner_interfaces=[
                SimpleNamespace(
                    name="OMPL",
                    pipeline_id="ompl",
                    planner_ids=["RRTConnect", "RRTstar"],
                ),
                SimpleNamespace(
                    name="Pilz", pipeline_id="pilz", planner_ids=["PTP", "LIN"]
                ),
            ]
        )
        assert parse_planner_interfaces(response) == {
            "ompl": ["RRTConnect", "RRTstar"],
            "pilz": ["PTP", "LIN"],
        }


class TestMoveMode:
    """Mode resolution is pure goal logic — no component needed."""

    @staticmethod
    def _goal(**fields):
        """Minimal stand-in for a MoveManipulator goal."""
        goal = SimpleNamespace(
            mode="",
            named_target="",
            target_object="",
            cartesian_waypoints=[],
            joint_names=[],
            joint_positions=[],
            target_pose=SimpleNamespace(
                header=SimpleNamespace(frame_id=""),
                pose=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0, z=0.0)),
            ),
        )
        for key, value in fields.items():
            setattr(goal, key, value)
        return goal

    def test_values(self):
        assert MoveMode.values() == [
            "pose", "joints", "named", "cartesian", "pick", "place",
        ]

    def test_inferred_pick_from_target_object(self):
        assert MoveMode.from_goal(self._goal(target_object="mug")) is MoveMode.PICK

    def test_pick_wins_over_a_pose_that_locates_it(self):
        """A pick may carry a pose as the object's location; naming an object
        makes the intent unambiguous."""
        goal = self._goal(target_object="mug")
        goal.target_pose.pose.position.x = 0.4
        assert MoveMode.from_goal(goal) is MoveMode.PICK

    def test_place_is_never_inferred(self):
        """A place goal's bare pose is indistinguishable from a POSE goal, so
        'place' must be explicit."""
        goal = self._goal()
        goal.target_pose.pose.position.x = 0.4
        assert MoveMode.from_goal(goal) is MoveMode.POSE
        assert MoveMode.from_goal(self._goal(mode="place")) is MoveMode.PLACE

    def test_is_a_string_enum(self):
        assert MoveMode.POSE == "pose"

    def test_explicit_mode_normalized(self):
        assert MoveMode.from_goal(self._goal(mode=" CARTESIAN ")) is MoveMode.CARTESIAN

    def test_unknown_mode_lists_valid_modes(self):
        with pytest.raises(ValueError, match="Valid modes"):
            MoveMode.from_goal(self._goal(mode="teleport"))

    def test_inferred_from_named_target(self):
        assert MoveMode.from_goal(self._goal(named_target="home")) is MoveMode.NAMED

    def test_inferred_from_waypoints(self):
        goal = self._goal(cartesian_waypoints=[object()])
        assert MoveMode.from_goal(goal) is MoveMode.CARTESIAN

    def test_inferred_from_joints(self):
        goal = self._goal(joint_names=["j1"], joint_positions=[0.5])
        assert MoveMode.from_goal(goal) is MoveMode.JOINTS

    def test_inferred_from_pose_frame_or_position(self):
        goal = self._goal()
        goal.target_pose.header.frame_id = "base_link"
        assert MoveMode.from_goal(goal) is MoveMode.POSE
        origin_goal = self._goal()
        origin_goal.target_pose.pose.position.z = 0.5
        assert MoveMode.from_goal(origin_goal) is MoveMode.POSE

    def test_empty_goal_raises(self):
        with pytest.raises(ValueError, match="No motion target"):
            MoveMode.from_goal(self._goal())


class TestMoveItMsgsGuard:
    def test_ensure_raises_without_moveit(self, monkeypatch):
        monkeypatch.setattr(
            "agents.utils.moveit.find_spec", lambda name: None
        )
        with pytest.raises(ModuleNotFoundError, match="moveit-msgs"):
            ensure_moveit_msgs()

    def test_builders_raise_install_hint_without_moveit(self, monkeypatch):
        """Every builder must surface the install instructions even when
        called directly, without going through the component."""
        from agents.utils import moveit as moveit_utils

        monkeypatch.setattr("agents.utils.moveit.find_spec", lambda name: None)
        for builder, args in [
            (moveit_utils.joints_to_constraints, ({"j1": 0.0}, 1e-3)),
            (moveit_utils.build_move_group_goal, (None, "arm")),
            (moveit_utils.build_cartesian_request, ([], "base", "arm", "ee")),
            (moveit_utils.moveit_error_string, (1,)),
        ]:
            with pytest.raises(ModuleNotFoundError, match="moveit-msgs"):
                builder(*args)


class TestRequestBuilders:
    """Builders produce real moveit_msgs — skipped when not installed."""

    @pytest.fixture(autouse=True)
    def _needs_moveit_msgs(self):
        pytest.importorskip("moveit_msgs.msg")

    @staticmethod
    def _pose_stamped(x=0.3, y=0.0, z=0.5, frame="base_link"):
        from geometry_msgs.msg import PoseStamped

        pose = PoseStamped()
        pose.header.frame_id = frame
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        return pose

    def test_pose_constraints(self):
        from shape_msgs.msg import SolidPrimitive

        from agents.utils.moveit import pose_to_constraints

        constraints = pose_to_constraints(
            self._pose_stamped(),
            link_name="ee_link",
            position_tolerance=1e-3,
            orientation_tolerance=1e-2,
        )
        position = constraints.position_constraints[0]
        assert position.link_name == "ee_link"
        assert position.header.frame_id == "base_link"
        sphere = position.constraint_region.primitives[0]
        assert sphere.type == SolidPrimitive.SPHERE
        assert list(sphere.dimensions) == [1e-3]
        assert position.constraint_region.primitive_poses[0].position.x == 0.3
        orientation = constraints.orientation_constraints[0]
        assert orientation.orientation.w == 1.0
        assert orientation.absolute_x_axis_tolerance == 1e-2
        assert position.weight == orientation.weight == 1.0

    def test_pose_constraints_per_axis_orientation_tolerance(self):
        """Per-axis tolerances support underactuated arms (e.g. relax only
        the tool axis on a 5-DOF arm)."""
        from agents.utils.moveit import pose_to_constraints

        constraints = pose_to_constraints(
            self._pose_stamped(),
            link_name="ee_link",
            position_tolerance=1e-3,
            orientation_tolerance=(1e-2, 1e-2, 3.14),
        )
        orientation = constraints.orientation_constraints[0]
        assert orientation.absolute_x_axis_tolerance == pytest.approx(1e-2)
        assert orientation.absolute_y_axis_tolerance == pytest.approx(1e-2)
        assert orientation.absolute_z_axis_tolerance == pytest.approx(3.14)

    def test_pose_constraints_bad_tolerance_length_raises(self):
        from agents.utils.moveit import pose_to_constraints

        with pytest.raises(ValueError, match="3 per-axis"):
            pose_to_constraints(
                self._pose_stamped(),
                link_name="ee_link",
                position_tolerance=1e-3,
                orientation_tolerance=(1e-2, 1e-2),
            )

    def test_joint_constraints(self):
        from agents.utils.moveit import joints_to_constraints

        constraints = joints_to_constraints({"j1": 0.5, "j2": -0.25}, tolerance=1e-3)
        assert len(constraints.joint_constraints) == 2
        first = constraints.joint_constraints[0]
        assert first.joint_name == "j1"
        assert first.position == 0.5
        assert first.tolerance_above == first.tolerance_below == 1e-3

    def test_move_group_goal(self):
        from agents.utils.moveit import build_move_group_goal, joints_to_constraints

        constraints = joints_to_constraints({"j1": 0.5}, tolerance=1e-3)
        goal = build_move_group_goal(
            constraints,
            "panda_arm",
            pipeline_id="ompl",
            planner_id="RRTConnect",
            num_attempts=3,
            allowed_time=2.0,
            velocity_scaling=0.2,
            acceleration_scaling=0.3,
            plan_only=True,
        )
        request = goal.request
        assert request.group_name == "panda_arm"
        assert request.goal_constraints == [constraints]
        assert request.pipeline_id == "ompl"
        assert request.planner_id == "RRTConnect"
        assert request.num_planning_attempts == 3
        assert request.allowed_planning_time == 2.0
        assert request.max_velocity_scaling_factor == pytest.approx(0.2)
        assert request.max_acceleration_scaling_factor == pytest.approx(0.3)
        # plan from the currently monitored robot state
        assert request.start_state.is_diff is True
        assert goal.planning_options.plan_only is True

    def test_cartesian_request(self):
        from agents.utils.moveit import build_cartesian_request

        waypoints = [self._pose_stamped().pose, self._pose_stamped(z=0.6).pose]
        request = build_cartesian_request(
            waypoints,
            frame_id="base_link",
            group_name="panda_arm",
            link_name="ee_link",
            max_step=0.005,
            avoid_collisions=False,
        )
        assert request.header.frame_id == "base_link"
        assert request.group_name == "panda_arm"
        assert request.link_name == "ee_link"
        assert len(request.waypoints) == 2
        assert request.max_step == pytest.approx(0.005)
        assert request.avoid_collisions is False
        assert request.start_state.is_diff is True

    def test_error_string(self):
        from moveit_msgs.msg import MoveItErrorCodes

        from agents.utils.moveit import moveit_error_string

        assert moveit_error_string(MoveItErrorCodes.SUCCESS) == "SUCCESS"
        assert (
            moveit_error_string(MoveItErrorCodes.PLANNING_FAILED)
            == "PLANNING_FAILED"
        )
        assert moveit_error_string(-987654) == "UNKNOWN_ERROR(-987654)"


class TestGoalExecution:
    """Goal handling in main_action_callback, with mocked move_group clients."""

    @pytest.fixture
    def component(self, rclpy_init):
        pytest.importorskip("moveit_msgs.msg")
        from agents.components.moveit import MoveIt
        from agents.config import MoveItConfig

        comp = MoveIt(
            config=MoveItConfig(
                arm_group_name="panda_arm",
                gripper_group_name="hand",
                end_effector_link="panda_link8",
            ),
            component_name="test_moveit_goals",
        )
        comp.get_logger = MagicMock()
        comp.health_status = MagicMock()
        comp._named_targets = {
            "panda_arm": {"ready": {"panda_joint1": 0.0}},
            "hand": {"open": {"panda_finger_joint1": 0.035}},
        }
        return comp

    @staticmethod
    def _client(error_code=None, returned=True, accepted=True):
        """Action client handler that has already returned a result."""
        from moveit_msgs.msg import MoveItErrorCodes

        client = MagicMock()
        client.send_request.return_value = accepted
        client.action_returned = returned
        client.cancel_request.return_value = (True, "canceled")
        client.action_result = MagicMock(
            error_code=MagicMock(
                val=MoveItErrorCodes.SUCCESS if error_code is None else error_code
            )
        )
        return client

    @staticmethod
    def _goal_handle(goal, active=True, cancel_requested=False):
        handle = MagicMock()
        handle.request = goal
        handle.is_active = active
        handle.is_cancel_requested = cancel_requested
        return handle

    @staticmethod
    def _joint_goal():
        from automatika_embodied_agents.action import MoveManipulator

        goal = MoveManipulator.Goal()
        goal.mode = "joints"
        goal.joint_names = ["panda_joint1"]
        goal.joint_positions = [0.5]
        return goal

    def test_successful_motion_succeeds_goal(self, component):
        component._move_client = self._client()
        handle = self._goal_handle(self._joint_goal())

        result = component.main_action_callback(handle)

        assert result.success is True
        assert result.message == "SUCCESS"
        handle.succeed.assert_called_once()
        handle.abort.assert_not_called()
        component.health_status.set_healthy.assert_called_once()

    def test_planning_failure_aborts_with_reason(self, component):
        from moveit_msgs.msg import MoveItErrorCodes

        component._move_client = self._client(
            error_code=MoveItErrorCodes.PLANNING_FAILED
        )
        handle = self._goal_handle(self._joint_goal())

        result = component.main_action_callback(handle)

        assert result.success is False
        assert result.message == "PLANNING_FAILED"
        assert result.error_code == MoveItErrorCodes.PLANNING_FAILED
        handle.abort.assert_called_once()
        component.health_status.set_fail_component.assert_called_once()

    def test_cancel_is_propagated_to_move_group(self, component):
        # goal never returns, so the wait loop sees the cancel request
        component._move_client = self._client(returned=False)
        handle = self._goal_handle(self._joint_goal(), cancel_requested=True)

        component.main_action_callback(handle)

        component._move_client.cancel_request.assert_called_once()
        handle.canceled.assert_called_once()
        handle.abort.assert_not_called()

    def test_preempted_goal_is_not_transitioned(self, component):
        """A goal aborted by preemption is terminal — transitioning it again
        would be an invalid state transition."""
        component._move_client = self._client(returned=False)
        handle = self._goal_handle(self._joint_goal(), active=False)

        component.main_action_callback(handle)

        component._move_client.cancel_request.assert_called_once()
        handle.canceled.assert_not_called()
        handle.abort.assert_not_called()
        handle.succeed.assert_not_called()

    def test_rejected_goal_aborts(self, component):
        component._move_client = self._client(accepted=False)
        handle = self._goal_handle(self._joint_goal())

        result = component.main_action_callback(handle)

        assert result.success is False
        assert "did not accept" in result.message
        handle.abort.assert_called_once()

    def test_invalid_goal_aborts_without_sending(self, component):
        from automatika_embodied_agents.action import MoveManipulator

        component._move_client = self._client()
        handle = self._goal_handle(MoveManipulator.Goal())  # no target set

        result = component.main_action_callback(handle)

        assert "No motion target" in result.message
        component._move_client.send_request.assert_not_called()
        handle.abort.assert_called_once()

    def test_named_target_resolved_to_group_and_joints(self, component):
        from automatika_embodied_agents.action import MoveManipulator

        component._move_client = self._client()
        goal = MoveManipulator.Goal()
        goal.named_target = "ready"

        component.main_action_callback(self._goal_handle(goal))

        sent = component._move_client.send_request.call_args[0][0]
        assert sent.request.group_name == "panda_arm"
        constraint = sent.request.goal_constraints[0].joint_constraints[0]
        assert constraint.joint_name == "panda_joint1"

    def test_unknown_named_target_aborts(self, component):
        from automatika_embodied_agents.action import MoveManipulator

        component._move_client = self._client()
        goal = MoveManipulator.Goal()
        goal.named_target = "nowhere"

        result = component.main_action_callback(self._goal_handle(goal))

        assert "not defined in the robot SRDF" in result.message
        component._move_client.send_request.assert_not_called()

    def test_pose_goal_uses_configured_link_and_scalings(self, component):
        from automatika_embodied_agents.action import MoveManipulator

        component._move_client = self._client()
        goal = MoveManipulator.Goal()
        goal.mode = "pose"
        goal.target_pose.header.frame_id = "panda_link0"
        goal.target_pose.pose.orientation.w = 1.0
        goal.velocity_scaling = 0.5

        component.main_action_callback(self._goal_handle(goal))

        sent = component._move_client.send_request.call_args[0][0]
        position = sent.request.goal_constraints[0].position_constraints[0]
        assert position.link_name == "panda_link8"
        # per-goal override wins, unset scaling falls back to the config value
        assert sent.request.max_velocity_scaling_factor == pytest.approx(0.5)
        assert sent.request.max_acceleration_scaling_factor == pytest.approx(0.1)

    def test_mismatched_joint_arrays_abort(self, component):
        component._move_client = self._client()
        goal = self._joint_goal()
        goal.joint_positions = [0.5, 0.7]

        result = component.main_action_callback(self._goal_handle(goal))

        assert "same length" in result.message
        component._move_client.send_request.assert_not_called()


class TestCartesianGoals:
    @pytest.fixture
    def component(self, rclpy_init):
        pytest.importorskip("moveit_msgs.msg")
        from agents.components.moveit import MoveIt
        from agents.config import MoveItConfig

        comp = MoveIt(
            config=MoveItConfig(arm_group_name="panda_arm"),
            component_name="test_moveit_cartesian",
        )
        comp.get_logger = MagicMock()
        comp.health_status = MagicMock()
        return comp

    @staticmethod
    def _cartesian_goal(plan_only=False):
        from automatika_embodied_agents.action import MoveManipulator
        from geometry_msgs.msg import Pose

        goal = MoveManipulator.Goal()
        goal.mode = "cartesian"
        waypoint = Pose()
        waypoint.position.z = 0.5
        waypoint.orientation.w = 1.0
        goal.cartesian_waypoints = [waypoint]
        goal.frame_id = "panda_link0"
        goal.plan_only = plan_only
        return goal

    @staticmethod
    def _path_response(fraction):
        from moveit_msgs.msg import MoveItErrorCodes, RobotTrajectory

        return SimpleNamespace(
            fraction=fraction,
            solution=RobotTrajectory(),
            error_code=MoveItErrorCodes(val=MoveItErrorCodes.SUCCESS),
        )

    def test_full_path_is_executed(self, component):
        from moveit_msgs.msg import MoveItErrorCodes

        component._cartesian_client = MagicMock()
        component._cartesian_client.send_request.return_value = self._path_response(1.0)
        component._exec_client = TestGoalExecution._client()
        handle = TestGoalExecution._goal_handle(self._cartesian_goal())

        result = component.main_action_callback(handle)

        assert result.success is True
        assert result.cartesian_fraction == pytest.approx(1.0)
        assert result.error_code == MoveItErrorCodes.SUCCESS
        component._exec_client.send_request.assert_called_once()
        handle.succeed.assert_called_once()

    def test_partial_path_aborts_without_executing(self, component):
        component._cartesian_client = MagicMock()
        component._cartesian_client.send_request.return_value = self._path_response(0.4)
        component._exec_client = TestGoalExecution._client()
        handle = TestGoalExecution._goal_handle(self._cartesian_goal())

        result = component.main_action_callback(handle)

        assert result.success is False
        assert result.cartesian_fraction == pytest.approx(0.4)
        assert "below the configured threshold" in result.message
        # nothing is executed when too little of the path is achievable
        component._exec_client.send_request.assert_not_called()
        handle.abort.assert_called_once()

    def test_plan_only_does_not_execute(self, component):
        component._cartesian_client = MagicMock()
        component._cartesian_client.send_request.return_value = self._path_response(1.0)
        component._exec_client = TestGoalExecution._client()
        handle = TestGoalExecution._goal_handle(self._cartesian_goal(plan_only=True))

        result = component.main_action_callback(handle)

        assert result.success is True
        component._exec_client.send_request.assert_not_called()
        handle.succeed.assert_called_once()

    def test_no_service_response_aborts(self, component):
        component._cartesian_client = MagicMock()
        component._cartesian_client.send_request.return_value = None
        handle = TestGoalExecution._goal_handle(self._cartesian_goal())

        result = component.main_action_callback(handle)

        assert result.success is False
        assert "No response" in result.message
        handle.abort.assert_called_once()


class TestGripperActions:
    @staticmethod
    def _component(rclpy_init, **config_fields):
        from agents.components.moveit import MoveIt
        from agents.config import MoveItConfig

        comp = MoveIt(
            config=MoveItConfig(
                arm_group_name="panda_arm", gripper_group_name="hand", **config_fields
            ),
            component_name=f"test_moveit_gripper_{len(config_fields)}",
        )
        comp.get_logger = MagicMock()
        comp._named_targets = {"hand": {"open": {"f1": 0.035}, "close": {"f1": 0.0}}}
        return comp

    @pytest.fixture
    def move_group_component(self, rclpy_init):
        pytest.importorskip("moveit_msgs.msg")
        return self._component(rclpy_init)

    @pytest.fixture
    def controller_component(self, rclpy_init):
        pytest.importorskip("moveit_msgs.msg")
        return self._component(
            rclpy_init,
            gripper_mode="gripper_command",
            gripper_command_action="/gripper_controller/gripper_cmd",
        )

    def test_open_gripper_sends_named_target(self, move_group_component):
        comp = move_group_component
        comp._gripper_client = TestGoalExecution._client()

        message = comp.open_gripper.__wrapped__(comp)

        sent = comp._gripper_client.send_request.call_args[0][0]
        assert sent.request.group_name == "hand"
        assert sent.request.goal_constraints[0].joint_constraints[0].position == 0.035
        assert "open" in message

    def test_close_gripper_uses_controller_position(self, controller_component):
        comp = controller_component
        comp._gripper_client = TestGoalExecution._client()
        comp._gripper_client.action_result = MagicMock(spec=[])  # no error_code field

        comp.close_gripper.__wrapped__(comp)

        sent = comp._gripper_client.send_request.call_args[0][0]
        assert sent.command.position == pytest.approx(0.0)

    def test_set_gripper_needs_controller_mode(self, move_group_component):
        comp = move_group_component
        comp._gripper_client = TestGoalExecution._client()

        message = comp.set_gripper.__wrapped__(comp, 0.02)

        assert "gripper_command" in message
        comp._gripper_client.send_request.assert_not_called()

    def test_set_gripper_sends_position(self, controller_component):
        comp = controller_component
        comp._gripper_client = TestGoalExecution._client()
        comp._gripper_client.action_result = MagicMock(spec=[])

        comp.set_gripper.__wrapped__(comp, 0.02)

        sent = comp._gripper_client.send_request.call_args[0][0]
        assert sent.command.position == pytest.approx(0.02)

    def test_stop_motion_cancels_active_goals(self, move_group_component):
        comp = move_group_component
        comp._move_client = TestGoalExecution._client()
        comp._move_client.goal_accepted = True
        comp._exec_client = TestGoalExecution._client()
        comp._exec_client.goal_accepted = False
        comp._gripper_client = None

        message = comp.stop_motion.__wrapped__(comp)

        comp._move_client.cancel_request.assert_called_once()
        comp._exec_client.cancel_request.assert_not_called()
        assert "arm" in message

    def test_stop_motion_without_active_goals(self, move_group_component):
        comp = move_group_component
        comp._move_client = TestGoalExecution._client()
        comp._move_client.goal_accepted = False
        comp._exec_client = None
        comp._gripper_client = None

        assert "No motion" in comp.stop_motion.__wrapped__(comp)


class TestSceneBuilders:
    """Collision objects and scene diffs the planning scene is fed with."""

    @pytest.fixture(autouse=True)
    def _needs_moveit_msgs(self):
        pytest.importorskip("moveit_msgs.msg")

    def test_box_object_from_a_position(self):
        from moveit_msgs.msg import CollisionObject

        from agents.utils.moveit import build_collision_object

        obj = build_collision_object("crate", "base_link", (0.4, 0.0, 0.1), (0.2, 0.3, 0.2))

        assert obj.id == "crate" and obj.header.frame_id == "base_link"
        # zero stamp, so move_group does no TF lookup on an object already
        # given in the planning frame
        assert obj.header.stamp.sec == 0 and obj.header.stamp.nanosec == 0
        assert obj.operation == CollisionObject.ADD
        assert list(obj.primitives[0].dimensions) == [0.2, 0.3, 0.2]
        pose = obj.primitive_poses[0]
        assert pose.position.x == 0.4
        # a bare Pose carries an all-zero quaternion, which is not a rotation
        assert pose.orientation.w == 1.0

    def test_box_object_from_a_pose_keeps_its_orientation(self):
        from geometry_msgs.msg import Pose

        from agents.utils.moveit import build_collision_object

        center = Pose()
        center.position.z = 0.5
        center.orientation.z = center.orientation.w = 0.7071

        pose = build_collision_object("o", "map", center, (0.1, 0.1, 0.1)).primitive_poses[0]
        assert pose.orientation.z == pytest.approx(0.7071)
        # the input pose is copied, not aliased
        center.position.z = 9.9
        assert pose.position.z == 0.5

    def test_an_all_zero_quaternion_becomes_identity(self):
        from geometry_msgs.msg import Pose

        from agents.utils.moveit import build_collision_object

        obj = build_collision_object("o", "map", Pose(), (0.1, 0.1, 0.1))
        assert obj.primitive_poses[0].orientation.w == 1.0

    def test_flat_boxes_get_a_minimum_thickness(self):
        """A fronto-parallel surface lifts with no extent along the view
        axis, and a zero-thickness box is invisible to collision checking."""
        from agents.utils.moveit import build_collision_object

        obj = build_collision_object("o", "map", (0, 0, 0), (0.04, 0.02, 0.0))
        assert list(obj.primitives[0].dimensions) == [0.04, 0.02, 0.01]

    def test_remove_object(self):
        from moveit_msgs.msg import CollisionObject

        from agents.utils.moveit import build_remove_object

        obj = build_remove_object("crate")
        assert obj.id == "crate" and obj.operation == CollisionObject.REMOVE

    def test_scene_diff_never_replaces_the_scene(self):
        from agents.utils.moveit import (
            build_collision_object,
            build_remove_object,
            build_scene_diff,
        )

        scene = build_scene_diff(
            [
                build_collision_object("a", "map", (0, 0, 0), (0.1, 0.1, 0.1)),
                build_remove_object("b"),
            ]
        )
        assert scene.is_diff
        # without this, applying the diff would overwrite the robot state
        # move_group is monitoring
        assert scene.robot_state.is_diff
        assert [o.id for o in scene.world.collision_objects] == ["a", "b"]

    def test_attach_and_detach(self):
        from moveit_msgs.msg import CollisionObject

        from agents.utils.moveit import build_attached_object, build_detach_object

        attached = build_attached_object(
            "det__orange_0", "gripper_link", ["gripper_link", "jaw_link"]
        )
        assert attached.link_name == "gripper_link"
        assert attached.object.id == "det__orange_0"
        assert attached.object.operation == CollisionObject.ADD
        assert list(attached.touch_links) == ["gripper_link", "jaw_link"]

        detached = build_detach_object("det__orange_0")
        assert detached.object.operation == CollisionObject.REMOVE
        assert detached.link_name == ""


class TestDetectionObjects:
    """Detections3D messages turned into collision objects."""

    @pytest.fixture(autouse=True)
    def _needs_messages(self):
        pytest.importorskip("moveit_msgs.msg")
        pytest.importorskip("automatika_embodied_agents.msg")

    @staticmethod
    def _detections(entries, frame="base_link"):
        from automatika_embodied_agents.msg import Bbox3D, Detections3D

        msg = Detections3D()
        msg.header.frame_id = frame
        for label, score, center, size in entries:
            box = Bbox3D()
            # coerced: humble's generated messages reject ints on float fields
            box.center.position.x, box.center.position.y, box.center.position.z = (
                float(v) for v in center
            )
            box.center.orientation.w = 1.0
            box.size.x, box.size.y, box.size.z = (float(v) for v in size)
            msg.boxes.append(box)
            msg.labels.append(label)
            msg.scores.append(float(score))
        return msg

    def test_ids_rank_per_label_by_score(self):
        """The strongest orange stays det__orange_0 between refreshes of a
        static scene, without needing a tracker."""
        from agents.utils.moveit import collision_objects_from_detections

        objects = collision_objects_from_detections(
            self._detections([
                ("orange", 0.5, (0.1, 0.0, 0.0), (0.05, 0.05, 0.05)),
                ("bowl", 0.7, (0.2, 0.0, 0.0), (0.2, 0.2, 0.1)),
                ("orange", 0.9, (0.3, 0.0, 0.0), (0.06, 0.06, 0.06)),
            ])
        )
        by_id = {o.id: o for o in objects}
        assert set(by_id) == {"det__orange_0", "det__orange_1", "det__bowl_0"}
        # the higher scoring orange takes rank 0
        assert by_id["det__orange_0"].primitive_poses[0].position.x == pytest.approx(0.3)
        assert by_id["det__orange_1"].primitive_poses[0].position.x == pytest.approx(0.1)

    def test_objects_carry_the_message_frame_and_geometry(self):
        from agents.utils.moveit import collision_objects_from_detections

        (obj,) = collision_objects_from_detections(
            self._detections(
                [("cup", 0.8, (0.4, -0.1, 0.05), (0.06, 0.06, 0.12))], frame="world"
            )
        )
        assert obj.header.frame_id == "world"
        assert list(obj.primitives[0].dimensions) == pytest.approx([0.06, 0.06, 0.12])

    def test_padding_inflates_every_side(self):
        from agents.utils.moveit import collision_objects_from_detections

        (obj,) = collision_objects_from_detections(
            self._detections([("cup", 0.8, (0, 0, 0), (0.06, 0.06, 0.12))]),
            padding=0.01,
        )
        assert list(obj.primitives[0].dimensions) == pytest.approx([0.08, 0.08, 0.14])

    def test_labels_are_sanitized_and_defaulted(self):
        from agents.utils.moveit import collision_objects_from_detections

        objects = collision_objects_from_detections(
            self._detections([
                ("sports ball", 0.9, (0, 0, 0), (0.1, 0.1, 0.1)),
                ("", 0.8, (1, 0, 0), (0.1, 0.1, 0.1)),
            ])
        )
        assert {o.id for o in objects} == {"det__sports_ball_0", "det__object_0"}

    def test_no_detections_no_objects(self):
        from agents.utils.moveit import collision_objects_from_detections

        assert collision_objects_from_detections(self._detections([])) == []


class TestTouchLinks:
    """Which links may stay in contact with a grasped object."""

    SO101_JOINT_ONLY = """
    <robot name="so101">
      <group name="arm"><chain base_link="base_link" tip_link="gripper_frame_link"/></group>
      <group name="gripper"><joint name="gripper"/></group>
      <end_effector name="eef" parent_link="gripper_frame_link" group="gripper"/>
      <disable_collisions link1="wrist_link" link2="gripper_link" reason="Adjacent"/>
      <disable_collisions link1="gripper_link" link2="moving_jaw_so101_v1_link" reason="Adjacent"/>
    </robot>
    """

    SO101_WITH_LINKS = SO101_JOINT_ONLY.replace(
        '<group name="gripper"><joint name="gripper"/></group>',
        '<group name="gripper"><joint name="gripper"/>'
        '<link name="gripper_frame_link"/><link name="gripper_link"/>'
        '<link name="moving_jaw_so101_v1_link"/></group>',
    )

    def test_explicit_config_wins(self):
        from agents.utils.moveit import derive_touch_links

        assert derive_touch_links(
            self.SO101_WITH_LINKS, "eef", explicit=["a", "b", "a"]
        ) == ["a", "b"]

    def test_links_declared_in_the_gripper_group(self):
        from agents.utils.moveit import derive_touch_links

        assert derive_touch_links(
            self.SO101_WITH_LINKS, "eef", gripper_group="gripper"
        ) == ["gripper_frame_link", "gripper_link", "moving_jaw_so101_v1_link"]

    def test_adjacent_partners_of_the_end_effector_parent(self):
        """The heuristic path, for SRDFs that declare neither links nor an
        explicit list. Only physically connected pairs describe the gripper;
        'Never' pairs span the whole robot."""
        from agents.utils.moveit import derive_touch_links

        srdf = """
        <robot>
          <end_effector name="eef" parent_link="hand" group="g"/>
          <disable_collisions link1="hand" link2="left_finger" reason="Adjacent"/>
          <disable_collisions link1="right_finger" link2="hand" reason="Adjacent"/>
          <disable_collisions link1="hand" link2="base" reason="Never"/>
        </robot>
        """
        assert derive_touch_links(srdf, "hand", gripper_group="g") == [
            "hand",
            "left_finger",
            "right_finger",
        ]

    def test_joint_only_group_without_adjacency_falls_back(self):
        """The SO-101's own shape: the end effector parent appears in no
        disable_collisions pair, so nothing can be derived and the configured
        link is returned alone."""
        from agents.utils.moveit import derive_touch_links

        assert derive_touch_links(
            self.SO101_JOINT_ONLY, "gripper_frame_link", gripper_group="gripper"
        ) == ["gripper_frame_link"]

    def test_no_srdf_falls_back(self):
        from agents.utils.moveit import derive_touch_links

        assert derive_touch_links(None, "tool0") == ["tool0"]


class TestSceneInputs:
    """The component takes detected objects as an input topic, to feed the
    planning scene with."""

    @pytest.fixture(autouse=True)
    def _needs_moveit_msgs(self):
        pytest.importorskip("moveit_msgs.msg")

    @staticmethod
    def _config():
        from agents.config import MoveItConfig

        return MoveItConfig(arm_group_name="arm")

    def test_a_detections_input_is_accepted_and_recorded(self, rclpy_init):
        from agents.components import MoveIt
        from agents.ros import Topic

        component = MoveIt(
            config=self._config(),
            component_name="m_scene_in",
            inputs=[Topic(name="detections_3d", msg_type="Detections3D")],
        )
        assert component._detections_topic.name == "detections_3d"

    def test_other_input_types_are_rejected(self, rclpy_init):
        """Regression: this used to raise AttributeError instead of a clear
        TypeError, since `allowed_inputs` was never assigned."""
        from agents.components import MoveIt
        from agents.ros import Topic

        with pytest.raises(TypeError, match="allowed"):
            MoveIt(
                config=self._config(),
                component_name="m_scene_bad",
                inputs=[Topic(name="image", msg_type="Image")],
            )

    def test_no_inputs_remains_valid(self, rclpy_init):
        from agents.components import MoveIt

        component = MoveIt(config=self._config(), component_name="m_scene_none")
        assert component._detections_topic is None

    def test_the_executable_reconstruction_shape_is_tolerated(self, rclpy_init):
        """The launch executable's generic branch passes model_client,
        db_client and trigger to every component it rebuilds in a child
        process; MoveIt takes none of them and must swallow them."""
        from agents.components import MoveIt
        from agents.ros import Topic

        component = MoveIt(
            inputs=[Topic(name="detections_3d", msg_type="Detections3D")],
            outputs=None,
            model_client=None,
            db_client=None,
            trigger=1.0,
            config=self._config(),
            component_name="m_scene_relaunch",
            config_file=None,
        )
        assert component._detections_topic.name == "detections_3d"


class TestSceneServiceClients:
    """The move_group planning scene services the component connects to."""

    @pytest.fixture(autouse=True)
    def _needs_moveit_msgs(self):
        pytest.importorskip("moveit_msgs.msg")

    def test_scene_clients_wire_to_the_move_group_services(
        self, rclpy_init, monkeypatch
    ):
        from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
        from std_srvs.srv import Empty

        import agents.components.moveit as moveit_module
        from agents.components import MoveIt
        from agents.config import MoveItConfig

        component = MoveIt(
            config=MoveItConfig(arm_group_name="arm", move_group_namespace="/bot"),
            component_name="m_scene_clients",
        )

        created = []

        def _fake_handler(_component, config):
            created.append(config)
            return MagicMock(config=config)

        monkeypatch.setattr(moveit_module, "ServiceClientHandler", _fake_handler)
        monkeypatch.setattr(
            "agents.components.component_base.Component.create_all_service_clients",
            lambda _: None,
        )

        component.create_all_service_clients()

        by_name = {config.name: config.srv_type for config in created}
        assert by_name["/bot/apply_planning_scene"] is ApplyPlanningScene
        assert by_name["/bot/get_planning_scene"] is GetPlanningScene
        assert by_name["/bot/clear_octomap"] is Empty
        assert component._apply_scene_client is not None
        assert component._get_scene_client is not None
        assert component._clear_octomap_client is not None

    def test_destroy_resets_the_scene_clients(self, rclpy_init, monkeypatch):
        from agents.components import MoveIt
        from agents.config import MoveItConfig

        component = MoveIt(
            config=MoveItConfig(arm_group_name="arm"),
            component_name="m_scene_destroy",
        )
        for name in ("_apply_scene_client", "_get_scene_client", "_clear_octomap_client"):
            setattr(component, name, MagicMock())
        component.destroy_client = MagicMock()
        monkeypatch.setattr(
            "agents.components.component_base.Component.destroy_all_service_clients",
            lambda _: None,
        )

        component.destroy_all_service_clients()

        assert component.destroy_client.call_count == 3
        assert component._apply_scene_client is None
        assert component._get_scene_client is None
        assert component._clear_octomap_client is None


class TestSceneActions:
    """The component actions that manage collision objects in the scene."""

    @pytest.fixture(autouse=True)
    def _needs_moveit_msgs(self):
        pytest.importorskip("moveit_msgs.msg")

    @pytest.fixture
    def component(self, rclpy_init):
        from agents.components import MoveIt
        from agents.config import MoveItConfig

        comp = MoveIt(
            config=MoveItConfig(
                arm_group_name="arm", pose_reference_frame="base_link"
            ),
            component_name="m_scene_actions",
        )
        comp.get_logger = MagicMock()
        comp._apply_scene_client = MagicMock()
        comp._apply_scene_client.send_request.return_value = SimpleNamespace(
            success=True
        )
        return comp

    @staticmethod
    def _applied_scene(component):
        request = component._apply_scene_client.send_request.call_args[0][0]
        return request.scene

    def test_add_sends_one_diff_and_tracks_the_object(self, component):
        from moveit_msgs.msg import CollisionObject

        message = component.add_collision_object.__wrapped__(
            component, "crate", [0.4, 0.0, 0.1], [0.2, 0.3, 0.2]
        )

        scene = self._applied_scene(component)
        assert scene.is_diff and scene.robot_state.is_diff
        (obj,) = scene.world.collision_objects
        assert obj.id == "crate" and obj.operation == CollisionObject.ADD
        # empty frame falls back to the configured reference frame
        assert obj.header.frame_id == "base_link"
        assert list(obj.primitives[0].dimensions) == [0.2, 0.3, 0.2]
        assert component._scene_objects["crate"]["source"] == "manual"
        assert "Added" in message

    def test_add_applies_the_thickness_floor(self, component):
        component.config.min_object_thickness = 0.05
        component.add_collision_object.__wrapped__(
            component, "sheet", [0, 0, 0], [0.4, 0.4, 0.0]
        )
        (obj,) = self._applied_scene(component).world.collision_objects
        assert list(obj.primitives[0].dimensions) == [0.4, 0.4, 0.05]

    def test_add_reports_failure_and_tracks_nothing(self, component):
        component._apply_scene_client.send_request.return_value = SimpleNamespace(
            success=False
        )
        message = component.add_collision_object.__wrapped__(
            component, "crate", [0, 0, 0], [0.1, 0.1, 0.1]
        )
        assert "Could not" in message
        assert "crate" not in component._scene_objects

    def test_add_rejects_malformed_geometry(self, component):
        with pytest.raises(ValueError, match="3 values"):
            component.add_collision_object.__wrapped__(
                component, "crate", [0, 0], [0.1, 0.1, 0.1]
            )

    def test_remove_sends_remove_and_forgets_the_object(self, component):
        from moveit_msgs.msg import CollisionObject

        component._scene_objects["crate"] = {"source": "manual", "last_seen": 0.0}
        message = component.remove_collision_object.__wrapped__(component, "crate")

        (obj,) = self._applied_scene(component).world.collision_objects
        assert obj.id == "crate" and obj.operation == CollisionObject.REMOVE
        assert "crate" not in component._scene_objects
        assert "Removed" in message

    def test_clear_removes_tracked_objects_in_one_diff(self, component):
        component._scene_objects = {
            "crate": {"source": "manual", "last_seen": 0.0},
            "det__orange_0": {"source": "detection", "last_seen": 0.0},
        }
        message = component.clear_collision_objects.__wrapped__(component)

        removed = {o.id for o in self._applied_scene(component).world.collision_objects}
        assert removed == {"crate", "det__orange_0"}
        assert component._scene_objects == {}
        assert "2" in message

    def test_clear_detections_only_keeps_manual_objects(self, component):
        component._scene_objects = {
            "crate": {"source": "manual", "last_seen": 0.0},
            "det__orange_0": {"source": "detection", "last_seen": 0.0},
        }
        component.clear_collision_objects.__wrapped__(component, detections_only=True)

        removed = {o.id for o in self._applied_scene(component).world.collision_objects}
        assert removed == {"det__orange_0"}
        assert set(component._scene_objects) == {"crate"}

    def test_clear_with_nothing_tracked_sends_nothing(self, component):
        message = component.clear_collision_objects.__wrapped__(component)
        component._apply_scene_client.send_request.assert_not_called()
        assert "no objects" in message

    def test_list_reads_the_scene_back_from_move_group(self, component):
        """move_group is authoritative: it also shows objects other tools
        placed there."""
        component._get_scene_client = MagicMock()
        component._get_scene_client.send_request.return_value = SimpleNamespace(
            scene=SimpleNamespace(
                world=SimpleNamespace(
                    collision_objects=[
                        SimpleNamespace(id="det__orange_0"),
                        SimpleNamespace(id="table"),
                    ]
                )
            )
        )
        message = component.list_collision_objects.__wrapped__(component)
        assert "det__orange_0, table" in message

    def test_list_falls_back_to_tracked_objects(self, component):
        component._get_scene_client = MagicMock()
        component._get_scene_client.send_request.return_value = None
        component._scene_objects = {"crate": {"source": "manual", "last_seen": 0.0}}
        message = component.list_collision_objects.__wrapped__(component)
        assert "did not answer" in message and "crate" in message

    def test_clear_octomap(self, component):
        from std_srvs.srv import Empty

        component._clear_octomap_client = MagicMock()
        component._clear_octomap_client.send_request.return_value = Empty.Response()
        message = component.clear_octomap.__wrapped__(component)
        assert isinstance(
            component._clear_octomap_client.send_request.call_args[0][0],
            Empty.Request,
        )
        assert "Cleared" in message

    def test_scene_failure_degrades_without_raising(self, component):
        """A scene problem must never take the component down."""
        component._apply_scene_client = None
        message = component.add_collision_object.__wrapped__(
            component, "crate", [0, 0, 0], [0.1, 0.1, 0.1]
        )
        assert "Could not" in message


class TestAttachDetach:
    """Attaching a grasped object to the robot and letting go of it again."""

    GRIPPER_SRDF = """
    <robot name="bot">
      <group name="gripper">
        <joint name="jaw"/>
        <link name="hand"/><link name="left_finger"/><link name="right_finger"/>
      </group>
      <end_effector name="eef" parent_link="hand" group="gripper"/>
    </robot>
    """

    @pytest.fixture(autouse=True)
    def _needs_moveit_msgs(self):
        pytest.importorskip("moveit_msgs.msg")

    @pytest.fixture
    def component(self, rclpy_init):
        from agents.components import MoveIt
        from agents.config import MoveItConfig

        comp = MoveIt(
            config=MoveItConfig(
                arm_group_name="arm",
                gripper_group_name="gripper",
                end_effector_link="hand",
            ),
            component_name="m_attach",
        )
        comp.get_logger = MagicMock()
        comp._apply_scene_client = MagicMock()
        comp._apply_scene_client.send_request.return_value = SimpleNamespace(
            success=True
        )
        comp._srdf = self.GRIPPER_SRDF
        return comp

    @staticmethod
    def _applied_scenes(component):
        return [
            call[0][0].scene
            for call in component._apply_scene_client.send_request.call_args_list
        ]

    def test_attach_defaults_to_the_end_effector_and_srdf_touch_links(
        self, component
    ):
        from moveit_msgs.msg import CollisionObject

        component._scene_objects["det__orange_0"] = {
            "source": "detection",
            "last_seen": 0.0,
        }
        message = component.attach_object.__wrapped__(component, "det__orange_0")

        (scene,) = self._applied_scenes(component)
        (attached,) = scene.robot_state.attached_collision_objects
        assert attached.link_name == "hand"
        assert attached.object.id == "det__orange_0"
        assert attached.object.operation == CollisionObject.ADD
        assert list(attached.touch_links) == ["hand", "left_finger", "right_finger"]
        assert component._scene_objects["det__orange_0"]["attached"] == "hand"
        assert "Attached" in message

    def test_explicit_touch_links_win_over_config_and_srdf(self, component):
        component.config.touch_links = ["from_config"]
        component.attach_object.__wrapped__(
            component, "o", touch_links=["only_this"]
        )
        (scene,) = self._applied_scenes(component)
        assert list(
            scene.robot_state.attached_collision_objects[0].touch_links
        ) == ["only_this"]

    def test_configured_touch_links_win_over_the_srdf(self, component):
        component.config.touch_links = ["from_config"]
        component.attach_object.__wrapped__(component, "o")
        (scene,) = self._applied_scenes(component)
        assert list(
            scene.robot_state.attached_collision_objects[0].touch_links
        ) == ["from_config"]

    def test_the_srdf_is_fetched_lazily_when_not_cached(self, component):
        """Attaching before any named-target use still finds the gripper."""
        component._srdf = None
        component._param_client = MagicMock()
        component._param_client.send_request_from_dict.return_value = SimpleNamespace(
            values=[SimpleNamespace(string_value=self.GRIPPER_SRDF)]
        )
        component.attach_object.__wrapped__(component, "o")
        (scene,) = self._applied_scenes(component)
        assert list(
            scene.robot_state.attached_collision_objects[0].touch_links
        ) == ["hand", "left_finger", "right_finger"]

    def test_attach_without_any_link_reports_it(self, component):
        component.config.end_effector_link = ""
        message = component.attach_object.__wrapped__(component, "o")
        assert "No link" in message
        component._apply_scene_client.send_request.assert_not_called()

    def test_attach_failure_marks_nothing(self, component):
        component._apply_scene_client.send_request.return_value = SimpleNamespace(
            success=False
        )
        component._scene_objects["o"] = {"source": "manual", "last_seen": 0.0}
        message = component.attach_object.__wrapped__(component, "o")
        assert "Could not" in message
        assert "attached" not in component._scene_objects["o"]

    def test_detach_returns_the_object_to_the_scene(self, component):
        from moveit_msgs.msg import CollisionObject

        component._scene_objects["o"] = {
            "source": "manual",
            "last_seen": 0.0,
            "attached": "hand",
        }
        message = component.detach_object.__wrapped__(component, "o")

        (scene,) = self._applied_scenes(component)
        (attached,) = scene.robot_state.attached_collision_objects
        assert attached.object.operation == CollisionObject.REMOVE
        # empty link searches every link for the object
        assert attached.link_name == ""
        assert "attached" not in component._scene_objects["o"]
        assert "remains in the scene" in message

    def test_detach_with_remove_deletes_in_a_second_change(self, component):
        component._scene_objects["o"] = {
            "source": "manual",
            "last_seen": 0.0,
            "attached": "hand",
        }
        message = component.detach_object.__wrapped__(component, "o", remove=True)

        detach_scene, remove_scene = self._applied_scenes(component)
        assert detach_scene.robot_state.attached_collision_objects
        (removed,) = remove_scene.world.collision_objects
        assert removed.id == "o"
        assert "o" not in component._scene_objects
        assert "removed" in message

    def test_detach_remove_reports_a_partial_failure(self, component):
        component._apply_scene_client.send_request.side_effect = [
            SimpleNamespace(success=True),
            SimpleNamespace(success=False),
        ]
        message = component.detach_object.__wrapped__(component, "o", remove=True)
        assert "Detached" in message and "could not remove" in message


class TestSceneRefresh:
    """Detections pushed into the planning scene, with TTL bookkeeping."""

    @pytest.fixture(autouse=True)
    def _needs_messages(self):
        pytest.importorskip("moveit_msgs.msg")
        pytest.importorskip("automatika_embodied_agents.msg")

    @pytest.fixture
    def component(self, rclpy_init):
        from agents.components import MoveIt
        from agents.config import MoveItConfig
        from agents.ros import Topic

        comp = MoveIt(
            config=MoveItConfig(arm_group_name="arm", scene_object_ttl=5.0),
            component_name="m_refresh",
            inputs=[Topic(name="d3", msg_type="Detections3D")],
        )
        comp.get_logger = MagicMock()
        comp._apply_scene_client = MagicMock()
        comp._apply_scene_client.send_request.return_value = SimpleNamespace(
            success=True
        )
        return comp

    @staticmethod
    def _wire(component, message):
        callback = MagicMock()
        callback.get_output.return_value = message
        component.callbacks = {"d3": callback}
        return callback

    @staticmethod
    def _applied_scene(component):
        request = component._apply_scene_client.send_request.call_args[0][0]
        return request.scene

    def _detections(self, entries):
        return TestDetectionObjects._detections(entries)

    def test_detections_become_tracked_scene_objects(self, component):
        from moveit_msgs.msg import CollisionObject

        callback = self._wire(
            component,
            self._detections([
                ("orange", 0.9, (0.3, 0.0, 0.05), (0.06, 0.06, 0.06)),
                ("bowl", 0.7, (0.5, 0.1, 0.05), (0.2, 0.2, 0.1)),
            ]),
        )
        message = component.update_planning_scene.__wrapped__(component)

        # the message is requested, not the callback's prompt-context default
        assert callback.get_output.call_args.kwargs.get("get_msg") is True
        scene = self._applied_scene(component)
        by_id = {o.id: o for o in scene.world.collision_objects}
        assert set(by_id) == {"det__orange_0", "det__bowl_0"}
        assert all(
            o.operation == CollisionObject.ADD for o in by_id.values()
        )
        assert component._scene_objects["det__orange_0"]["source"] == "detection"
        assert "2 detected object(s)" in message

    def test_without_a_detections_input_it_says_so(self, rclpy_init):
        from agents.components import MoveIt
        from agents.config import MoveItConfig

        comp = MoveIt(
            config=MoveItConfig(arm_group_name="arm"),
            component_name="m_refresh_none",
        )
        assert "No detections input" in comp.update_planning_scene.__wrapped__(comp)

    def test_before_any_message_it_says_so(self, component):
        self._wire(component, None)
        assert "No detections have been received" in (
            component.update_planning_scene.__wrapped__(component)
        )

    def test_stale_objects_are_removed_in_the_same_diff(self, component):
        from moveit_msgs.msg import CollisionObject

        import time as time_module

        component._scene_objects["det__cup_0"] = {
            "source": "detection",
            "last_seen": time_module.time() - 10.0,  # past the 5 s TTL
        }
        self._wire(
            component,
            self._detections([("orange", 0.9, (0.3, 0, 0), (0.06, 0.06, 0.06))]),
        )
        message = component.update_planning_scene.__wrapped__(component)

        scene = self._applied_scene(component)
        ops = {o.id: o.operation for o in scene.world.collision_objects}
        assert ops["det__orange_0"] == CollisionObject.ADD
        assert ops["det__cup_0"] == CollisionObject.REMOVE
        assert "det__cup_0" not in component._scene_objects
        assert "removed 1 stale" in message

    def test_recently_seen_objects_survive_a_dropout(self, component):
        component._scene_objects["det__cup_0"] = {
            "source": "detection",
            "last_seen": __import__("time").time() - 1.0,  # within the TTL
        }
        self._wire(
            component,
            self._detections([("orange", 0.9, (0.3, 0, 0), (0.06, 0.06, 0.06))]),
        )
        component.update_planning_scene.__wrapped__(component)

        ids = {o.id for o in self._applied_scene(component).world.collision_objects}
        assert "det__cup_0" not in ids  # no REMOVE sent
        assert "det__cup_0" in component._scene_objects

    def test_manual_objects_are_never_ttl_evicted(self, component):
        component._scene_objects["table"] = {
            "source": "manual",
            "last_seen": __import__("time").time() - 100.0,
        }
        self._wire(
            component,
            self._detections([("orange", 0.9, (0.3, 0, 0), (0.06, 0.06, 0.06))]),
        )
        component.update_planning_scene.__wrapped__(component)

        ids = {o.id for o in self._applied_scene(component).world.collision_objects}
        assert "table" not in ids
        assert "table" in component._scene_objects

    def test_the_scene_freezes_while_an_object_is_held(self, component):
        """Detection ids are score ranks, not identities: refreshing while
        holding could hide a real obstacle behind the held object's id, or
        re-add the held object as a world box at the gripper. Freeze, and
        reconcile after release."""
        component._scene_objects["det__orange_0"] = {
            "source": "detection",
            "last_seen": __import__("time").time() - 100.0,
            "attached": "hand",
        }
        # three oranges: the held one, seen at the gripper, and two on the
        # table, the stronger of which ranks 0 and wears the held one's id
        self._wire(
            component,
            self._detections([
                ("orange", 0.9, (0.4, 0.0, 0.05), (0.06, 0.06, 0.06)),
                ("orange", 0.8, (0.5, 0.1, 0.05), (0.06, 0.06, 0.06)),
                ("orange", 0.7, (0.2, 0.3, 0.2), (0.06, 0.06, 0.06)),
            ]),
        )
        message = component.update_planning_scene.__wrapped__(component)

        assert "held" in message
        component._apply_scene_client.send_request.assert_not_called()
        assert component._scene_objects["det__orange_0"]["attached"] == "hand"

    def test_the_first_refresh_after_release_reconciles(self, component):
        entry = {
            "source": "detection",
            "last_seen": __import__("time").time() - 100.0,
            "attached": "hand",
        }
        component._scene_objects["det__orange_0"] = entry
        self._wire(
            component,
            self._detections([
                ("orange", 0.9, (0.4, 0.0, 0.05), (0.06, 0.06, 0.06)),
                ("orange", 0.8, (0.5, 0.1, 0.05), (0.06, 0.06, 0.06)),
            ]),
        )
        component.update_planning_scene.__wrapped__(component)
        component._apply_scene_client.send_request.assert_not_called()
        entry.pop("attached")

        component.update_planning_scene.__wrapped__(component)

        scene = self._applied_scene(component)
        ids = {o.id for o in scene.world.collision_objects}
        assert ids == {"det__orange_0", "det__orange_1"}

    def test_partition_still_exempts_attached_as_a_race_guard(self, component):
        """The freeze normally keeps attached entries out of a refresh; the
        partition exemption remains for an attach landing mid-refresh."""
        from agents.utils.moveit import collision_objects_from_detections

        component._scene_objects["det__orange_0"] = {
            "source": "detection",
            "last_seen": __import__("time").time() - 100.0,
            "attached": "hand",
        }
        objects = collision_objects_from_detections(
            self._detections([("orange", 0.9, (0.4, 0, 0), (0.06, 0.06, 0.06))])
        )
        added, stale = component._partition_scene_changes(
            objects, __import__("time").time()
        )
        assert "det__orange_0" not in {o.id for o in added}
        assert "det__orange_0" not in stale

    def test_nothing_seen_and_nothing_stale_is_a_no_op(self, component):
        self._wire(component, self._detections([]))
        message = component.update_planning_scene.__wrapped__(component)
        component._apply_scene_client.send_request.assert_not_called()
        assert "already up to date" in message

    def test_apply_failure_leaves_bookkeeping_untouched(self, component):
        component._apply_scene_client.send_request.return_value = SimpleNamespace(
            success=False
        )
        self._wire(
            component,
            self._detections([("orange", 0.9, (0.3, 0, 0), (0.06, 0.06, 0.06))]),
        )
        message = component.update_planning_scene.__wrapped__(component)
        assert "Could not" in message
        assert component._scene_objects == {}


class TestSceneUpdateModes:
    """When detections reach the scene without being asked for explicitly."""

    @pytest.fixture(autouse=True)
    def _needs_moveit_msgs(self):
        pytest.importorskip("moveit_msgs.msg")

    @pytest.fixture
    def component(self, rclpy_init):
        from agents.components import MoveIt
        from agents.config import MoveItConfig
        from agents.ros import Topic

        comp = MoveIt(
            config=MoveItConfig(arm_group_name="panda_arm"),
            component_name="m_modes",
            inputs=[Topic(name="d3", msg_type="Detections3D")],
        )
        comp.get_logger = MagicMock()
        comp.health_status = MagicMock()
        return comp

    def test_on_goal_refreshes_before_planning(self, component):
        component.config.scene_update_mode = "on_goal"
        component._refresh_scene_from_detections = MagicMock(
            return_value="Planning scene updated with 1 detected object(s)"
        )
        component._move_client = TestGoalExecution._client()
        handle = TestGoalExecution._goal_handle(TestGoalExecution._joint_goal())

        result = component.main_action_callback(handle)

        component._refresh_scene_from_detections.assert_called_once()
        # the refresh is announced through the goal's feedback
        feedback = handle.publish_feedback.call_args_list[0][0][0]
        assert feedback.state == "UPDATING_SCENE"
        assert result.success

    def test_on_goal_scene_failure_never_aborts_the_motion(self, component):
        component.config.scene_update_mode = "on_goal"
        component._refresh_scene_from_detections = MagicMock(
            return_value="Could not update the planning scene"
        )
        component._move_client = TestGoalExecution._client()
        handle = TestGoalExecution._goal_handle(TestGoalExecution._joint_goal())

        result = component.main_action_callback(handle)

        assert result.success  # planned against the previous scene

    def test_manual_mode_does_not_refresh_on_goals(self, component):
        component._refresh_scene_from_detections = MagicMock()
        component._move_client = TestGoalExecution._client()
        handle = TestGoalExecution._goal_handle(TestGoalExecution._joint_goal())

        component.main_action_callback(handle)

        component._refresh_scene_from_detections.assert_not_called()

    def test_continuous_mode_starts_a_timer_on_activation(
        self, component, monkeypatch
    ):
        component.config.scene_update_mode = "continuous"
        component.config.scene_update_rate = 2.0
        component.create_timer = MagicMock()
        monkeypatch.setattr(
            "ros_sugar.core.component.BaseComponent.custom_on_activate",
            lambda _: None,
        )

        component.custom_on_activate()

        period = component.create_timer.call_args[0][0]
        assert period == pytest.approx(0.5)
        assert component._scene_timer is component.create_timer.return_value

    def test_modes_without_a_detections_input_warn_instead(
        self, rclpy_init, monkeypatch
    ):
        from agents.components import MoveIt
        from agents.config import MoveItConfig

        comp = MoveIt(
            config=MoveItConfig(
                arm_group_name="arm", scene_update_mode="continuous"
            ),
            component_name="m_modes_warn",
        )
        comp.get_logger = MagicMock()
        comp.create_timer = MagicMock()
        monkeypatch.setattr(
            "ros_sugar.core.component.BaseComponent.custom_on_activate",
            lambda _: None,
        )

        comp.custom_on_activate()

        comp.create_timer.assert_not_called()
        assert "no Detections3D input" in comp.get_logger().warning.call_args[0][0]

    def test_deactivation_stops_the_timer(self, component, monkeypatch):
        component._scene_timer = MagicMock()
        component.destroy_timer = MagicMock()
        monkeypatch.setattr(
            "ros_sugar.core.component.BaseComponent.custom_on_deactivate",
            lambda _: None,
        )

        component.custom_on_deactivate()

        component.destroy_timer.assert_called_once()
        assert component._scene_timer is None

    # ------------------------------------------------------------------
    # The continuous tick's skip guard
    # ------------------------------------------------------------------

    @staticmethod
    def _wire_message(component, message):
        callback = MagicMock()
        callback.get_output.return_value = message
        component.callbacks = {"d3": callback}

    def test_tick_refreshes_on_a_new_message(self, component):
        component._refresh_scene_from_detections = MagicMock(return_value="ok")
        self._wire_message(component, object())

        component._scene_refresh_tick()

        component._refresh_scene_from_detections.assert_called_once()

    def test_tick_skips_an_already_applied_message(self, component):
        component._refresh_scene_from_detections = MagicMock(return_value="ok")
        message = object()
        self._wire_message(component, message)

        component._scene_refresh_tick()
        component._scene_refresh_tick()

        assert component._refresh_scene_from_detections.call_count == 1

    def test_tick_refreshes_a_stale_scene_even_without_a_new_message(
        self, component
    ):
        """Objects due for eviction get their removal without waiting for the
        detector to publish again."""
        component._refresh_scene_from_detections = MagicMock(return_value="ok")
        message = object()
        self._wire_message(component, message)
        component._scene_refresh_tick()

        component._scene_objects["det__cup_0"] = {
            "source": "detection",
            "last_seen": __import__("time").time() - 100.0,
        }
        component._scene_refresh_tick()

        assert component._refresh_scene_from_detections.call_count == 2

    def test_tick_without_any_message_does_nothing(self, component):
        component._refresh_scene_from_detections = MagicMock()
        self._wire_message(component, None)

        component._scene_refresh_tick()

        component._refresh_scene_from_detections.assert_not_called()


class TestSceneGeometryCache:
    """Scene entries carry the geometry they were applied with, so targets
    can be resolved by name without asking move_group."""

    @pytest.fixture(autouse=True)
    def _needs_messages(self):
        pytest.importorskip("moveit_msgs.msg")
        pytest.importorskip("automatika_embodied_agents.msg")

    def test_manual_objects_cache_their_applied_geometry(self, rclpy_init):
        from agents.components import MoveIt
        from agents.config import MoveItConfig

        comp = MoveIt(
            config=MoveItConfig(arm_group_name="arm", min_object_thickness=0.05),
            component_name="m_geo_manual",
        )
        comp.get_logger = MagicMock()
        comp._apply_scene_client = MagicMock()
        comp._apply_scene_client.send_request.return_value = SimpleNamespace(
            success=True
        )

        comp.add_collision_object.__wrapped__(
            comp, "sheet", [0.4, 0.0, 0.1], [0.3, 0.3, 0.0]
        )

        entry = comp._scene_objects["sheet"]
        assert entry["center"] == (0.4, 0.0, 0.1)
        # the cached size is what was applied: thickness floor included
        assert entry["size"] == (0.3, 0.3, 0.05)

    def test_detection_objects_geometry_follows_the_object(self, rclpy_init):
        from agents.components import MoveIt
        from agents.config import MoveItConfig
        from agents.ros import Topic

        comp = MoveIt(
            config=MoveItConfig(arm_group_name="arm"),
            component_name="m_geo_det",
            inputs=[Topic(name="d3", msg_type="Detections3D")],
        )
        comp.get_logger = MagicMock()
        comp._apply_scene_client = MagicMock()
        comp._apply_scene_client.send_request.return_value = SimpleNamespace(
            success=True
        )

        def wire(center):
            callback = MagicMock()
            callback.get_output.return_value = TestDetectionObjects._detections(
                [("mug", 0.9, center, (0.06, 0.06, 0.1))]
            )
            comp.callbacks = {"d3": callback}

        wire((0.3, 0.0, 0.05))
        comp.update_planning_scene.__wrapped__(comp)
        assert comp._scene_objects["det__mug_0"]["center"] == (0.3, 0.0, 0.05)

        # the mug moved: a refresh moves the cached geometry with it
        wire((0.5, 0.1, 0.05))
        comp.update_planning_scene.__wrapped__(comp)
        entry = comp._scene_objects["det__mug_0"]
        assert entry["center"] == (0.5, 0.1, 0.05)
        assert entry["size"] == (0.06, 0.06, 0.1)


class TestPickSequence:
    """The pick mode: approach, open, descend, grasp, attach, lift."""

    @pytest.fixture(autouse=True)
    def _needs_moveit_msgs(self):
        pytest.importorskip("moveit_msgs.msg")

    @pytest.fixture
    def component(self, rclpy_init):
        from agents.components import MoveIt
        from agents.config import MoveItConfig

        comp = MoveIt(
            config=MoveItConfig(
                arm_group_name="arm",
                gripper_group_name="gripper",
                end_effector_link="hand",
                pose_reference_frame="base",
                approach_clearance=0.1,
                touch_links=["hand", "jaw"],
            ),
            component_name="m_pick",
        )
        comp.get_logger = MagicMock()
        comp.health_status = MagicMock()
        comp._named_targets = {
            "gripper": {"open": {"jaw": 1.0}, "close": {"jaw": 0.0}}
        }
        comp._move_client = TestGoalExecution._client()
        comp._exec_client = TestGoalExecution._client()
        comp._gripper_client = TestGoalExecution._client()
        comp._cartesian_client = MagicMock()
        comp._cartesian_client.send_request.return_value = SimpleNamespace(
            fraction=1.0, solution=_trajectory(), error_code=SimpleNamespace(val=1)
        )
        comp._apply_scene_client = MagicMock()
        comp._apply_scene_client.send_request.return_value = SimpleNamespace(
            success=True
        )
        comp._scene_objects["det__mug_0"] = {
            "source": "detection",
            "last_seen": 0.0,
            "center": (0.4, 0.0, 0.05),
            "size": (0.06, 0.06, 0.1),
        }
        return comp

    @staticmethod
    def _pick_goal(**fields):
        from automatika_embodied_agents.action import MoveManipulator

        goal = MoveManipulator.Goal()
        goal.mode = "pick"
        for key, value in fields.items():
            setattr(goal, key, value)
        return goal

    @staticmethod
    def _states(handle):
        return [call[0][0].state for call in handle.publish_feedback.call_args_list]

    def test_pick_runs_the_whole_sequence(self, component):
        handle = TestGoalExecution._goal_handle(self._pick_goal(target_object="mug"))

        result = component.main_action_callback(handle)

        assert result.success and "det__mug_0" in result.message
        handle.succeed.assert_called_once()
        assert self._states(handle) == [
            "PRE_GRASP", "OPEN_GRIPPER", "DESCEND", "GRASP", "ATTACH", "LIFT",
        ]
        # approach and lift both go to the clearance height above the box top:
        # center z 0.05 + half height 0.05 + clearance 0.1
        move_goals = [
            call[0][0]
            for call in component._move_client.send_request.call_args_list
        ]
        heights = [
            g.request.goal_constraints[0]
            .position_constraints[0].constraint_region.primitive_poses[0].position.z
            for g in move_goals
        ]
        assert heights == [pytest.approx(0.2), pytest.approx(0.2)]
        # the descent is the one contact-permitted segment, down to the center
        request = component._cartesian_client.send_request.call_args[0][0]
        assert request.avoid_collisions is False
        assert request.waypoints[0].position.z == pytest.approx(0.05)
        # the object now travels with the arm
        assert component._scene_objects["det__mug_0"]["attached"] == "hand"

    def test_pick_resolves_labels_to_the_best_rank(self, component):
        component._scene_objects["det__mug_1"] = {
            "source": "detection", "last_seen": 0.0,
            "center": (0.9, 0.0, 0.05), "size": (0.06, 0.06, 0.1),
        }
        object_id, center, _ = component._resolve_pick_target(
            self._pick_goal(target_object="mug")
        )
        assert object_id == "det__mug_0" and center == (0.4, 0.0, 0.05)

    def test_pick_resolves_by_nearest_pose(self, component):
        goal = self._pick_goal()
        goal.target_pose.header.frame_id = "base"
        goal.target_pose.pose.position.x = 0.45
        object_id, _, _ = component._resolve_pick_target(goal)
        assert object_id == "det__mug_0"

    def test_a_far_pose_matches_nothing(self, component):
        goal = self._pick_goal()
        goal.target_pose.pose.position.x = 2.0
        with pytest.raises(ValueError, match="No scene object within"):
            component._resolve_pick_target(goal)

    def test_a_failed_resolution_aborts_the_goal_without_crashing(self, component):
        handle = TestGoalExecution._goal_handle(
            self._pick_goal(target_object="bottle")
        )

        result = component.main_action_callback(handle)

        assert not result.success and "det__mug_0" in result.message
        handle.abort.assert_called_once()
        component._move_client.send_request.assert_not_called()

    def test_the_match_radius_comes_from_config(self, component):
        component.config.target_match_radius = 0.01
        goal = self._pick_goal()
        goal.target_pose.pose.position.x = 0.45
        with pytest.raises(ValueError, match="within 0.01 m"):
            component._resolve_pick_target(goal)

    def test_an_attached_object_is_not_a_candidate(self, component):
        component._scene_objects["det__mug_0"]["attached"] = "hand"
        with pytest.raises(ValueError, match="No scene object matches"):
            component._resolve_pick_target(self._pick_goal(target_object="mug"))

    def test_an_unknown_object_lists_the_scene(self, component):
        with pytest.raises(ValueError, match="det__mug_0"):
            component._resolve_pick_target(self._pick_goal(target_object="bottle"))

    def test_a_failed_approach_aborts_with_the_step_named(self, component):
        from moveit_msgs.msg import MoveItErrorCodes

        component._move_client = TestGoalExecution._client(
            error_code=MoveItErrorCodes.PLANNING_FAILED
        )
        handle = TestGoalExecution._goal_handle(self._pick_goal(target_object="mug"))

        result = component.main_action_callback(handle)

        assert not result.success
        assert result.message.startswith("Pre-grasp:")
        handle.abort.assert_called_once()
        # the sequence stopped before touching anything
        assert "attached" not in component._scene_objects["det__mug_0"]

    def test_a_partial_descent_aborts(self, component):
        component._cartesian_client.send_request.return_value = SimpleNamespace(
            fraction=0.4, solution=_trajectory(), error_code=SimpleNamespace(val=1)
        )
        handle = TestGoalExecution._goal_handle(self._pick_goal(target_object="mug"))

        result = component.main_action_callback(handle)

        assert not result.success and result.message.startswith("Descent:")

    def test_cancellation_cancels_the_goal(self, component):
        handle = TestGoalExecution._goal_handle(
            self._pick_goal(target_object="mug"), cancel_requested=True
        )
        component._move_client.action_returned = False

        component.main_action_callback(handle)

        handle.canceled.assert_called_once()


class TestPlaceSequence:
    """The place mode: approach, descend, release, detach, retreat."""

    @pytest.fixture(autouse=True)
    def _needs_moveit_msgs(self):
        pytest.importorskip("moveit_msgs.msg")

    @pytest.fixture
    def component(self, rclpy_init):
        from agents.components import MoveIt
        from agents.config import MoveItConfig

        comp = MoveIt(
            config=MoveItConfig(
                arm_group_name="arm",
                gripper_group_name="gripper",
                end_effector_link="hand",
                pose_reference_frame="base",
                approach_clearance=0.1,
                touch_links=["hand", "jaw"],
            ),
            component_name="m_place",
        )
        comp.get_logger = MagicMock()
        comp.health_status = MagicMock()
        comp._named_targets = {
            "gripper": {"open": {"jaw": 1.0}, "close": {"jaw": 0.0}}
        }
        comp._move_client = TestGoalExecution._client()
        comp._exec_client = TestGoalExecution._client()
        comp._gripper_client = TestGoalExecution._client()
        comp._cartesian_client = MagicMock()
        comp._cartesian_client.send_request.return_value = SimpleNamespace(
            fraction=1.0, solution=_trajectory(), error_code=SimpleNamespace(val=1)
        )
        comp._apply_scene_client = MagicMock()
        comp._apply_scene_client.send_request.return_value = SimpleNamespace(
            success=True
        )
        comp._scene_objects["det__mug_0"] = {
            "source": "detection",
            "last_seen": 0.0,
            "center": (0.4, 0.0, 0.05),
            "size": (0.06, 0.06, 0.1),
            "attached": "hand",
        }
        return comp

    @staticmethod
    def _place_goal(x=0.6, y=0.1, z=0.08):
        from automatika_embodied_agents.action import MoveManipulator

        goal = MoveManipulator.Goal()
        goal.mode = "place"
        position = goal.target_pose.pose.position
        position.x, position.y, position.z = x, y, z
        return goal

    def test_place_runs_the_whole_sequence(self, component):
        handle = TestGoalExecution._goal_handle(self._place_goal())

        result = component.main_action_callback(handle)

        assert result.success and "det__mug_0" in result.message
        handle.succeed.assert_called_once()
        assert TestPickSequence._states(handle) == [
            "PRE_PLACE", "DESCEND", "RELEASE", "DETACH", "RETREAT",
        ]
        # approach and retreat above the release point: z 0.08 + half
        # height 0.05 + clearance 0.1
        heights = [
            call[0][0].request.goal_constraints[0]
            .position_constraints[0].constraint_region.primitive_poses[0].position.z
            for call in component._move_client.send_request.call_args_list
        ]
        assert heights == [pytest.approx(0.23), pytest.approx(0.23)]
        # the descent is contact permitted, down to the release center
        request = component._cartesian_client.send_request.call_args[0][0]
        assert request.avoid_collisions is False
        assert request.waypoints[0].position.z == pytest.approx(0.08)
        # released: no longer attached, and the scene records the new spot
        entry = component._scene_objects["det__mug_0"]
        assert "attached" not in entry
        assert entry["center"] == (0.6, 0.1, 0.08)

    def test_place_with_nothing_attached_aborts(self, component):
        del component._scene_objects["det__mug_0"]["attached"]
        handle = TestGoalExecution._goal_handle(self._place_goal())

        result = component.main_action_callback(handle)

        assert not result.success and "Nothing is attached" in result.message
        handle.abort.assert_called_once()
        component._move_client.send_request.assert_not_called()

    def test_a_failed_approach_keeps_the_object_attached(self, component):
        from moveit_msgs.msg import MoveItErrorCodes

        component._move_client = TestGoalExecution._client(
            error_code=MoveItErrorCodes.PLANNING_FAILED
        )
        handle = TestGoalExecution._goal_handle(self._place_goal())

        result = component.main_action_callback(handle)

        assert not result.success
        assert result.message.startswith("Pre-place:")
        entry = component._scene_objects["det__mug_0"]
        assert entry["attached"] == "hand" and entry["center"] == (0.4, 0.0, 0.05)

    def test_a_partial_descent_aborts_before_release(self, component):
        component._cartesian_client.send_request.return_value = SimpleNamespace(
            fraction=0.3, solution=_trajectory(), error_code=SimpleNamespace(val=1)
        )
        handle = TestGoalExecution._goal_handle(self._place_goal())

        result = component.main_action_callback(handle)

        assert not result.success and result.message.startswith("Descent:")
        assert component._scene_objects["det__mug_0"]["attached"] == "hand"

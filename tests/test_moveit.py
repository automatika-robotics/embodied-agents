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
        assert MoveMode.values() == ["pose", "joints", "named", "cartesian"]

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

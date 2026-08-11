"""Tests for MoveIt request-building utilities — no ROS node needed."""

from types import SimpleNamespace

import pytest

from agents.utils.moveit import (
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

"""Utilities for building MoveIt 2 requests from plain targets.

MoveIt's move_group node accepts motion targets only as constraint sets
(``moveit_msgs/Constraints``). These helpers do the same message
construction that MoveIt's C++ MoveGroupInterface performs client side, so
no MoveIt python bindings are required. MoveIt message definitions are imported
lazily.
"""

import xml.etree.ElementTree as ET
from importlib.util import find_spec
from typing import Any, Dict, Iterable, List, Sequence, Union

_MOVEIT_INSTALL_HINT = (
    "'moveit_msgs' module is required to use the MoveIt component but it is "
    "not installed. Please install the 'ros-<distro>-moveit-msgs' package."
)


def ensure_moveit_msgs() -> None:
    """Raise an actionable error when moveit_msgs is unavailable."""
    if find_spec("moveit_msgs") is None:
        raise ModuleNotFoundError(_MOVEIT_INSTALL_HINT)


def pose_to_constraints(
    target_pose: Any,
    link_name: str,
    position_tolerance: float,
    orientation_tolerance: Union[float, Sequence[float]],
) -> Any:
    """Express an end-effector pose target as a MoveIt constraint set.

    The position target becomes a spherical constraint region of radius
    ``position_tolerance`` centered at the target point; the orientation
    target is constrained per axis by ``orientation_tolerance``.

    :param target_pose: geometry_msgs PoseStamped target
    :param link_name: End-effector link the constraints apply to
    :param position_tolerance: Position tolerance in meters
    :param orientation_tolerance: Orientation tolerance in radians — a single
        value applied to all axes, or a sequence of 3 per-axis values
        (x, y, z). Relaxing a single axis (e.g. z to ~pi) makes many poses
        reachable for underactuated (e.g. 5-DOF) arms and rotationally
        symmetric tools
    :returns: moveit_msgs Constraints
    """
    if isinstance(orientation_tolerance, (int, float)):
        axis_tolerances = (float(orientation_tolerance),) * 3
    else:
        axis_tolerances = tuple(float(value) for value in orientation_tolerance)
        if len(axis_tolerances) != 3:
            raise ValueError(
                "orientation_tolerance must be a single value or a sequence "
                f"of 3 per-axis (x, y, z) values, got {len(axis_tolerances)}"
            )
    ensure_moveit_msgs()
    from moveit_msgs.msg import (
        BoundingVolume,
        Constraints,
        OrientationConstraint,
        PositionConstraint,
    )
    from shape_msgs.msg import SolidPrimitive

    constraints = Constraints()

    position = PositionConstraint()
    position.header = target_pose.header
    position.link_name = link_name
    region = BoundingVolume()
    region.primitives.append(
        SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[position_tolerance])
    )
    region.primitive_poses.append(target_pose.pose)
    position.constraint_region = region
    position.weight = 1.0  # hard constraint
    constraints.position_constraints.append(position)

    orientation = OrientationConstraint()
    orientation.header = target_pose.header
    orientation.link_name = link_name
    orientation.orientation = target_pose.pose.orientation
    orientation.absolute_x_axis_tolerance = axis_tolerances[0]
    orientation.absolute_y_axis_tolerance = axis_tolerances[1]
    orientation.absolute_z_axis_tolerance = axis_tolerances[2]
    orientation.weight = 1.0  # hard constraint
    constraints.orientation_constraints.append(orientation)

    return constraints


def joints_to_constraints(joint_positions: Dict[str, float], tolerance: float) -> Any:
    """Express a joint-space target as a MoveIt constraint set.

    :param joint_positions: Mapping of joint name to target position
    :param tolerance: Position tolerance applied above and below the target
    :returns: moveit_msgs Constraints
    """
    ensure_moveit_msgs()
    from moveit_msgs.msg import Constraints, JointConstraint

    constraints = Constraints()
    for name, position in joint_positions.items():
        constraints.joint_constraints.append(
            JointConstraint(
                joint_name=name,
                position=float(position),
                tolerance_above=tolerance,
                tolerance_below=tolerance,
                weight=1.0,  # hard constraint
            )
        )
    return constraints


def build_move_group_goal(
    constraints: Any,
    group_name: str,
    *,
    pipeline_id: str = "",
    planner_id: str = "",
    num_attempts: int = 5,
    allowed_time: float = 5.0,
    velocity_scaling: float = 0.1,
    acceleration_scaling: float = 0.1,
    plan_only: bool = False,
) -> Any:
    """Build a MoveGroup action goal around a constraint set.

    The start state is marked as a diff so move_group plans from the robot
    state it is currently monitoring.

    :param constraints: moveit_msgs Constraints (from pose_to_constraints or
        joints_to_constraints)
    :param group_name: SRDF planning group to plan for
    :returns: moveit_msgs MoveGroup.Goal
    """
    ensure_moveit_msgs()
    from moveit_msgs.action import MoveGroup

    goal = MoveGroup.Goal()
    request = goal.request
    request.group_name = group_name
    request.goal_constraints = [constraints]
    request.pipeline_id = pipeline_id
    request.planner_id = planner_id
    request.num_planning_attempts = num_attempts
    request.allowed_planning_time = allowed_time
    request.max_velocity_scaling_factor = velocity_scaling
    request.max_acceleration_scaling_factor = acceleration_scaling
    request.start_state.is_diff = True
    goal.planning_options.plan_only = plan_only
    return goal


def build_cartesian_request(
    waypoints: Iterable[Any],
    frame_id: str,
    group_name: str,
    link_name: str,
    *,
    max_step: float = 0.0025,
    jump_threshold: float = 0.0,
    avoid_collisions: bool = True,
    velocity_scaling: float = 0.1,
    acceleration_scaling: float = 0.1,
) -> Any:
    """Build a GetCartesianPath service request for a straight-line path.

    :param waypoints: geometry_msgs Pose waypoints for the end effector
    :param frame_id: Reference frame of the waypoints
    :param group_name: SRDF planning group
    :param link_name: End-effector link that follows the waypoints
    :returns: moveit_msgs GetCartesianPath.Request
    """
    ensure_moveit_msgs()
    from moveit_msgs.srv import GetCartesianPath

    request = GetCartesianPath.Request()
    request.header.frame_id = frame_id
    request.group_name = group_name
    request.link_name = link_name
    request.waypoints = list(waypoints)
    request.max_step = max_step
    request.avoid_collisions = avoid_collisions
    request.start_state.is_diff = True
    # NOTE: These fields vary across MoveIt versions (humble..rolling), set only
    # when present
    for field_name, value in (
        ("jump_threshold", jump_threshold),
        ("max_velocity_scaling_factor", velocity_scaling),
        ("max_acceleration_scaling_factor", acceleration_scaling),
    ):
        if hasattr(request, field_name):
            setattr(request, field_name, value)
    return request


def parse_srdf_group_states(srdf_xml: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Extract named states per planning group from an SRDF document.

    :param srdf_xml: SRDF document content
    :returns: Mapping of group name to {state name: {joint name: position}}
    """
    root = ET.fromstring(srdf_xml)
    states: Dict[str, Dict[str, Dict[str, float]]] = {}
    for group_state in root.iter("group_state"):
        group = group_state.get("group")
        name = group_state.get("name")
        if not group or not name:
            continue
        joints = {}
        for joint in group_state.iter("joint"):
            joint_name = joint.get("name")
            value = joint.get("value")
            if joint_name and value is not None:
                # multi-dof joint values are space separated; keep the first
                joints[joint_name] = float(value.split()[0])
        states.setdefault(group, {})[name] = joints
    return states


def parse_planner_interfaces(response: Any) -> Dict[str, List[str]]:
    """Map available planning pipelines to their planner ids.

    :param response: moveit_msgs QueryPlannerInterfaces.Response
    :returns: Mapping of pipeline id to the planner ids it provides
    """
    pipelines: Dict[str, List[str]] = {}
    for interface in response.planner_interfaces:
        pipelines.setdefault(interface.pipeline_id, []).extend(interface.planner_ids)
    return pipelines


def moveit_error_string(code: int) -> str:
    """Translate a raw MoveItErrorCodes value to its constant name.

    Built dynamically from the installed message definition.

    :param code: MoveItErrorCodes value
    :returns: Constant name, e.g. 'SUCCESS' or 'PLANNING_FAILED'
    """
    ensure_moveit_msgs()
    from moveit_msgs.msg import MoveItErrorCodes

    names = {
        value: name
        for name, value in vars(MoveItErrorCodes).items()
        if name.isupper() and isinstance(value, int)
    }
    return names.get(code, f"UNKNOWN_ERROR({code})")

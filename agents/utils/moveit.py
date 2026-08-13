"""Utilities for building MoveIt 2 requests from plain targets.

MoveIt's move_group node accepts motion targets only as constraint sets
(``moveit_msgs/Constraints``). These helpers do the same message
construction that MoveIt's C++ MoveGroupInterface performs client side, so
no MoveIt python bindings are required. MoveIt message definitions are imported
lazily.
"""

import xml.etree.ElementTree as ET
from enum import Enum
from importlib.util import find_spec
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from rclpy.logging import get_logger

from .utils import _read_spec_file

_MOVEIT_INSTALL_HINT = (
    "'moveit_msgs' module is required to use the MoveIt component but it is "
    "not installed. Please install the 'ros-<distro>-moveit-msgs' package."
)

# Parameter of the move_group node that carries the robot SRDF
SRDF_PARAMETER = "robot_description_semantic"

# =========================================================================
# General Utils
# =========================================================================


class MoveMode(str, Enum):
    """Kinds of motion target accepted in a manipulation goal.

    - ``POSE``: an end-effector pose.
    - ``JOINTS``: explicit joint positions.
    - ``NAMED``: a named target defined in the robot's SRDF (e.g. "home").
    - ``CARTESIAN``: a straight-line end-effector path through waypoints.
    """

    POSE = "pose"
    JOINTS = "joints"
    NAMED = "named"
    CARTESIAN = "cartesian"

    @classmethod
    def values(cls) -> List[str]:
        """List the valid mode strings"""
        return [mode.value for mode in cls]

    @classmethod
    def from_goal(cls, goal: Any) -> "MoveMode":
        """Get the mode of a manipulation goal, inferring it when not set.

        When the goal's ``mode`` field is empty the mode is inferred from
        whichever target field carries a value.

        :param goal: MoveManipulator goal
        :raises ValueError: If the mode is unknown or no target is given
        """
        mode = (goal.mode or "").strip().lower()
        if mode:
            try:
                return cls(mode)
            except ValueError:
                raise ValueError(
                    f"Unknown mode '{mode}'. Valid modes: {cls.values()}"
                ) from None
        if goal.named_target:
            return cls.NAMED
        if len(goal.cartesian_waypoints):
            return cls.CARTESIAN
        if len(goal.joint_names):
            return cls.JOINTS
        if goal.target_pose.header.frame_id or any([
            goal.target_pose.pose.position.x,
            goal.target_pose.pose.position.y,
            goal.target_pose.pose.position.z,
        ]):
            return cls.POSE
        raise ValueError(
            "No motion target given. Set 'mode' and the matching target field: "
            f"{cls.values()}"
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

    from ..ros import SolidPrimitive

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


def load_named_targets(
    srdf: Optional[Union[str, ET.Element]] = None,
    srdf_file: Optional[str] = None,
    overrides: Optional[Dict] = None,
    logger_name: str = "moveit",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Collect the named targets available per planning group.

    Named targets are read from the robot SRDF, either from already
    retrieved content or from a file when no content is given. Manually
    provided targets take precedence over the ones defined in the SRDF.

    :param srdf: SRDF document content, e.g. as read from the move_group node
    :param srdf_file: Path or URL of an SRDF file, used when no content is given
    :param overrides: Manually defined targets, {group: {name: {joint: position}}}
    :param logger_name: Name of the logger to report failures on
    :returns: Mapping of group name to {state name: {joint name: position}}
    """
    states: Dict[str, Dict[str, Dict[str, float]]] = {}
    if srdf is None and srdf_file:
        try:
            srdf = _read_spec_file(srdf_file, spec_type="xml")
        except Exception as e:
            get_logger(logger_name).warning(
                f"Could not read SRDF file '{srdf_file}': {e}"
            )
    if srdf is not None:
        try:
            states = parse_srdf_group_states(srdf)
        except Exception as e:
            get_logger(logger_name).warning(f"Could not parse the robot SRDF: {e}")

    for group, group_states in (overrides or {}).items():
        states.setdefault(group, {}).update(group_states)
    return states


def parse_srdf_group_states(
    srdf_xml: Union[str, ET.Element],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Extract named states per planning group from an SRDF document.

    :param srdf_xml: SRDF document content, or an already parsed XML root
        (as returned by ``_read_spec_file`` for local files)
    :returns: Mapping of group name to {state name: {joint name: position}}
    """
    root = ET.fromstring(srdf_xml) if isinstance(srdf_xml, str) else srdf_xml
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


# =========================================================================
# Planning scene
# =========================================================================

# Prefix namespacing collision objects that came from detections, so clearing
# them leaves manually added objects alone
DETECTION_ID_PREFIX = "det__"


def object_id_for(label: str, rank: int, prefix: str = DETECTION_ID_PREFIX) -> str:
    """Scene id of a detected object, e.g. ``det__orange_0``.

    :param label: Detection label
    :param rank: Position among same-labelled objects, by descending score
    :param prefix: Namespace marking the object as detection-sourced
    """
    clean = (label or "object").strip().replace(" ", "_")
    return f"{prefix}{clean}_{rank}"


def build_collision_object(
    object_id: str,
    frame_id: str,
    center: Any,
    size: Sequence[float],
    min_thickness: float = 0.01,
) -> Any:
    """A box-shaped collision object MoveIt can plan around.

    The object is expressed directly in ``frame_id`` with a zero timestamp,
    which move_group takes as-is without a TF lookup, so objects should be built
    objects in the planning frame. Re-adding an existing id moves the object.

    :param object_id: Scene-wide identifier
    :param frame_id: Frame the center is given in, normally the planning frame
    :param center: geometry_msgs Pose, or an (x, y, z) position for an
        axis-aligned box
    :param size: Full extents of the box in meters
    :param min_thickness: Floor for each extent. A fronto-parallel surface is
        lifted with no measurable extent along the view axis, and a zero
        thickness box is invisible to collision checking
    :returns: moveit_msgs CollisionObject with operation ADD
    """
    ensure_moveit_msgs()
    from moveit_msgs.msg import CollisionObject

    from ..ros import ROSPose, SolidPrimitive

    pose = ROSPose()
    if hasattr(center, "position"):
        pose.position.x = center.position.x
        pose.position.y = center.position.y
        pose.position.z = center.position.z
        source = center.orientation
        if any((source.x, source.y, source.z, source.w)):
            pose.orientation.x = source.x
            pose.orientation.y = source.y
            pose.orientation.z = source.z
            pose.orientation.w = source.w
        else:
            # All zeros is not a rotation and MoveIt rejects it
            pose.orientation.w = 1.0
    else:
        pose.position.x, pose.position.y, pose.position.z = (
            float(value) for value in center
        )
        # A bare Pose carries the same all-zero quaternion
        pose.orientation.w = 1.0

    obj = CollisionObject()
    obj.id = object_id
    obj.header.frame_id = frame_id  # stamp stays zero to avoid TF lookup in move_group
    obj.primitives.append(
        SolidPrimitive(
            type=SolidPrimitive.BOX,
            dimensions=[max(float(extent), min_thickness) for extent in size],
        )
    )
    obj.primitive_poses.append(pose)
    obj.operation = CollisionObject.ADD  # an octet field: use the constants
    return obj


def build_remove_object(object_id: str) -> Any:
    """An instruction taking one object out of the planning scene.

    :param object_id: Id the object was added under
    :returns: moveit_msgs CollisionObject with operation REMOVE
    """
    ensure_moveit_msgs()
    from moveit_msgs.msg import CollisionObject

    obj = CollisionObject()
    obj.id = object_id
    obj.operation = CollisionObject.REMOVE
    return obj


def build_scene_diff(
    collision_objects: Sequence[Any] = (),
    attached_objects: Sequence[Any] = (),
) -> Any:
    """Wrap scene changes as one diff against move_group's current scene.

    A diff only touches what it carries. The robot state is marked as a diff too.
    Without that, applying the scene would overwrite the state move_group is monitoring.

    :param collision_objects: World objects to add, move or remove
    :param attached_objects: Objects to attach to or detach from the robot
    :returns: moveit_msgs PlanningScene diff
    """
    ensure_moveit_msgs()
    from moveit_msgs.msg import PlanningScene

    scene = PlanningScene()
    scene.is_diff = True
    scene.robot_state.is_diff = True
    scene.world.collision_objects.extend(collision_objects)
    scene.robot_state.attached_collision_objects.extend(attached_objects)
    return scene


def collision_objects_from_detections(
    detections: Any,
    *,
    min_thickness: float = 0.01,
    padding: float = 0.0,
    id_prefix: str = DETECTION_ID_PREFIX,
) -> List[Any]:
    """Turn a Detections3D message into collision objects.

    Ranks in IDs counted per label by descending score, so on a static scene
    the same object keeps the same id from one refresh to the next without needing
    a tracker.

    Boxes are used in the planning frame (already transformed).

    :param detections: Detections3D message
    :param min_thickness: Floor for each box extent in meters
    :param padding: Margin added on every side of every box in meters, for
        planning clearance around imperfectly measured objects
    :returns: moveit_msgs CollisionObjects with operation ADD
    """
    frame = detections.header.frame_id
    labels = list(detections.labels)
    scores = list(detections.scores)
    entries = []
    for index, box in enumerate(detections.boxes):
        entries.append((
            labels[index] if index < len(labels) else "",
            scores[index] if index < len(scores) else 0.0,
            box,
        ))

    objects = []
    ranks: Dict[str, int] = {}
    for label, _, box in sorted(entries, key=lambda entry: -entry[1]):
        rank = ranks.get(label, 0)
        ranks[label] = rank + 1
        objects.append(
            build_collision_object(
                object_id_for(label, rank, prefix=id_prefix),
                frame,
                box.center,
                (
                    box.size.x + 2 * padding,
                    box.size.y + 2 * padding,
                    box.size.z + 2 * padding,
                ),
                min_thickness=min_thickness,
            )
        )
    return objects


def build_attached_object(
    object_id: str, link_name: str, touch_links: Sequence[str] = ()
) -> Any:
    """Attach a scene object to a robot link, as on a grasp.

    NOTE: Attaching by id moves an object already in the scene: move_group removes
    it from the world and carries it with the link, excluding contact with
    the ``touch_links`` from collision checking. With the wrong touch links
    the first motion after a grasp fails on a collision between gripper and
    held object.

    :param object_id: Id of an object already in the planning scene
    :param link_name: Link the object is rigidly attached to
    :param touch_links: Links allowed to stay in contact with the object
    :returns: moveit_msgs AttachedCollisionObject
    """
    ensure_moveit_msgs()
    from moveit_msgs.msg import AttachedCollisionObject, CollisionObject

    attached = AttachedCollisionObject()
    attached.link_name = link_name
    attached.object.id = object_id
    attached.object.operation = CollisionObject.ADD
    attached.touch_links = list(touch_links)
    return attached


def build_detach_object(object_id: str, link_name: str = "") -> Any:
    """Detach an attached object, returning it to the world.

    :param object_id: Id the object was attached under
    :param link_name: Link holding the object; empty searches every link
    :returns: moveit_msgs AttachedCollisionObject with operation REMOVE
    """
    ensure_moveit_msgs()
    from moveit_msgs.msg import AttachedCollisionObject, CollisionObject

    attached = AttachedCollisionObject()
    attached.link_name = link_name
    attached.object.id = object_id
    attached.object.operation = CollisionObject.REMOVE
    return attached


def derive_touch_links(
    srdf: Optional[Union[str, ET.Element]],
    end_effector_link: str,
    gripper_group: str = "",
    explicit: Optional[Sequence[str]] = None,
    logger_name: str = "moveit",
) -> List[str]:
    """Resolve the links allowed to stay in contact with a grasped object.

    CAUTION: Once an object is attached it is in permanent contact with the gripper, and
    any gripper link missing from this list makes every subsequent motion start in
    collision.

    Resolution order:

    1. An explicitly configured list is used as given.
    2. ``<link>`` entries of the SRDF gripper group, the declarative source.
    3. The SRDF end effector's parent link plus its ``Adjacent``
       ``disable_collisions`` partners. This is a heuristic: it finds jaws
       attached directly to the parent link but misses links one joint
       further out, so a joint-only gripper group is better served by adding
       its links to the SRDF or to the config.
    4. The end effector link alone, with a warning.

    :param srdf: SRDF document content, or an already parsed XML root
    :param end_effector_link: Configured end-effector link, the last resort
    :param gripper_group: SRDF group of the gripper, used to pick its links
        and its end effector entry
    :param explicit: Configured touch links, trusted as given
    :param logger_name: Name of the logger to report the fallback on
    :returns: Link names, deduplicated, order preserved
    """
    if explicit:
        return list(dict.fromkeys(explicit))

    root = None
    if srdf is not None:
        try:
            root = ET.fromstring(srdf) if isinstance(srdf, str) else srdf
        except ET.ParseError as e:
            get_logger(logger_name).warning(f"Could not parse the robot SRDF: {e}")

    if root is not None:
        if gripper_group:
            links = [
                link.get("name")
                for group in root.iter("group")
                if group.get("name") == gripper_group
                for link in group.iter("link")
                if link.get("name")
            ]
            if links:
                return list(dict.fromkeys(links))

        parent = next(
            (
                eef.get("parent_link")
                for eef in root.iter("end_effector")
                if not gripper_group or eef.get("group") == gripper_group
            ),
            None,
        )
        if parent:
            # NOTE:Only pairs disabled for being physically connected say anything
            # about the gripper's shape; "Never" pairs span the whole robot
            partners = []
            for pair in root.iter("disable_collisions"):
                if (pair.get("reason") or "").lower() != "adjacent":
                    continue
                link1, link2 = pair.get("link1"), pair.get("link2")
                if parent == link1 and link2:
                    partners.append(link2)
                elif parent == link2 and link1:
                    partners.append(link1)
            if partners:
                return list(dict.fromkeys([parent, *partners]))

    get_logger(logger_name).warning(
        "Could not resolve the gripper's links from the SRDF, so only "
        f"'{end_effector_link}' may touch an attached object. If grasping "
        "fails on a collision between gripper and object, list the gripper "
        "links in the SRDF gripper group or in the component config."
    )
    return [end_effector_link]

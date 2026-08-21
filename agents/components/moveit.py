import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from ..config import MoveItConfig
from ..ros import (
    ActionClientConfig,
    ActionClientHandler,
    ActionPhase,
    ComponentRunType,
    Detections3D,
    Empty,
    GetParameters,
    MoveManipulator,
    ServiceClientConfig,
    ServiceClientHandler,
    component_action,
)
from ..utils import validate_func_args
from ..utils.moveit import (
    MoveMode,
    build_cartesian_request,
    build_move_group_goal,
    ensure_moveit_msgs,
    joints_to_constraints,
    load_named_targets,
    moveit_error_string,
    parse_planner_interfaces,
    pose_to_constraints,
    SRDF_PARAMETER,
)
from .component_base import Component


class MoveIt(Component):
    """
    This component provides classical, collision aware manipulation by driving a running [MoveIt 2](https://moveit.ai) `move_group` node.

    The component runs as a ROS2 Action Server exposing the `<component_name>/manipulate_with_moveit` action, which takes a motion target and plans and executes a path to it. Six kinds of targets are supported, selected with the goal's `mode` field (or inferred from the fields that are populated):

    - **pose**: an end-effector pose, planned with the configured planner and reached within the configured position/orientation tolerances.
    - **joints**: explicit joint positions.
    - **named**: a named target defined in the robot's SRDF (e.g. "home", "ready"), resolved to joint positions by reading the SRDF from the running `move_group` node.
    - **cartesian**: a straight-line end-effector path through a list of waypoints, computed with MoveIt's Cartesian path service and executed only if enough of the path is achievable (see `cartesian_fraction_threshold`). Waypoints without an orientation preference (identity) hold the current end-effector orientation, read from move_group's FK service — an arbitrary default may not be achievable, especially on underactuated arms.
    - **pick**: a full grasp sequence on a planning scene object, named with `target_object` (a scene id or detection label) or located by `target_pose`: approach above it, open the gripper, descend, close, attach the object and lift it.
    - **place**: the inverse sequence for the currently attached object: approach above `target_pose`, descend, open the gripper, detach (the object stays in the scene where released) and retreat.

    Gripper control is available as component actions (`open_gripper`, `close_gripper`, `set_gripper`), either through the gripper planning group or through a gripper controller, based on the `gripper_mode` config parameter. All component actions and the action server itself are discoverable as tools by the Cortex component, making the manipulator directly usable by LLM agents.

    This component requires a running `move_group` node, which is normally launched from the robot's MoveIt configuration package. It can be brought up alongside the components in the same recipe (see the example below), and the `moveit_msgs` package must be installed (`ros-<distro>-moveit-msgs`).

    :param config: The configuration for the MoveIt component. `arm_group_name` is required and must match a planning group in the robot's SRDF.
    :type config: MoveItConfig
    :param inputs: Optional input topics for the component, limited to Detections3D type: detected objects used to populate the planning scene (see `scene_update_mode` in the config), so that motions plan around what the robot's cameras see. Not required for manipulation.
    :type inputs: Optional[list[Topic]]
    :param outputs: Optional output topics for the component. Not required for manipulation.
    :type outputs: Optional[list[Topic]]
    :param component_name: The name of the MoveIt component.
    :type component_name: str
    :param kwargs: Additional keyword arguments.

    Example usage:
    ```python
    from agents.components import MoveIt
    from agents.config import MoveItConfig
    from agents.ros import Launcher

    config = MoveItConfig(
        arm_group_name="panda_arm",
        gripper_group_name="hand",
        max_velocity_scaling=0.2,
    )
    manipulator = MoveIt(config=config, component_name="manipulator")

    launcher = Launcher()
    # bring up the robot's MoveIt configuration alongside the component
    launcher.include_launch_file(
        package="moveit_resources_panda_moveit_config", launch_file="demo.launch.py"
    )
    launcher.add_pkg(components=[manipulator])
    launcher.bringup()
    ```

    A motion goal can then be sent to the running component, e.g. from the command line:
    ```shell
    ros2 action send_goal /manipulator/manipulate_with_moveit automatika_embodied_agents/action/MoveManipulator "{mode: named, named_target: ready}"
    ```

    And the gripper can be commanded with:
    ```shell
    ros2 service call /manipulator/execute_method automatika_ros_sugar/srv/ExecuteMethod "{name: open_gripper, kwargs_json: '{}'}"
    ```
    """

    @validate_func_args
    def __init__(
        self,
        *,
        config: MoveItConfig,
        component_name: str,
        inputs: Optional[List] = None,
        outputs: Optional[List] = None,
        **kwargs,
    ):
        # Fail fast with installation instructions
        ensure_moveit_msgs()

        self.config: MoveItConfig = config

        self.allowed_inputs = {"Required": [], "Optional": [Detections3D]}
        self._detections_topic = inputs[0] if inputs else None

        # Set the component to run as an action server
        self.run_type = ComponentRunType.ACTION_SERVER

        # TODO: Trigger will be passed by serialized component in the kwargs
        # We will remove it here. This can be improved by passing kwargs differently
        if "trigger" in list(kwargs.keys()):
            kwargs.pop("trigger")

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            config=self.config,
            trigger=None,
            component_name=component_name,
            main_action_type=MoveManipulator,
            **kwargs,
        )

        # set a meaningful component action name
        self.main_action_name = f"{self.node_name}/manipulate_with_moveit"

        # Clients to move_group, created on activation
        self._move_client: Optional[ActionClientHandler] = None
        self._exec_client: Optional[ActionClientHandler] = None
        self._gripper_client: Optional[ActionClientHandler] = None
        self._cartesian_client: Optional[ServiceClientHandler] = None
        self._param_client: Optional[ServiceClientHandler] = None
        self._planner_client: Optional[ServiceClientHandler] = None
        # Planning scene clients
        self._apply_scene_client: Optional[ServiceClientHandler] = None
        self._get_scene_client: Optional[ServiceClientHandler] = None
        self._clear_octomap_client: Optional[ServiceClientHandler] = None
        self._fk_client: Optional[ServiceClientHandler] = None

        # Named targets read from the SRDF, resolved lazily and cached, and
        # the raw SRDF they were read from
        self._named_targets: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None
        self._srdf: Optional[str] = None

        # Objects this component has placed in the planning scene, as
        # id -> {"source": "manual" | "detection", "last_seen": time}
        self._scene_objects: Dict[str, Dict[str, Any]] = {}
        self._scene_lock = threading.Lock()
        # Id of a scene object the gripper is currently allowed to touch
        self._contact_freeze: Optional[str] = None

        # Continuous mode refresh timer and message last applied
        self._scene_timer = None
        self._last_scene_message: Optional[Any] = None

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def create_all_action_clients(self):
        """Create action clients to the move_group node"""
        super().create_all_action_clients()

        from control_msgs.action import GripperCommand
        from moveit_msgs.action import ExecuteTrajectory, MoveGroup

        # NOTE: The client handler cancels a goal when it receives no new feedback
        # within feedback_check_timeout. move_group only publishes feedback on
        # state transitions (planning -> executing), so this is set to the
        # execution timeout to avoid cancelling long but healthy motions
        self._move_client = ActionClientHandler(
            self, config=self._action_client_config(MoveGroup, "move_action")
        )
        self._exec_client = ActionClientHandler(
            self,
            config=self._action_client_config(ExecuteTrajectory, "execute_trajectory"),
        )
        if self.config.gripper_mode == "gripper_command":
            self._gripper_client = ActionClientHandler(
                self,
                config=ActionClientConfig(
                    action_type=GripperCommand,
                    name=self.config.gripper_command_action,
                    timeout_secs=self.config.server_timeout,
                    feedback_check_timeout=self.config.execution_timeout,
                ),
            )
        else:
            # A separate handler on the same action, so that gripper motions
            # do not disturb the state of an in-flight arm goal
            self._gripper_client = ActionClientHandler(
                self, config=self._action_client_config(MoveGroup, "move_action")
            )

    def destroy_all_action_clients(self):
        """Destroy action clients to the move_group node"""
        super().destroy_all_action_clients()

        for handler in (self._move_client, self._exec_client, self._gripper_client):
            if handler:
                handler.client.destroy()
        self._move_client = self._exec_client = self._gripper_client = None

    def create_all_service_clients(self):
        """Create service clients to the move_group node"""
        super().create_all_service_clients()

        from moveit_msgs.srv import GetCartesianPath, QueryPlannerInterfaces

        self._cartesian_client = ServiceClientHandler(
            self,
            config=ServiceClientConfig(
                srv_type=GetCartesianPath,
                name=self._resolve_name("compute_cartesian_path"),
                timeout_secs=self.config.server_timeout,
            ),
        )
        self._planner_client = ServiceClientHandler(
            self,
            config=ServiceClientConfig(
                srv_type=QueryPlannerInterfaces,
                name=self._resolve_name("query_planner_interface"),
                timeout_secs=self.config.server_timeout,
            ),
        )
        # Named targets live in the SRDF, which is a parameter of the move_group
        # node
        self._param_client = ServiceClientHandler(
            self,
            config=ServiceClientConfig(
                srv_type=GetParameters,
                name=self._resolve_name(
                    f"{self.config.move_group_node_name}/get_parameters"
                ),
                timeout_secs=self.config.server_timeout,
            ),
        )

        # Planning scene services, only used from the action and component-action
        # threads
        from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene

        self._apply_scene_client = ServiceClientHandler(
            self,
            config=ServiceClientConfig(
                srv_type=ApplyPlanningScene,
                name=self._resolve_name("apply_planning_scene"),
                timeout_secs=self.config.server_timeout,
            ),
        )
        self._get_scene_client = ServiceClientHandler(
            self,
            config=ServiceClientConfig(
                srv_type=GetPlanningScene,
                name=self._resolve_name("get_planning_scene"),
                timeout_secs=self.config.server_timeout,
            ),
        )
        self._clear_octomap_client = ServiceClientHandler(
            self,
            config=ServiceClientConfig(
                srv_type=Empty,
                name=self._resolve_name("clear_octomap"),
                timeout_secs=self.config.server_timeout,
            ),
        )

        from moveit_msgs.srv import GetPositionFK

        # Current end-effector pose, used to give Cartesian motions an
        # achievable default orientation
        self._fk_client = ServiceClientHandler(
            self,
            config=ServiceClientConfig(
                srv_type=GetPositionFK,
                name=self._resolve_name("compute_fk"),
                timeout_secs=self.config.server_timeout,
            ),
        )

    def destroy_all_service_clients(self):
        """Destroy service clients to the move_group node"""
        super().destroy_all_service_clients()

        for handler in (
            self._cartesian_client,
            self._planner_client,
            self._param_client,
            self._apply_scene_client,
            self._get_scene_client,
            self._clear_octomap_client,
            self._fk_client,
        ):
            if handler:
                self.destroy_client(handler.client)
        self._cartesian_client = self._planner_client = self._param_client = None
        self._apply_scene_client = self._get_scene_client = None
        self._clear_octomap_client = self._fk_client = None

    def custom_on_activate(self):
        """Activate component and check the configured planner against move_group"""
        super().custom_on_activate()
        self._validate_planner_config()

        if self.config.scene_update_mode != "manual" and not self._detections_topic:
            self.get_logger().warning(
                f"scene_update_mode is '{self.config.scene_update_mode}' but the "
                "component has no Detections3D input topic to update the "
                "planning scene from"
            )
        elif self.config.scene_update_mode == "continuous":
            from ..ros import MutuallyExclusiveCallbackGroup

            # Create a timer for updating the scene
            self._scene_timer = self.create_timer(
                1.0 / self.config.scene_update_rate,
                self._scene_refresh_tick,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )

    def custom_on_deactivate(self):
        """Stop the scene refresh before the clients it uses are destroyed"""
        if self._scene_timer:
            self.destroy_timer(self._scene_timer)
            self._scene_timer = None
        super().custom_on_deactivate()

    def _execution_step(self, *_, **__):
        """NOT USED. The component is triggered through its action server"""
        pass

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _resolve_name(self, name: str) -> str:
        """Get the fully qualified name of a move_group interface

        :param name: Interface name, e.g. "move_action"
        :returns: Absolute interface name, e.g. "/robot_a/move_action"
        """
        namespace = (self.config.move_group_namespace or "").strip().strip("/")
        return f"/{namespace}/{name}" if namespace else f"/{name}"

    def _action_client_config(self, action_type: type, name: str) -> ActionClientConfig:
        """Client config for an action provided by move_group"""
        return ActionClientConfig(
            action_type=action_type,
            name=self._resolve_name(name),
            timeout_secs=self.config.server_timeout,
            feedback_check_timeout=self.config.execution_timeout,
        )

    def _validate_planner_config(self) -> None:
        """Warn if the configured pipeline/planner is not provided by move_group"""
        if not self.config.planning_pipeline and not self.config.planner_id:
            return
        if not self._planner_client:
            return
        from moveit_msgs.srv import QueryPlannerInterfaces

        # get all planning pipeline and planners
        response = self._planner_client.send_request(QueryPlannerInterfaces.Request())
        if not response:
            self.get_logger().warning(
                "Could not query the available planners from move_group, skipping "
                "validation of 'planning_pipeline' and 'planner_id'"
            )
            return
        # verify pipeline
        pipelines = parse_planner_interfaces(response)
        pipeline = self.config.planning_pipeline
        if pipeline and pipeline not in pipelines:
            self.get_logger().warning(
                f"Configured planning_pipeline '{pipeline}' is not provided by "
                f"move_group. Available pipelines: {sorted(pipelines)}"
            )
            return
        # verify planner in pipeline
        if self.config.planner_id:
            available = (
                pipelines.get(pipeline)
                if pipeline
                else [p for ids in pipelines.values() for p in ids]
            )
            if available and self.config.planner_id not in available:
                self.get_logger().warning(
                    f"Configured planner_id '{self.config.planner_id}' is not "
                    f"provided by move_group. Available planners: {sorted(available)}"
                )

    def _named_target_states(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get named targets per planning group, reading the SRDF once

        The SRDF is read from the running move_group node, falling back to
        the configured SRDF file when the node cannot provide it.
        """
        if self._named_targets is None:
            srdf = None
            if self._param_client:
                response = self._param_client.send_request_from_dict({
                    "names": [SRDF_PARAMETER]
                })
                if response and response.values and response.values[0].string_value:
                    srdf = response.values[0].string_value
                    # Keep raw for consumers that read more than named targets
                    self._srdf = srdf
                else:
                    self.get_logger().warning(
                        f"Could not read '{SRDF_PARAMETER}' from "
                        f"'{self._param_client.config.name}'"
                    )
            self._named_targets = load_named_targets(
                srdf=srdf,
                srdf_file=self.config.srdf_file,
                overrides=self.config.named_targets,
                logger_name=self.node_name,
            )
        return self._named_targets

    def _resolve_named_target(self, name: str) -> Tuple[str, Dict[str, float]]:
        """Resolve a named target to a planning group and joint positions

        :param name: Named target, as defined in the robot SRDF
        :raises ValueError: If the target is not defined for any known group
        """
        states = self._named_target_states()
        # Prefer the arm group, then the gripper group, then any other group
        groups = [self.config.arm_group_name]
        if self.config.gripper_group_name:
            groups.append(self.config.gripper_group_name)
        groups += [group for group in states if group not in groups]

        for group in groups:
            if name in states.get(group, {}):
                return group, states[group][name]

        available = {
            group: sorted(group_states) for group, group_states in states.items()
        }
        raise ValueError(
            f"Named target '{name}' is not defined in the robot SRDF. "
            f"Available named targets per group: {available}"
        )

    @staticmethod
    def _orientation_given(orientation: Optional[Any]) -> bool:
        """Whether a quaternion expresses an actual orientation preference.

        ROS2 defaults an untouched Quaternion field to identity (w=1), so an
        identity on the wire cannot be distinguished from "no preference"
        and is read as such. Any quaternion with a rotation axis (a non-zero
        x, y or z) counts as explicit.
        """
        return orientation is not None and any((
            orientation.x,
            orientation.y,
            orientation.z,
        ))

    def _current_ee_orientation(self) -> Optional[Any]:
        """Orientation of the end effector, read from move_group's FK service.

        Used as the default orientation of Cartesian motions. The Cartesian
        interpolator validates the achieved orientation of every step, and
        holding the current orientation is achievable wherever the arm can
        move at all, where an arbitrary default (identity) may not be —
        especially on underactuated arms.

        :returns: geometry_msgs Quaternion, or None when FK is unavailable
        """
        if not self._fk_client or not self.config.end_effector_link:
            return None
        from moveit_msgs.srv import GetPositionFK

        request = GetPositionFK.Request()
        request.header.frame_id = self.config.pose_reference_frame
        request.fk_link_names = [self.config.end_effector_link]
        # an empty diff state means "the current monitored state"
        request.robot_state.is_diff = True
        response = self._fk_client.send_request(request)
        if (
            response is None
            or getattr(response.error_code, "val", 0) != 1
            or not response.pose_stamped
        ):
            self.get_logger().warning(
                "Could not read the current end-effector pose from "
                f"'{self._resolve_name('compute_fk')}'"
            )
            return None
        return response.pose_stamped[0].pose.orientation

    def _orient_waypoints(self, waypoints: Any) -> List[Any]:
        """Give waypoints without an orientation preference an achievable one.

        Waypoints carrying no orientation (see `_orientation_given`) follow
        the current end-effector orientation. Waypoints with an explicit
        orientation pass through unchanged, as does everything when FK is
        unavailable (identity, the previous behavior).
        """
        from copy import deepcopy

        oriented, default = [], None
        for waypoint in waypoints:
            if self._orientation_given(waypoint.orientation):
                oriented.append(waypoint)
                continue
            if default is None:
                default = self._current_ee_orientation()
            if default is not None:
                waypoint = deepcopy(waypoint)
                waypoint.orientation = default
            oriented.append(waypoint)
        return oriented

    def _scalings(self, goal: Any) -> Tuple[float, float]:
        """Per-goal scaling overrides, falling back to the configured values"""
        velocity = goal.velocity_scaling or self.config.max_velocity_scaling
        acceleration = goal.acceleration_scaling or self.config.max_acceleration_scaling
        return float(velocity), float(acceleration)

    def _publish_state(self, goal_handle, state: str) -> None:
        """Report the current step of a sequence through the goal feedback"""
        feedback = MoveManipulator.Feedback()
        feedback.state = state
        goal_handle.publish_feedback(feedback)

    def _feedback_forwarder(self, handler: ActionClientHandler, goal_handle):
        """Create a listener that republishes move_group feedback on our goal"""

        def _forward():
            feedback = handler.feedback_msg
            if feedback is None:
                return
            state = getattr(getattr(feedback, "feedback", feedback), "state", "")
            if state:
                self._publish_state(goal_handle, str(state))

        return _forward

    def _await_result(
        self, handler: ActionClientHandler, goal_handle, deadline: float
    ) -> str:
        """Wait for a client goal to return, propagating cancellation

        :returns: One of 'returned', 'canceled', 'preempted', 'timeout'
        """
        poll_period = 1 / self.config.loop_rate
        while not handler.action_returned:
            if not goal_handle.is_active:
                handler.cancel_request()
                return "preempted"
            if goal_handle.is_cancel_requested:
                handler.cancel_request()
                return "canceled"
            if time.time() > deadline:
                handler.cancel_request()
                return "timeout"
            time.sleep(poll_period)
        return "returned"

    # =========================================================================
    # MAIN ACTION
    # =========================================================================

    def main_action_callback(self, goal_handle):
        """
        Callback for the MoveIt component main action server

        :param goal_handle: Incoming action goal
        :type goal_handle: MoveManipulator.Goal

        :return: Action result
        :rtype: MoveManipulator.Result
        """
        goal = goal_handle.request
        result = MoveManipulator.Result()
        result.success = False

        handler: Optional[ActionClientHandler] = None
        listener = None
        deadline = time.time() + self.config.execution_timeout

        try:
            mode = MoveMode.from_goal(goal)
            self.get_logger().info(f"Received a '{mode.value}' motion goal")

            self._refresh_before_goal(goal_handle)

            if mode is MoveMode.PICK:
                outcome = self._execute_pick(goal, goal_handle, result, deadline)
                return self._finish_sequence(outcome, goal_handle, result)

            if mode is MoveMode.PLACE:
                outcome = self._execute_place(goal, goal_handle, result, deadline)
                return self._finish_sequence(outcome, goal_handle, result)

            if mode is MoveMode.CARTESIAN:
                trajectory = self._plan_cartesian_path(goal, result)
                if trajectory is None:
                    return self._terminate(goal_handle, result, abort=True)
                if goal.plan_only:
                    result.success = True
                    result.message = (
                        f"Planned {result.cartesian_fraction:.2%} of the requested path"
                    )
                    return self._terminate(goal_handle, result)

                from moveit_msgs.action import ExecuteTrajectory

                execute_goal = ExecuteTrajectory.Goal()
                execute_goal.trajectory = trajectory
                handler = self._exec_client
                handler.reset()
                if not handler.send_request(execute_goal):
                    result.message = (
                        "move_group did not accept the trajectory for execution"
                    )
                    return self._terminate(goal_handle, result, abort=True)
            else:
                move_goal = self._build_move_goal(goal, mode)
                handler = self._move_client
                handler.reset()
                listener = self._feedback_forwarder(handler, goal_handle)
                handler.add_feedback_listener(listener)
                if not handler.send_request(move_goal):
                    result.message = (
                        "move_group did not accept the motion goal. Is the "
                        f"'{self._resolve_name('move_action')}' server available?"
                    )
                    return self._terminate(goal_handle, result, abort=True)

            outcome = self._await_result(handler, goal_handle, deadline)
            if terminated := self._terminate_by_wait(
                outcome, goal_handle, result, listener, handler
            ):
                return terminated

            error_code = getattr(handler.action_result, "error_code", None)
            code = getattr(error_code, "val", 0)
            result.error_code = code
            result.message = moveit_error_string(code)
            result.success = result.message == "SUCCESS"
            return self._terminate(
                goal_handle,
                result,
                abort=not result.success,
                listener=listener,
                handler=handler,
            )

        except Exception as e:
            self.get_logger().error(f"Motion execution error - {e}")
            result.message = str(e)
            return self._terminate(
                goal_handle, result, abort=True, listener=listener, handler=handler
            )

    def _build_move_goal(self, goal: Any, mode: MoveMode) -> Any:
        """Build a MoveGroup goal for a pose, joints or named target"""
        group_name = self.config.arm_group_name
        velocity, acceleration = self._scalings(goal)

        if mode is MoveMode.POSE:
            target_pose = goal.target_pose
            if not target_pose.header.frame_id:
                target_pose.header.frame_id = self.config.pose_reference_frame
            constraints = pose_to_constraints(
                target_pose,
                link_name=self.config.end_effector_link,
                position_tolerance=self.config.goal_position_tolerance,
                orientation_tolerance=self.config.goal_orientation_tolerance,
            )
        elif mode is MoveMode.JOINTS:
            if len(goal.joint_names) != len(goal.joint_positions):
                raise ValueError(
                    "joint_names and joint_positions must have the same length, got "
                    f"{len(goal.joint_names)} and {len(goal.joint_positions)}"
                )
            constraints = joints_to_constraints(
                dict(zip(goal.joint_names, goal.joint_positions)),
                tolerance=self.config.goal_joint_tolerance,
            )
        else:
            group_name, joint_positions = self._resolve_named_target(goal.named_target)
            constraints = joints_to_constraints(
                joint_positions, tolerance=self.config.goal_joint_tolerance
            )

        return build_move_group_goal(
            constraints,
            group_name,
            pipeline_id=self.config.planning_pipeline,
            planner_id=self.config.planner_id,
            num_attempts=self.config.num_planning_attempts,
            allowed_time=self.config.allowed_planning_time,
            velocity_scaling=velocity,
            acceleration_scaling=acceleration,
            plan_only=goal.plan_only,
        )

    def _plan_cartesian_path(self, goal: Any, result: Any) -> Optional[Any]:
        """Compute a Cartesian path, returning the trajectory if usable"""
        velocity, acceleration = self._scalings(goal)
        request = build_cartesian_request(
            self._orient_waypoints(goal.cartesian_waypoints),
            frame_id=goal.frame_id or self.config.pose_reference_frame,
            # A dedicated Cartesian group, if any. Carries the orientation-tracking IK
            group_name=self.config.cartesian_group_name or self.config.arm_group_name,
            link_name=self.config.end_effector_link,
            max_step=self.config.cartesian_max_step,
            jump_threshold=self.config.cartesian_jump_threshold,
            avoid_collisions=self.config.cartesian_avoid_collisions,
            velocity_scaling=velocity,
            acceleration_scaling=acceleration,
        )
        # NOTE: The service handler waits for the server with a timeout, but
        # blocks until the response arrives once the request is sent
        response = self._cartesian_client.send_request(request)
        if not response:
            result.message = (
                "No response from the Cartesian path service at "
                f"'{self._resolve_name('compute_cartesian_path')}'"
            )
            return None

        result.cartesian_fraction = float(response.fraction)
        result.error_code = response.error_code.val
        if response.fraction < self.config.cartesian_fraction_threshold:
            result.message = (
                f"Only {response.fraction:.2%} of the requested Cartesian path is "
                f"achievable, below the configured threshold of "
                f"{self.config.cartesian_fraction_threshold:.2%}"
            )
            return None
        return response.solution

    def _terminate(
        self,
        goal_handle,
        result: Any,
        abort: bool = False,
        cancel: bool = False,
        listener=None,
        handler: Optional[ActionClientHandler] = None,
    ) -> Any:
        """Transition the goal to its terminal state and clean up

        A goal that is no longer active was already terminated elsewhere
        (preempted by a new goal), so it is not transitioned again.
        """
        if listener and handler:
            handler.remove_feedback_listener(listener)

        interrupted_grasp = None
        with self._main_goal_lock:
            if not goal_handle.is_active:
                self.get_logger().info(
                    "Goal already terminated (preempted by a new goal), stopping motion"
                )
            elif cancel or goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info("Goal canceled by client")
            elif abort:
                goal_handle.abort()
                self.get_logger().error(f"Motion failed: {result.message}")
                self.health_status.set_fail_component()
            else:
                goal_handle.succeed()
                if self._contact_freeze and not goal_handle.request.plan_only:
                    # if first executed motion after pick or place succeeds
                    # the contact allowance is retired and refreshes resume
                    interrupted_grasp, self._contact_freeze = (
                        self._contact_freeze,
                        None,
                    )
                self.get_logger().info(f"Motion completed: {result.message}")
                self.health_status.set_healthy()
        if interrupted_grasp:
            # a scene service round trip, kept outside the goal lock
            self._set_grasp_contact(interrupted_grasp, False)
        return result

    def _terminate_by_wait(
        self,
        waited: str,
        goal_handle,
        result: Any,
        listener=None,
        handler: Optional[ActionClientHandler] = None,
    ) -> Optional[Any]:
        """Terminate a goal whose wait ended without a result, if it did.

        :param waited: What _await_result reported
        :returns: The terminated result, or None when the motion returned
            normally and the caller should read its outcome
        """
        if waited == "preempted":
            return self._terminate(
                goal_handle, result, listener=listener, handler=handler
            )
        if waited == "canceled":
            result.message = "Motion canceled"
            return self._terminate(
                goal_handle, result, cancel=True, listener=listener, handler=handler
            )
        if waited == "timeout":
            result.message = (
                f"Motion did not complete within {self.config.execution_timeout}s"
            )
            return self._terminate(
                goal_handle, result, abort=True, listener=listener, handler=handler
            )
        return None

    def _command_gripper(
        self, named_target: Optional[str], position: float, description: str
    ) -> Tuple[bool, str]:
        """Send a gripper command through the configured backend

        :returns: (success, human readable description of what happened)
        """
        if not self._gripper_client:
            return False, "The component is not active, cannot command the gripper"

        handler = self._gripper_client
        handler.reset()

        if self.config.gripper_mode == "gripper_command":
            from control_msgs.action import GripperCommand

            goal = GripperCommand.Goal()
            goal.command.position = position
            goal.command.max_effort = self.config.gripper_max_effort
        else:
            if not self.config.gripper_group_name:
                return False, (
                    "No gripper is configured. Set 'gripper_group_name' in "
                    "MoveItConfig to control a gripper through move_group."
                )
            try:
                group, joint_positions = self._resolve_named_target(named_target)
            except ValueError as e:
                return False, str(e)
            goal = build_move_group_goal(
                joints_to_constraints(
                    joint_positions, tolerance=self.config.goal_joint_tolerance
                ),
                group,
                pipeline_id=self.config.planning_pipeline,
                planner_id=self.config.planner_id,
                num_attempts=self.config.num_planning_attempts,
                allowed_time=self.config.allowed_planning_time,
                velocity_scaling=self.config.max_velocity_scaling,
                acceleration_scaling=self.config.max_acceleration_scaling,
            )

        if not handler.send_request(goal):
            return False, f"Gripper '{description}' command was not accepted"

        deadline = time.time() + self.config.execution_timeout
        while not handler.action_returned and time.time() < deadline:
            time.sleep(1 / self.config.loop_rate)

        if not handler.action_returned:
            handler.cancel_request()
            return False, f"Gripper '{description}' command timed out"

        error_code = getattr(handler.action_result, "error_code", None)
        if error_code is not None:
            status = moveit_error_string(getattr(error_code, "val", 0))
            if status != "SUCCESS":
                return False, f"Gripper '{description}' command failed: {status}"
        return True, f"Gripper moved to '{description}'"

    # =========================================================================
    # PLANNING SCENE
    # =========================================================================

    def _apply_scene(self, collision_objects=(), attached_objects=(), acm=None) -> bool:
        """Apply scene changes as one diff through move_group.

        :param collision_objects: World objects to add, move or remove
        :param attached_objects: Objects to attach to or detach from the robot
        :param acm: Complete AllowedCollisionMatrix to install, None leaves
            the matrix alone
        :returns: Whether move_group confirmed the change
        """
        from moveit_msgs.srv import ApplyPlanningScene

        from ..utils.moveit import build_scene_diff

        if not self._apply_scene_client:
            self.get_logger().error(
                "The planning scene client is not available, is the component active?"
            )
            return False

        request = ApplyPlanningScene.Request()
        request.scene = build_scene_diff(collision_objects, attached_objects, acm)
        response = self._apply_scene_client.send_request(request)
        if response is None or not response.success:
            self.get_logger().error(
                "move_group did not apply the planning scene change"
            )
            return False
        return True

    def _set_grasp_contact(self, object_id: str, allowed: bool) -> None:
        """Toggle the grasp-contact allowance between the gripper and a target.

        During the pick, the touch links may meet the target's collision box,
        and every other link keeps avoiding it. Best effort: on scene trouble the
        pick proceeds with the contact still forbidden.
        """
        from moveit_msgs.msg import PlanningSceneComponents
        from moveit_msgs.srv import GetPlanningScene

        from ..utils.moveit import set_acm_contact

        # get touch links
        links = self._resolve_touch_links(None)
        if not (self._get_scene_client and self._apply_scene_client and links):
            return
        # get current allowed collision matrix
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        response = self._get_scene_client.send_request(request)
        if response is None:
            self.get_logger().warning(
                "Could not read the allowed collision matrix; gripper contact "
                f"with '{object_id}' stays as it was"
            )
            return
        # set new acm
        acm = set_acm_contact(
            response.scene.allowed_collision_matrix, links, object_id, allowed
        )
        if self._apply_scene(acm=acm):
            self.get_logger().info(
                f"Grasp contact {'allowed' if allowed else 'forbidden'}: "
                f"{links} x '{object_id}'"
            )

    def _resolve_touch_links(self, explicit: Optional[List[str]]) -> List[str]:
        """Links allowed to stay in contact with an attached object.

        Priority: the `touch_links` passed to the `attach_object` call, then
        the configured `touch_links`, then the robot SRDF (gripper group
        links or the end effector's neighborhood), then the end effector
        link alone with a warning.
        """
        from ..utils.moveit import derive_touch_links

        if not explicit and not self.config.touch_links and self._srdf is None:
            # Fetch SRDF and cache along with the named targets
            self._named_target_states()
        return derive_touch_links(
            self._srdf,
            self.config.end_effector_link,
            gripper_group=self.config.gripper_group_name or "",
            explicit=explicit or self.config.touch_links,
            logger_name=self.node_name,
        )

    def _do_attach(
        self,
        object_id: str,
        link_name: str = "",
        touch_links: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Attach a scene object to a robot link.

        :returns: (success, human readable description of what happened)
        """
        from ..utils.moveit import build_attached_object

        link = link_name or self.config.end_effector_link
        if not link:
            return False, (
                "No link to attach to: pass link_name or set "
                "'end_effector_link' in the component config"
            )

        links = self._resolve_touch_links(touch_links)
        if not self._apply_scene(
            attached_objects=[build_attached_object(object_id, link, links)]
        ):
            return False, f"Could not attach '{object_id}' to '{link}'"

        with self._scene_lock:
            if entry := self._scene_objects.get(object_id):
                entry["attached"] = link
        return True, (
            f"Attached '{object_id}' to '{link}' (touch links: {', '.join(links)})"
        )

    def _do_detach(self, object_id: str, remove: bool = False) -> Tuple[bool, str]:
        """Detach an attached object, returning it to the world.

        :returns: (success, human readable description of what happened)
        """
        from ..utils.moveit import build_detach_object, build_remove_object

        if not self._apply_scene(attached_objects=[build_detach_object(object_id)]):
            return False, f"Could not detach '{object_id}'"

        with self._scene_lock:
            if entry := self._scene_objects.get(object_id):
                entry.pop("attached", None)

        if not remove:
            return True, f"Detached '{object_id}', it remains in the scene"

        # Detaching returns the object to the world, so removing it is a
        # separate scene change
        if not self._apply_scene([build_remove_object(object_id)]):
            return False, (
                f"Detached '{object_id}' but could not remove it from the scene"
            )
        with self._scene_lock:
            self._scene_objects.pop(object_id, None)
        return True, f"Detached '{object_id}' and removed it from the scene"

    def _refresh_scene_from_detections(self, message: Optional[Any] = None) -> str:
        """Push detections into the planning scene as one diff.

        Objects the detector stopped reporting are kept for
        `scene_object_ttl` seconds before a refresh removes them, so
        detection dropouts do not flicker. Attached objects are never evicted
        or re-added. While an object is held the scene is frozen entirely. The
        first refresh after release reconciles it. The scene is equally frozen
        while a grasp-contact allowance is active since the allowance is keyed
        to a detection id, and a refresh could hand that id to a different object

        :param message: Detections to refresh from, when the caller has
            already read them. Default reads the latest received message
        :returns: What happened, for the caller and for tool use
        """
        from ..utils.moveit import (
            build_remove_object,
            collision_objects_from_detections,
        )

        with self._scene_lock:
            held = any(entry.get("attached") for entry in self._scene_objects.values())
        if held or self._contact_freeze:
            return (
                (
                    "Scene refresh skipped while an object is held: the scene "
                    "stays as captured before the grasp, until release"
                )
                if held
                else (
                    f"Scene refresh skipped while gripper contact with "
                    f"'{self._contact_freeze}' is allowed: the allowance is "
                    "keyed to its detection id, which a refresh could re-rank "
                    "onto a different object. Refreshes resume once a motion "
                    "succeeds, or when update_planning_scene is called "
                    "explicitly"
                )
            )

        if message is None:
            if not self._detections_topic:
                return (
                    "No detections input is connected to this component, so "
                    "there is nothing to update the planning scene from"
                )
            callback = self.callbacks.get(self._detections_topic.name)
            # the callback's default output is a text summary for prompts, get
            # the message itself for the boxes
            message = callback.get_output(get_msg=True) if callback else None
            if message is None:
                return "No detections have been received yet"

        now = time.time()
        objects, stale = self._partition_scene_changes(
            collision_objects_from_detections(
                message,
                min_thickness=self.config.min_object_thickness,
                padding=self.config.object_padding,
                include_labels=self.config.scene_detection_labels,
            ),
            now,
        )

        changes = objects + [build_remove_object(object_id) for object_id in stale]
        if not changes:
            return "The planning scene is already up to date"
        if not self._apply_scene(changes):
            return "Could not update the planning scene"

        added = []
        with self._scene_lock:
            for obj in objects:
                # allocate only on first sighting, bump the rest in place
                entry = self._scene_objects.setdefault(
                    obj.id, {"source": "detection", "last_seen": now}
                )
                entry["last_seen"] = now
                # an ADD moves an existing object, so the geometry follows it
                position = obj.primitive_poses[0].position
                dimensions = obj.primitives[0].dimensions
                entry["center"] = (position.x, position.y, position.z)
                entry["size"] = (dimensions[0], dimensions[1], dimensions[2])
                # NOTE: The padded box overlaps whatever the object is resting
                # on, which would put the first post-grasp motion's start state
                # in collision. Add the measured extents without the planning
                # padding, for re-shaping the box at attachment.
                padding = 2 * self.config.object_padding
                entry["attach_size"] = tuple(
                    max(extent - padding, self.config.min_object_thickness)
                    for extent in entry["size"]
                )
                added.append(obj.id)
            for object_id in stale:
                self._scene_objects.pop(object_id, None)

        summary = f"Planning scene updated with {len(added)} detected object(s)"
        if added:
            summary += f": {', '.join(added)}"
        if stale:
            summary += f"; removed {len(stale)} stale object(s)"
        return summary

    def _partition_scene_changes(
        self, objects: List[Any], now: float
    ) -> Tuple[List[Any], List[str]]:
        """Split a refresh into objects to add and tracked ids gone stale.

        Attached objects are exempt from both sides, the rest expire by
        their last sighting. Refreshes are frozen while holding, so attached
        entries normally never reach this; the exemption remains as the
        guard for an attach landing between the freeze check and this
        partition.

        :param objects: Collision objects built from the latest detections
        :param now: Refresh time, for the TTL comparison
        :returns: (objects to add, tracked ids to remove)
        """
        seen = {obj.id for obj in objects}
        attached, stale = set(), []
        with self._scene_lock:
            for object_id, entry in self._scene_objects.items():
                if entry.get("attached"):
                    attached.add(object_id)
                elif (
                    entry["source"] == "detection"
                    and object_id not in seen
                    and now - entry["last_seen"] > self.config.scene_object_ttl
                ):
                    stale.append(object_id)
        if attached:
            objects = [obj for obj in objects if obj.id not in attached]
        return objects, stale

    def _scene_refresh_tick(self):
        """The `continuous` mode scene refresh. Skips ticks that would change nothing.

        A tick refreshes only when a detections message arrived since the
        last applied one, or a tracked object has outlived its TTL and is due
        for removal.
        """
        callback = self.callbacks.get(self._detections_topic.name)
        # the callback's default output is a text summary for prompts, get
        # the message itself for the boxes
        message = callback.get_output(get_msg=True) if callback else None
        if message is None:
            return

        now = time.time()
        if message is self._last_scene_message and not self._has_stale_objects(now):
            # Dont refresh
            return

        self._last_scene_message = message
        result = self._refresh_scene_from_detections(message)
        self.get_logger().debug(result)

    def _has_stale_objects(self, now: float) -> bool:
        """Whether any tracked detection has outlived the TTL"""
        with self._scene_lock:
            return any(
                entry["source"] == "detection"
                and not entry.get("attached")
                and now - entry["last_seen"] > self.config.scene_object_ttl
                for entry in self._scene_objects.values()
            )

    def _refresh_before_goal(self, goal_handle) -> None:
        """The on_goal mode scene refresh, before any planning starts.

        NOTE: The scene is advisory. A failure here doesnt abort the mission.
        It degrades to planning against the previous scene.
        """
        if self.config.scene_update_mode != "on_goal" or not self._detections_topic:
            return
        self._publish_state(goal_handle, "UPDATING_SCENE")
        scene_status = self._refresh_scene_from_detections()
        self.get_logger().info(scene_status)

    # =========================================================================
    # PICK AND PLACE
    # =========================================================================
    # TODO: The step-sequencing machinery here (_finish_sequence,
    # _sequence_motion, per-step feedback) is a custom routine engine.
    # Rebase it on the Routine/MonitoredAction primitives once
    # https://github.com/automatika-robotics/sugarcoat/issues/55 lands with
    # result conditions, goal parameterization and inter-step data flow.

    def _finish_sequence(self, outcome: str, goal_handle, result: Any) -> Any:
        """Transition a sequence goal by the outcome of its last step"""
        if outcome != "ok" and self._contact_freeze:
            # an interrupted pick or place keeps its contact allowance and
            # freeze so the arm can retreat.
            result.message = (
                f"{result.message or 'Motion canceled'}. Gripper contact with "
                f"'{self._contact_freeze}' stays allowed and scene refreshes stay "
                "frozen until a motion succeeds"
            )
        if outcome == "canceled":
            return self._terminate(goal_handle, result, cancel=True)
        if outcome == "preempted":
            return self._terminate(goal_handle, result)
        result.success = outcome == "ok"
        return self._terminate(goal_handle, result, abort=not result.success)

    def _sequence_motion(
        self,
        handler: ActionClientHandler,
        motion_goal: Any,
        goal_handle,
        deadline: float,
    ) -> Tuple[str, str]:
        """Run one motion of a sequence to completion.

        :param handler: Action client carrying the motion (move or execute)
        :param motion_goal: Goal for that client
        :returns: (outcome, message) with outcome "ok", "aborted",
            "canceled" or "preempted"
        """
        handler.reset()
        if not handler.send_request(motion_goal):
            return "aborted", "move_group did not accept the motion goal"

        waited = self._await_result(handler, goal_handle, deadline)
        if waited == "preempted":
            return "preempted", "Goal preempted by a new goal"
        if waited == "canceled":
            return "canceled", "Motion canceled"
        if waited == "timeout":
            return "aborted", (
                f"Motion did not complete within {self.config.execution_timeout}s"
            )

        code = getattr(getattr(handler.action_result, "error_code", None), "val", 0)
        status = moveit_error_string(code)
        if status != "SUCCESS":
            return "aborted", f"Motion failed: {status}"
        return "ok", status

    def _pose_motion_goal(self, x: float, y: float, z: float, orientation=None) -> Any:
        """A collision-aware motion goal to an end-effector position.

        :param orientation: Optional end-effector orientation. None or
            identity means no preference, with the configured
            orientation tolerance giving the planner its freedom
        """
        from ..ros import ROSPoseStamped

        target = ROSPoseStamped()
        target.header.frame_id = self.config.pose_reference_frame
        position = target.pose.position
        position.x, position.y, position.z = float(x), float(y), float(z)
        if self._orientation_given(orientation):
            target.pose.orientation = orientation
        else:
            target.pose.orientation.w = 1.0

        constraints = pose_to_constraints(
            target,
            link_name=self.config.end_effector_link,
            position_tolerance=self.config.goal_position_tolerance,
            orientation_tolerance=self.config.goal_orientation_tolerance,
        )
        return build_move_group_goal(
            constraints,
            self.config.arm_group_name,
            pipeline_id=self.config.planning_pipeline,
            planner_id=self.config.planner_id,
            num_attempts=self.config.num_planning_attempts,
            allowed_time=self.config.allowed_planning_time,
            velocity_scaling=self.config.max_velocity_scaling,
            acceleration_scaling=self.config.max_acceleration_scaling,
        )

    def _sequence_descent(
        self, x: float, y: float, z: float, orientation, goal_handle, deadline: float
    ) -> Tuple[str, str]:
        """Straight-line end-effector motion with contact permitted.

        Collision checking is off for this one motion, so the gripper may reach
        into the target's collision box. Everything before and after remains
        collision-aware.

        :returns: (outcome, message), as in _sequence_motion
        """
        from moveit_msgs.action import ExecuteTrajectory

        from ..ros import ROSPose

        waypoint = ROSPose()
        waypoint.position.x, waypoint.position.y, waypoint.position.z = (
            float(x),
            float(y),
            float(z),
        )
        if self._orientation_given(orientation):
            waypoint.orientation = orientation
        # Without a preference the waypoint holds the orientation the
        # approach motion just reached: straight down at the current
        # orientation is achievable, an arbitrary default may not be
        waypoint = self._orient_waypoints([waypoint])[0]

        request = build_cartesian_request(
            [waypoint],
            frame_id=self.config.pose_reference_frame,
            group_name=self.config.cartesian_group_name or self.config.arm_group_name,
            link_name=self.config.end_effector_link,
            max_step=self.config.cartesian_max_step,
            jump_threshold=self.config.cartesian_jump_threshold,
            avoid_collisions=False,
            velocity_scaling=self.config.max_velocity_scaling,
            acceleration_scaling=self.config.max_acceleration_scaling,
        )
        response = self._cartesian_client.send_request(request)
        if not response:
            return "aborted", "No response from the Cartesian path service"
        if response.fraction < self.config.cartesian_fraction_threshold:
            return "aborted", (
                f"Only {response.fraction:.0%} of the straight-line segment is "
                "achievable"
            )

        execute_goal = ExecuteTrajectory.Goal()
        execute_goal.trajectory = response.solution
        return self._sequence_motion(
            self._exec_client, execute_goal, goal_handle, deadline
        )

    def _resolve_pick_target(
        self, goal: Any
    ) -> Tuple[str, Tuple[float, float, float], Tuple[float, float, float]]:
        """Find the scene object a pick goal refers to.

        Attached objects are never candidates (already held).

        :returns: (object id, center, size) from the scene bookkeeping
        :raises ValueError: When nothing in the scene matches
        """
        from ..utils.moveit import DETECTION_ID_PREFIX

        # Candidates: everything with known geometry, minus attached objects
        with self._scene_lock:
            candidates = {
                object_id: (entry["center"], entry["size"])
                for object_id, entry in self._scene_objects.items()
                if "center" in entry and not entry.get("attached")
            }

        name = (goal.target_object or "").strip()
        if name:
            # 1. target_object as an exact scene id
            if name in candidates:
                return name, *candidates[name]
            # 2. target_object as a detection label, best rank wins
            matches = [
                object_id
                for object_id in candidates
                if object_id.startswith(f"{DETECTION_ID_PREFIX}{name}_")
            ]
            if matches:
                # ranks order by score. 0 is the strongest detection
                best = min(matches, key=lambda object_id: (len(object_id), object_id))
                return best, *candidates[best]
            raise ValueError(
                f"No scene object matches '{name}'. Objects in the scene: "
                f"{sorted(candidates) or 'none'}"
            )

        # 3. If no name is given, take nearest object to target_pose, within the radius
        position = goal.target_pose.pose.position
        nearest = min(
            candidates.items(),
            key=lambda item: (
                (item[1][0][0] - position.x) ** 2
                + (item[1][0][1] - position.y) ** 2
                + (item[1][0][2] - position.z) ** 2
            ),
            default=None,
        )
        if nearest:
            object_id, (center, size) = nearest
            distance = (
                sum(
                    (a - b) ** 2
                    for a, b in zip(center, (position.x, position.y, position.z))
                )
                ** 0.5
            )
            if distance <= self.config.target_match_radius:
                return object_id, center, size
        raise ValueError(
            "No scene object within "
            f"{self.config.target_match_radius} m of the given pose. Add the "
            "object to the planning scene first, or name it with target_object."
        )

    def _execute_pick(
        self, goal: Any, goal_handle, result: Any, deadline: float
    ) -> str:
        """Grasp a scene object and lift it.

        Approach the object (collision-aware), open, descend straight onto it
        (contact permitted), close, attach, lift. With approach_mode "side"
        the approach comes from behind at grasp height, the descent becomes
        the slide into the grasp, and the lift goes straight up. Throughout,
        the touch links may contact the target's box (every other link keeps
        avoiding it) and scene refreshes are frozen. After an interruption
        both stay that way until a motion succeeds, so the arm can retreat.

        :returns: Outcome of the last step, for _finish_sequence
        """
        from ..utils.moveit import build_collision_object

        object_id, center, size = self._resolve_pick_target(goal)
        self.get_logger().info(f"Picking '{object_id}' at {center}")

        # allow the touch links to meet the target's box, while every other
        # link keeps avoiding it.
        self._contact_freeze = object_id
        self._set_grasp_contact(object_id, True)
        orientation = goal.target_pose.pose.orientation

        above = (
            center[0],
            center[1],
            center[2] + size[2] / 2 + self.config.approach_clearance,
        )
        approach = above
        side = self.config.approach_mode == "side"
        if side:
            # Pre-grasp BEHIND the target at grasp height, backed off along
            # the base->target bearing
            horizontal = (center[0] ** 2 + center[1] ** 2) ** 0.5
            standoff = max(size[0], size[1]) / 2 + self.config.approach_clearance
            scale = (
                max(horizontal - standoff, 0.0) / horizontal
                if horizontal > 1e-6
                else 0.0
            )
            approach = (center[0] * scale, center[1] * scale, center[2])

        self._publish_state(goal_handle, "PRE_GRASP")
        outcome, message = self._sequence_motion(
            self._move_client,
            self._pose_motion_goal(*approach, orientation),
            goal_handle,
            deadline,
        )
        if outcome != "ok":
            result.message = f"Pre-grasp: {message}"
            return outcome

        self._publish_state(goal_handle, "OPEN_GRIPPER")
        ok, message = self._command_gripper(
            named_target=self.config.gripper_open_target,
            position=self.config.gripper_open_position,
            description="open",
        )
        if not ok:
            result.message = message
            return "aborted"

        self._publish_state(goal_handle, "DESCEND")
        outcome, message = self._sequence_descent(
            center[0], center[1], center[2], orientation, goal_handle, deadline
        )
        if outcome != "ok":
            result.message = f"Descent: {message}"
            return outcome

        self._publish_state(goal_handle, "GRASP")
        ok, message = self._command_gripper(
            named_target=self.config.gripper_close_target,
            position=self.config.gripper_close_position,
            description="close",
        )
        if not ok:
            result.message = message
            return "aborted"

        self._publish_state(goal_handle, "ATTACH")
        # Attaching by id carries the world box along with the link, so
        # re-shape it first to the measured extents without planning padding
        # (manual objects were never padded and keep their size).
        with self._scene_lock:
            entry = self._scene_objects.get(object_id, {})
            attach_size = entry.get("attach_size", size)
        self._apply_scene([
            build_collision_object(
                object_id,
                self.config.pose_reference_frame,
                center,
                attach_size,
                min_thickness=self.config.min_object_thickness,
            )
        ])
        ok, message = self._do_attach(object_id)
        if not ok:
            result.message = message
            return "aborted"
        # The objects gripper contact is now governed by the attachment's
        # touch_links, so the world allowance is retired
        self._set_grasp_contact(object_id, False)
        self._contact_freeze = None

        self._publish_state(goal_handle, "LIFT")
        # In side mode the lift goes straight up, position-only
        outcome, message = self._sequence_motion(
            self._move_client,
            self._pose_motion_goal(*above, None if side else orientation),
            goal_handle,
            deadline,
        )
        if outcome != "ok":
            result.message = f"Lift: {message}"
            return outcome

        result.message = f"Picked '{object_id}'"
        return "ok"

    def _execute_place(
        self, goal: Any, goal_handle, result: Any, deadline: float
    ) -> str:
        """Set the held object down at the goal pose and release it.

        The inverse of pick: approach above the release pose (planned with
        the held object still attached), descend straight down (contact
        permitted), open, detach and retreat back up. The released object
        stays in the scene, and the touch links may contact it from the
        detach until the retreat succeeds — the pick's grasp contract,
        mirrored.

        :returns: Outcome of the last step, for _finish_sequence
        :raises ValueError: When nothing is attached
        """
        with self._scene_lock:
            held = next(
                (
                    (object_id, entry["size"])
                    for object_id, entry in self._scene_objects.items()
                    if entry.get("attached")
                ),
                None,
            )
        if not held:
            raise ValueError(
                "Nothing is attached to the gripper: pick an object before placing"
            )
        object_id, size = held

        # target_pose gives where the object's center ends up and the gripper
        # holds the object at its center
        position = goal.target_pose.pose.position
        orientation = goal.target_pose.pose.orientation
        approach_z = position.z + size[2] / 2 + self.config.approach_clearance
        self.get_logger().info(
            f"Placing '{object_id}' at "
            f"({position.x:.3f}, {position.y:.3f}, {position.z:.3f})"
        )

        self._publish_state(goal_handle, "PRE_PLACE")
        outcome, message = self._sequence_motion(
            self._move_client,
            self._pose_motion_goal(position.x, position.y, approach_z, orientation),
            goal_handle,
            deadline,
        )
        if outcome != "ok":
            result.message = f"Pre-place: {message}"
            return outcome

        self._publish_state(goal_handle, "DESCEND")
        outcome, message = self._sequence_descent(
            position.x, position.y, position.z, orientation, goal_handle, deadline
        )
        if outcome != "ok":
            result.message = f"Descent: {message}"
            return outcome

        self._publish_state(goal_handle, "RELEASE")
        ok, message = self._command_gripper(
            named_target=self.config.gripper_open_target,
            position=self.config.gripper_open_position,
            description="open",
        )
        if not ok:
            result.message = message
            return "aborted"

        self._publish_state(goal_handle, "DETACH")
        # NOTE: Detaching puts the object's box back into the world, wrapped by
        # the open jaws. Allow that contact first, or the retreat's start state
        # is born in collision.
        self._contact_freeze = object_id
        self._set_grasp_contact(object_id, True)
        ok, message = self._do_detach(object_id)
        if not ok:
            result.message = message
            return "aborted"
        # The object now sits where it was released, not where it was picked
        with self._scene_lock:
            if entry := self._scene_objects.get(object_id):
                entry["center"] = (position.x, position.y, position.z)
                entry["last_seen"] = time.time()

        self._publish_state(goal_handle, "RETREAT")
        outcome, message = self._sequence_motion(
            self._move_client,
            self._pose_motion_goal(position.x, position.y, approach_z, orientation),
            goal_handle,
            deadline,
        )
        if outcome != "ok":
            result.message = f"Retreat: {message}"
            return outcome
        # clear of the released object
        self._set_grasp_contact(object_id, False)
        self._contact_freeze = None

        result.message = f"Placed '{object_id}'"
        return "ok"

    # =========================================================================
    # COMPONENT ACTIONS
    # =========================================================================

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "open_gripper",
                "description": "Open the robot's gripper, for example before grasping an object or to release a held object.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        active=True,
        phase=ActionPhase.EXECUTION,
    )
    def open_gripper(self) -> str:
        """Open the gripper"""
        _, message = self._command_gripper(
            named_target=self.config.gripper_open_target,
            position=self.config.gripper_open_position,
            description="open",
        )
        return message

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "close_gripper",
                "description": "Close the robot's gripper, for example to grasp an object that the gripper is positioned around.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        active=True,
        phase=ActionPhase.EXECUTION,
    )
    def close_gripper(self) -> str:
        """Close the gripper"""
        _, message = self._command_gripper(
            named_target=self.config.gripper_close_target,
            position=self.config.gripper_close_position,
            description="close",
        )
        return message

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "set_gripper",
                "description": "Move the robot's gripper to a specific opening, for grasping objects of a known size. Only available when the gripper is controlled through a gripper controller.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "position": {
                            "type": "number",
                            "description": "Target gripper position, in the units used by the gripper controller (usually meters of finger opening).",
                        }
                    },
                    "required": ["position"],
                },
            },
        },
        active=True,
        phase=ActionPhase.EXECUTION,
    )
    def set_gripper(self, position: float) -> str:
        """Move the gripper to a given position"""
        if self.config.gripper_mode != "gripper_command":
            return (
                "Setting a specific gripper position requires gripper_mode to be "
                "'gripper_command'. Use open_gripper or close_gripper instead."
            )
        _, message = self._command_gripper(
            named_target=None, position=float(position), description=str(position)
        )
        return message

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "stop_motion",
                "description": "Immediately stop the arm and the gripper by cancelling any motion that is currently being executed.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        active=True,
        phase=ActionPhase.EXECUTION,
    )
    def stop_motion(self) -> str:
        """Cancel any motion in progress"""
        stopped = []
        for name, handler in (
            ("arm", self._move_client),
            ("trajectory", self._exec_client),
            ("gripper", self._gripper_client),
        ):
            if handler and handler.goal_accepted:
                success, _ = handler.cancel_request()
                if success:
                    stopped.append(name)
        if not stopped:
            return "No motion is currently being executed"
        self.get_logger().warning(f"Stopped motion: {', '.join(stopped)}")
        return f"Stopped motion: {', '.join(stopped)}"

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "get_named_targets",
                "description": "List the named poses the robot arm and gripper can be sent to, as defined in the robot's configuration (e.g. 'home', 'ready').",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        phase=ActionPhase.BOTH,
    )
    def get_named_targets(self) -> str:
        """List the named targets available per planning group"""
        states = self._named_target_states()
        if not states:
            return "No named targets are defined for this robot"
        return "; ".join(
            f"{group}: {', '.join(sorted(group_states))}"
            for group, group_states in states.items()
            if group_states
        )

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "add_collision_object",
                "description": "Add a box-shaped obstacle to the motion planning scene, so that subsequent arm motions avoid it. Use for known fixed objects like tables, walls or shelves.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_id": {
                            "type": "string",
                            "description": "Name for the object in the scene. Re-using a name moves the existing object.",
                        },
                        "center": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Center of the box as [x, y, z] in meters.",
                        },
                        "size": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Full extents of the box as [x, y, z] in meters.",
                        },
                        "frame_id": {
                            "type": "string",
                            "description": "Frame the center is given in. Empty uses the planning frame.",
                        },
                    },
                    "required": ["object_id", "center", "size"],
                },
            },
        },
        active=True,
        phase=ActionPhase.EXECUTION,
    )
    def add_collision_object(
        self,
        object_id: str,
        center: List[float],
        size: List[float],
        frame_id: str = "",
    ) -> str:
        """Add a box-shaped collision object to the planning scene.

        :param object_id: Scene-wide name; re-using one moves the object
        :param center: Center of the box as [x, y, z] in meters
        :param size: Full extents of the box as [x, y, z] in meters
        :param frame_id: Frame the center is given in. Empty falls back to the
            configured `pose_reference_frame`, and an empty frame is read by
            move_group as its planning frame
        :returns: What happened, for the caller and for tool use
        """
        from ..utils.moveit import build_collision_object

        if len(center) != 3 or len(size) != 3:
            raise ValueError(
                "center and size must each have 3 values (x, y, z), got "
                f"{len(center)} and {len(size)}"
            )

        obj = build_collision_object(
            object_id,
            frame_id or self.config.pose_reference_frame,
            tuple(float(value) for value in center),
            tuple(float(value) for value in size),
            min_thickness=self.config.min_object_thickness,
        )
        if not self._apply_scene([obj]):
            return f"Could not add '{object_id}' to the planning scene"

        with self._scene_lock:
            # geometry as applied (thickness floor included)
            position = obj.primitive_poses[0].position
            dimensions = obj.primitives[0].dimensions
            self._scene_objects[object_id] = {
                "source": "manual",
                "last_seen": time.time(),
                "center": (position.x, position.y, position.z),
                "size": (dimensions[0], dimensions[1], dimensions[2]),
            }
        return f"Added '{object_id}' to the planning scene"

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "remove_collision_object",
                "description": "Remove an obstacle from the motion planning scene by its name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_id": {
                            "type": "string",
                            "description": "Name of the object to remove, as it was added or as listed by list_collision_objects.",
                        },
                    },
                    "required": ["object_id"],
                },
            },
        },
        active=True,
        phase=ActionPhase.EXECUTION,
    )
    def remove_collision_object(self, object_id: str) -> str:
        """Remove one collision object from the planning scene.

        :param object_id: Id the object was added under
        :returns: What happened, for the caller and for tool use
        """
        from ..utils.moveit import build_remove_object

        if not self._apply_scene([build_remove_object(object_id)]):
            return f"Could not remove '{object_id}' from the planning scene"

        with self._scene_lock:
            self._scene_objects.pop(object_id, None)
        return f"Removed '{object_id}' from the planning scene"

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "clear_collision_objects",
                "description": "Remove the obstacles this component has added to the motion planning scene.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "detections_only": {
                            "type": "boolean",
                            "description": "Only remove objects that came from detections, keeping manually added ones. Default is false.",
                        },
                    },
                    "required": [],
                },
            },
        },
        active=True,
        phase=ActionPhase.EXECUTION,
    )
    def clear_collision_objects(self, detections_only: bool = False) -> str:
        """Remove the collision objects this component put in the scene.

        Only objects this component knows it added are touched, so obstacles
        placed by other tools stay where they are.

        :param detections_only: Only remove detection-sourced objects,
            keeping manually added ones
        :returns: What happened, for the caller and for tool use
        """
        from ..utils.moveit import build_remove_object

        with self._scene_lock:
            ids = [
                object_id
                for object_id, entry in self._scene_objects.items()
                if not detections_only or entry["source"] == "detection"
            ]
        if not ids:
            return "The planning scene holds no objects added by this component"

        if not self._apply_scene([build_remove_object(i) for i in ids]):
            return "Could not clear the planning scene objects"

        with self._scene_lock:
            for object_id in ids:
                self._scene_objects.pop(object_id, None)
        return f"Removed {len(ids)} object(s) from the planning scene"

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "list_collision_objects",
                "description": "List the obstacles currently in the motion planning scene.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        active=True,
        phase=ActionPhase.BOTH,
    )
    def list_collision_objects(self) -> str:
        """List the collision objects in the planning scene.

        The scene is read back from move_group, which is authoritative: it
        also shows objects placed there by other tools. When move_group does
        not answer, the objects this component itself added are listed.

        :returns: The object names, for the caller and for tool use
        """
        from moveit_msgs.msg import PlanningSceneComponents
        from moveit_msgs.srv import GetPlanningScene

        if self._get_scene_client:
            request = GetPlanningScene.Request()
            request.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
            response = self._get_scene_client.send_request(request)
            if response is not None:
                names = sorted(o.id for o in response.scene.world.collision_objects)
                if not names:
                    return "The planning scene contains no collision objects"
                return f"Collision objects in the planning scene: {', '.join(names)}"

        with self._scene_lock:
            names = sorted(self._scene_objects)
        if not names:
            return "The planning scene contains no collision objects"
        return (
            "move_group did not answer; objects added by this component: "
            f"{', '.join(names)}"
        )

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "clear_octomap",
                "description": "Clear the octomap built from depth sensors in the motion planning scene. Use when stale sensor data blocks valid motions.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        active=True,
        phase=ActionPhase.EXECUTION,
    )
    def clear_octomap(self) -> str:
        """Clear the octomap accumulated in the planning scene.

        :returns: What happened, for the caller and for tool use
        """
        if not self._clear_octomap_client:
            return "The octomap client is not available, is the component active?"

        response = self._clear_octomap_client.send_request(Empty.Request())
        if response is None:
            return "move_group did not answer the octomap clear request"
        return "Cleared the planning scene octomap"

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "attach_object",
                "description": "Attach a planning scene object to the robot's gripper, so it moves with the arm and contact with the gripper stops counting as a collision. Call right after closing the gripper on an object; without it, the first motion while holding the object fails on a collision between gripper and object.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_id": {
                            "type": "string",
                            "description": "Name of the object in the planning scene, as listed by list_collision_objects.",
                        },
                        "link_name": {
                            "type": "string",
                            "description": "Robot link to attach the object to. Default is the configured end effector link.",
                        },
                    },
                    "required": ["object_id"],
                },
            },
        },
        active=True,
        phase=ActionPhase.EXECUTION,
    )
    def attach_object(
        self,
        object_id: str,
        link_name: str = "",
        touch_links: Optional[List[str]] = None,
    ) -> str:
        """Attach a scene object to a robot link, as on a grasp.

        move_group removes the object from the world and carries it with the
        link; contact with the touch links stops counting as collision.

        :param object_id: Name of an object already in the planning scene
        :param link_name: Link the object is rigidly attached to. Empty uses
            the configured end effector link
        :param touch_links: Links allowed to stay in contact with the object.
            Default resolves them from the config or the robot SRDF
        :returns: What happened, for the caller and for tool use
        """
        _, message = self._do_attach(object_id, link_name, touch_links)
        return message

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "detach_object",
                "description": "Detach a held object from the robot's gripper in the planning scene. Call after releasing an object. By default the object stays in the scene where it was released.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_id": {
                            "type": "string",
                            "description": "Name of the attached object.",
                        },
                        "remove": {
                            "type": "boolean",
                            "description": "Also delete the object from the scene entirely, e.g. after handing it away. Default is false.",
                        },
                    },
                    "required": ["object_id"],
                },
            },
        },
        active=True,
        phase=ActionPhase.EXECUTION,
    )
    def detach_object(self, object_id: str, remove: bool = False) -> str:
        """Detach an attached object, returning it to the world.

        :param object_id: Name the object was attached under
        :param remove: Also remove the object from the scene entirely. The
            default keeps it where it was released, which matches what
            physically happened
        :returns: What happened, for the caller and for tool use
        """
        _, message = self._do_detach(object_id, remove)
        return message

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "update_planning_scene",
                "description": "Update the motion planning scene from the latest camera detections, so that the next motions plan around the objects currently seen. Call before planning a motion in a scene that may have changed. While an object is held in the gripper the scene is frozen and this does nothing; it resumes after release.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        active=True,
        phase=ActionPhase.EXECUTION,
    )
    def update_planning_scene(self) -> str:
        """Update the planning scene from the latest received detections.

        Calling this explicitly overrides the automatic freeze left by an
        interrupted pick or place. The freeze while an object is held stands.

        :returns: What happened, for the caller and for tool use
        """
        if self._contact_freeze:
            self._set_grasp_contact(self._contact_freeze, False)
            self._contact_freeze = None
        return self._refresh_scene_from_detections()

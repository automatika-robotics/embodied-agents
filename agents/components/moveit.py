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
    SRDF_PARAMETER
)
from .component_base import Component


class MoveIt(Component):
    """
    This component provides classical, collision aware manipulation by driving a running [MoveIt 2](https://moveit.ai) `move_group` node.

    The component runs as a ROS2 Action Server exposing the `<component_name>/manipulate_with_moveit` action, which takes a motion target and plans and executes a path to it. Four kinds of targets are supported, selected with the goal's `mode` field (or inferred from the fields that are populated):

    - **pose**: an end-effector pose, planned with the configured planner and reached within the configured position/orientation tolerances.
    - **joints**: explicit joint positions.
    - **named**: a named target defined in the robot's SRDF (e.g. "home", "ready"), resolved to joint positions by reading the SRDF from the running `move_group` node.
    - **cartesian**: a straight-line end-effector path through a list of waypoints, computed with MoveIt's Cartesian path service and executed only if enough of the path is achievable (see `cartesian_fraction_threshold`).

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

        # Named targets read from the SRDF, resolved lazily and cached
        self._named_targets: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None

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
        ):
            if handler:
                self.destroy_client(handler.client)
        self._cartesian_client = self._planner_client = self._param_client = None
        self._apply_scene_client = self._get_scene_client = None
        self._clear_octomap_client = None

    def custom_on_activate(self):
        """Activate component and check the configured planner against move_group"""
        super().custom_on_activate()
        self._validate_planner_config()

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

    def _scalings(self, goal: Any) -> Tuple[float, float]:
        """Per-goal scaling overrides, falling back to the configured values"""
        velocity = goal.velocity_scaling or self.config.max_velocity_scaling
        acceleration = goal.acceleration_scaling or self.config.max_acceleration_scaling
        return float(velocity), float(acceleration)

    def _feedback_forwarder(self, handler: ActionClientHandler, goal_handle):
        """Create a listener that republishes move_group feedback on our goal"""

        def _forward():
            feedback = handler.feedback_msg
            if feedback is None:
                return
            state = getattr(getattr(feedback, "feedback", feedback), "state", "")
            if not state:
                return
            message = MoveManipulator.Feedback()
            message.state = str(state)
            goal_handle.publish_feedback(message)

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

            if outcome == "preempted":
                return self._terminate(
                    goal_handle, result, listener=listener, handler=handler
                )
            if outcome == "canceled":
                result.message = "Motion canceled"
                return self._terminate(
                    goal_handle, result, cancel=True, listener=listener, handler=handler
                )
            if outcome == "timeout":
                result.message = (
                    f"Motion did not complete within {self.config.execution_timeout}s"
                )
                return self._terminate(
                    goal_handle, result, abort=True, listener=listener, handler=handler
                )

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
            goal.cartesian_waypoints,
            frame_id=goal.frame_id or self.config.pose_reference_frame,
            group_name=self.config.arm_group_name,
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
                self.get_logger().info(f"Motion completed: {result.message}")
                self.health_status.set_healthy()
        return result

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
        return self._command_gripper(
            named_target=self.config.gripper_open_target,
            position=self.config.gripper_open_position,
            description="open",
        )

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
        return self._command_gripper(
            named_target=self.config.gripper_close_target,
            position=self.config.gripper_close_position,
            description="close",
        )

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
        return self._command_gripper(
            named_target=None, position=float(position), description=str(position)
        )

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

    def _command_gripper(
        self, named_target: Optional[str], position: float, description: str
    ) -> str:
        """Send a gripper command through the configured backend"""
        if not self._gripper_client:
            return "The component is not active, cannot command the gripper"

        handler = self._gripper_client
        handler.reset()

        if self.config.gripper_mode == "gripper_command":
            from control_msgs.action import GripperCommand

            goal = GripperCommand.Goal()
            goal.command.position = position
            goal.command.max_effort = self.config.gripper_max_effort
        else:
            if not self.config.gripper_group_name:
                return (
                    "No gripper is configured. Set 'gripper_group_name' in "
                    "MoveItConfig to control a gripper through move_group."
                )
            try:
                group, joint_positions = self._resolve_named_target(named_target)
            except ValueError as e:
                return str(e)
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
            return f"Gripper '{description}' command was not accepted"

        deadline = time.time() + self.config.execution_timeout
        while not handler.action_returned and time.time() < deadline:
            time.sleep(1 / self.config.loop_rate)

        if not handler.action_returned:
            handler.cancel_request()
            return f"Gripper '{description}' command timed out"

        error_code = getattr(handler.action_result, "error_code", None)
        if error_code is not None:
            status = moveit_error_string(getattr(error_code, "val", 0))
            if status != "SUCCESS":
                return f"Gripper '{description}' command failed: {status}"
        return f"Gripper moved to '{description}'"

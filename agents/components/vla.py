from typing import Optional, List, Dict, Callable
import queue
import threading
import numpy as np
from rclpy.logging import get_logger

from ..config import VLAConfig
import time
from ..ros import (
    RGBD,
    Image,
    Topic,
    JointTrajectory,
    JointJog,
    JointState,
    ComponentRunType,
    MutuallyExclusiveCallbackGroup,
    VisionLanguageAction,
)
from ..utils import validate_func_args
from ..utils.actions import JointsData, find_missing_values, parse_urdf_joints
from ..clients.lerobot import LeRobotClient
from .model_component import ModelComponent


class VLA(ModelComponent):
    """Vision-Language-Agent (VLA) Component."""

    @validate_func_args
    def __init__(
        self,
        *,
        inputs: List[Topic],
        outputs: List[Topic],
        model_client: LeRobotClient,
        config: VLAConfig,
        component_name: str,
        **kwargs,
    ):
        self.config: VLAConfig = config
        self.allowed_inputs = {
            "Required": [JointState, [Image, RGBD]],
        }
        self.handled_outputs = [JointTrajectory, JointJog]

        self.model_client = model_client

        # Verify config and model definition
        self._verify_config(component_name)

        # Set the component to run as an action server and set the main action type
        self.run_type = ComponentRunType.ACTION_SERVER
        self.action_type = VisionLanguageAction

        # queue aggregation function
        self._aggregate_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = (
            lambda _, y: y
        )

        super().__init__(
            inputs,
            outputs,
            model_client,
            self.config,
            None,
            component_name,
            **kwargs,
        )

    def custom_on_activate(self):
        """Custom activation"""

        if not isinstance(self.model_client, LeRobotClient):
            raise TypeError(
                "Currenlty VLA component only takes in LeRobotClient. Please use LeRobot Policy Server to serve your VLA."
            )

        if self.config.warmup:
            # TODO: warmup with lerobot client
            self.get_logger().warning(
                "Warmup cannot not be called with LeRobot client."
            )
            self.config.warmup = False

        # Activate component and initialize client
        super().custom_on_activate()

        # Queue for receiving actions
        self._actions_received = queue.Queue()
        self._action_queue_lock = threading.Lock()

        # track last executed action timestep
        self._last_executed_timestep_lock = threading.Lock()
        self._last_executed_timestep = -1

        # Action timers
        self.__action_sending_timer = None
        self.__action_receiving_timer = None

        # Look for state topic
        self._state_topic = None
        for key, callback in self.callbacks.items():
            if callback.input_topic.msg_type == JointState:
                self._state_topic = key
                break
        if not self._state_topic:
            raise RuntimeError(
                "Could not find a topic of type JointState. VLA component needs at least one topic of type JointState as input."
            )

    def custom_on_deactivate(self):
        """Custom deactivation"""

        # mark any pendings actions as finished
        while not self._actions_received.empty():
            try:
                self._actions_received.get_nowait()
                self._actions_received.task_done()
            except queue.Empty:
                break

        # Deactivate component
        super().custom_on_deactivate()

    def _verify_config(self, component_name: str):
        """Run checks on provided config and model definition"""

        # Check dataset keys from model definition in keys from config
        joint_keys_missing = find_missing_values(
            self.config.joint_names_map.keys(), self.model_client.model.joint_keys
        )
        if joint_keys_missing:
            raise ValueError(
                f"Your 'joint_names_map' in VLAConfig does not map all the dataset joint names to the robot joint names correctly. The following joint names from the dataset info are unmapped: {joint_keys_missing}"
            )

        if self.config.robot_urdf_file:
            # Read robot joint limits
            self.robot_joints_limits = parse_urdf_joints(self.config.robot_urdf_file)

            # Check for mapping joint names in urdf
            joint_keys_missing = find_missing_values(
                self.robot_joints_limits.keys(),
                list(self.config.joint_names_map.values()),
            )
            get_logger(component_name).warning(
                f"Your 'joint_names_map' in VLAConfig includes robot joint names that do not exist in the provided URDF file. This might cause errors later on. The following joint names were not found in the URDF file: {joint_keys_missing}"
            )
        else:
            self.robot_joints_limits = None

        # TODO:: Handle partially available image keys with error logging
        image_keys_missing = find_missing_values(
            self.config.camera_inputs_map.keys(), self.model_client.model.image_keys
        )
        if image_keys_missing:
            raise ValueError(
                f"Your 'image_keys_missing' in VLAConfig does not map all the dataset camera names to the robot camera topics correctly. The following camera names from the dataset info are unmapped: {image_keys_missing}"
            )

    def _receive_actions_from_client(self):
        """Timer callback for continuosly receiving actions from client"""
        latest_actions = self.model_client.receive_actions()
        if latest_actions:
            self._update_actions_queue(latest_actions)

    def _update_actions_queue(
        self,
        new_actions: List,
    ):
        """Update actions queue with new result. Similar to LeRobot async_client implementation"""

        with self._actions_received.mutex:
            # Get internal deque
            internal_deque = self._actions_received.queue

            # Mapping: timestep -> TimedAction object
            action_map = {a.timestep: a for a in internal_deque}

            # Process new actions
            queue_modified = False

            for new_act in new_actions:
                ts = new_act.timestep

                # Get last executed action
                with self._last_executed_timestep_lock:
                    _latest_action = self._last_executed_timestep

                # Skip: actions that have already passed
                if ts <= _latest_action:
                    continue

                queue_modified = True

                # Aggregate: If timestep exists, merge arrays
                if ts in action_map:
                    existing_act = action_map[ts]

                    # Perform the aggregation on the array
                    merged_array = self._aggregate_fn(
                        existing_act.action, new_act.action
                    )

                    # Update the existing action object
                    existing_act.action = merged_array
                    existing_act.timestamp = new_act.timestamp

                # Insert: If timestep is new, add to map
                else:
                    action_map[ts] = new_act

            # Rebuild the Queue only if data changed
            if queue_modified:
                # Clear the internal deque
                internal_deque.clear()

                # Sort by timestep to ensure FIFO execution order
                sorted_timesteps = sorted(action_map.keys())

                # Bulk extend the deque
                internal_deque.extend([action_map[k] for k in sorted_timesteps])

                # Manually notify any consumers
                self._actions_received.not_empty.notify()

    def _get_action(self):
        with self._action_queue_lock:
            if self._actions_received.not_empty:
                # Return the immediate next action
                # TODO: Remove torch depenedency here once server can send numpy arrays
                return self._actions_received.get().action.numpy()
        return None

    def _create_input(self, task: str) -> Optional[Dict]:
        """Prepare observations from current inputs

        :param task: Task string
        :return: True if inference input is ready, False otherwise
        """
        if not self._state_topic:
            return
        joint_state: JointsData = self.callbacks[self._state_topic].get_output()

        # map robot state to dataset keys
        mapped_state = joint_state.get_mapped_state(
            self.config.state_input_type, self.config.joint_names_map
        )

        # Return if no mapped state found
        if not mapped_state:
            self.get_logger().warning(
                f"Did not receive all joint states of type: {self.config.state_input_type}, not sending input for inference"
            )
            return

        # Get images
        images = {}
        for key, value in self.config.camera_inputs_map.items():
            img_out = self.callbacks[value.name].get_output(clear_last=True)
            if img_out is None:
                self.get_logger().warning(
                    f"Did not receive an image for key: {key}, not sending input for inference"
                )
                return
            images[key] = img_out

        # Combine state, images and task
        inference_input = {
            "timestamp": time.time(),
            **mapped_state,
            **images,
            "task": task,
        }
        return inference_input

    def _send_action_commands(self):
        """Send action commands"""
        # TODO: implement sending a command to the robot from the commands queue
        pass

    def _destroy_action_timers(self):
        """Destroy action timers"""
        if self.__action_sending_timer is not None:
            self.destroy_timer(self.__action_sending_timer)
            self.__action_sending_timer = None
        if self.__action_receiving_timer is not None:
            self.destroy_timer(self.__action_receiving_timer)
            self.__action_receiving_timer = None

    def main_action_callback(self, goal_handle: VisionLanguageAction.Goal):
        """
        Callback for the VLA main action server

        :param goal_handle: Incoming action goal
        :type goal_handle: VisionLanguageAction.Goal

        :return: Action result
        :rtype: VisionLanguageAction.Result
        """
        # Clear the action queue
        self._actions_received.queue.clear()

        # Get request
        task: str = goal_handle.request.task

        # Setup response and feedback of the action
        action_feedback_msg = VisionLanguageAction.Feedback()

        action_result = VisionLanguageAction.Result()

        action_result.success = False

        # Create a timer to send actions at a fixed rate
        self.__action_sending_timer = self.create_timer(
            1 / self.config.action_sending_rate,
            self._send_action_commands,
        )

        # Create a tight timer for receiving actions from lerobot client
        self.__action_receiving_timer = self.create_timer(
            timer_period_sec=0.001,
            callback=self._receive_actions_from_client,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.get_logger().debug("Started timer for receiving actions")

        # Wait for all inputs to be available
        _timeout = 0.0
        while not self.got_all_inputs() and _timeout < self.config.input_timeout:
            self.get_logger().warn(
                "Inputs topics are not available, waiting to start executing actions...",
                once=True,
            )
            _timeout += 1 / self.config.loop_rate
            time.sleep(1 / self.config.loop_rate)

        try:
            while not self._action_done(action_feedback_msg):
                # Check if goal is canceled
                if not goal_handle.is_active or goal_handle.is_cancel_requested:
                    self._destroy_action_timers()
                    self.get_logger().info("Goal Canceled")
                    return action_result

                # Get new observations from inputs
                model_observations = self._create_input(task)

                if model_observations:
                    # Get last executed action
                    with self._last_executed_timestep_lock:
                        model_observations["timestep"] = self._last_executed_timestep
                    # send input for inference
                    self.model_client.inference(model_observations)
                else:
                    self.get_logger().warn(
                        "Could not prepare inference input, skipping this step..."
                    )
                    continue

                # Compute errors and publish feedback
                goal_handle.publish_feedback(action_feedback_msg)
                self.get_logger().debug(f"Action Feedback: {action_feedback_msg}")
                # NOTE: using Python time directly, as ros rate sleep (from self.create_rate) was not functioning as expected
                time.sleep(1 / self.config.loop_rate)

        except Exception as e:
            self.get_logger().error(f"Action execution error - {e}")
            with self._main_goal_lock:
                self._destroy_action_timers()
                goal_handle.abort()
                goal_handle.reset()

        # Get the final goal state
        # action_result = ...

        action_result.success = True
        # Publish zero commands to stop the robot

        with self._main_goal_lock:
            self._destroy_action_timers()
            goal_handle.succeed()

        return action_result

    def _action_done(self, action_feedback: VisionLanguageAction.Feedback) -> bool:
        """Check if action is done

        :param goal: Current action goal
        :return: True if action is done, False otherwise
        """
        # TODO: implement action done check
        # True if feedback errors are less than tolerance (for example)
        return False

    def _warmup(self):
        """Warm up and stat check"""
        # TODO: implement warmup
        pass

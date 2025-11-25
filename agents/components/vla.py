from typing import Optional, List
from ..clients.model_base import ModelClient
from ..config import VLAConfig
import time
from ..ros import (
    Image,
    String,
    Topic,
    JointTrajectory,
    JointJog,
    JointState,
    ComponentRunType,
)
from ..utils import validate_func_args
from .model_component import ModelComponent

from automatika_embodied_agents.action import VisionLanguageAction


class VLA(ModelComponent):
    """Vision-Language-Agent (VLA) Component."""

    @validate_func_args
    def __init__(
        self,
        *,
        inputs: List[Topic],
        outputs: List[Topic],
        model_client: ModelClient,
        config: Optional[VLAConfig] = None,
        component_name: str,
        **kwargs,
    ):
        self.config: VLAConfig = config or VLAConfig()
        self.allowed_inputs = {
            "Required": [String, Image, JointState],
        }
        self.handled_outputs = [[JointTrajectory, JointJog]]

        self.model_client = model_client

        # Set the component to run as an action server and set the main action type
        self.run_type = ComponentRunType.ACTION_SERVER
        self.action_type = VisionLanguageAction

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

        # Activate component
        super().custom_on_activate()

        # TODO: Create threads/queues for publishing commands and sending inference

    def custom_on_deactivate(self):
        # TODO: Join threads and clear queues

        # Deactivate component
        super().custom_on_deactivate()

    def main_action_callback(self, goal_handle: VisionLanguageAction.Goal):
        """
        Callback for the VLA main action server

        :param goal_handle: Incoming action goal
        :type goal_handle: VisionLanguageAction.Goal

        :return: Action result
        :rtype: VisionLanguageAction.Result
        """

        self._update_state()

        # Get request
        command: str = goal_handle.request.cmd

        # Setup response and feedback of the action
        action_feedback_msg = VisionLanguageAction.Feedback()

        action_result = VisionLanguageAction.Result()

        action_result.success = False

        self.__action_sending_timer = self.create_timer(
            1 / self.config.action_sending_rate,
            self._send_action_commands,
        )

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
                    self.get_logger().info("Goal Canceled")
                    if self.__action_sending_timer is not None:
                        self.destroy_timer(self.__action_sending_timer)
                    return action_result

                # Get new observations from inputs
                model_observations = self._prepare_observations(command)

                if model_observations:
                    # publish feedback
                    result = self._send_observations_for_inference(model_observations)
                    self._update_actions_queue(result)
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
                self.destroy_timer(self.__action_sending_timer)
                self.__action_sending_timer = None
                goal_handle.abort()
                goal_handle.reset()

        # Get the final goal state
        # action_result = ...

        action_result.success = True
        # Publish zero commands to stop the robot

        with self._main_goal_lock:
            self.destroy_timer(self.__action_sending_timer)
            self.__action_sending_timer = None
            goal_handle.succeed()

        return action_result

    def _update_actions_queue(self, new_result):
        """Update actions queue with new result

        :param new_result: New inference result
        """
        pass

    def _prepare_observations(self, command: str):
        """Prepare observations from current inputs

        :param command: Command string
        :return: True if inference input is ready, False otherwise
        """
        # TODO: implement state update from component inputs
        # TODO: implement inference input preparation
        # Return False if inference input cannot be prepared or inputs are not available
        model_observations = {"command": command}
        return model_observations

    def _send_observations_for_inference(self, model_observations):
        """Send observations to model for inference"""
        # TODO: implement sending observations for inference and getting the result
        # return result
        pass

    def _send_action_commands(self):
        """Send action commands"""
        # TODO: implement sending a command to the robot from the commands queue
        pass

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

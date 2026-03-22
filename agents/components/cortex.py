import json
from typing import Optional, Union, List, Dict, Set, cast

from attrs import define, field
from ros_sugar.core.monitor import Monitor

from ..clients.model_base import ModelClient
from ..config import CortexConfig
from ..ros import (
    String,
    Topic,
    Action,
    Event,
    ComponentRunType,
    VisionLanguageAction,
)
from ..utils import validate_func_args, strip_think_tokens
from .model_component import ModelComponent


@define
class CortexAction:
    """Wraps an Action with metadata for LLM tool generation.

    The Cortex component converts these into OpenAI-format tool descriptions
    so the LLM can select and dispatch actions by name. When selected, the
    action is dispatched through the EMOS event system -- not called directly.

    :param action: The Action (or list of Actions) to dispatch when the LLM
        selects this tool.
    :type action: Union[Action, List[Action]]
    :param description: Human-readable description of what this action does.
        This is passed directly to the LLM as the tool description.
    :type description: str

    Example usage:
    ```python
    from agents.ros import Action

    grasp_action = CortexAction(
        action=Action(method=gripper.grasp),
        description="Close the gripper to grasp an object",
    )

    # Multiple actions dispatched together
    alert_action = CortexAction(
        action=[
            Action(method=speaker.say, args=("Alert!",)),
            Action(method=lights.flash),
        ],
        description="Sound an alert and flash the lights",
    )
    ```
    """

    action: Union[Action, List[Action]] = field()
    description: str = field()


class Cortex(ModelComponent, Monitor):
    """
    The Cortex component is an LLM-powered task planner and executor that
    also serves as the system monitor.

    Named after the cerebral cortex, the brain region responsible for
    higher-order planning, reasoning, and action sequencing, this component
    takes a high-level task, uses an LLM to decompose it into sub-tasks,
    and executes them by dispatching Actions registered on other components.

    It follows the ReAct (Reason + Act) pattern: the LLM reasons about what
    to do next, selects an action, observes the result, and repeats until the
    task is complete.

    The component runs as a ROS2 action server, receiving task goals and
    providing feedback during execution.

    :param inputs: Optional input topics for context (e.g., sensor data).
    :type inputs: list[Topic]
    :param outputs: Output topics for publishing task status/results.
    :type outputs: list[Topic]
    :param actions: The action palette -- a list of CortexAction objects
        describing the actions available to the planner.
    :type actions: list[CortexAction]
    :param model_client: The model client for LLM inference.
        Optional if ``enable_local_model`` is set to True in the config.
    :type model_client: Optional[ModelClient]
    :param config: Configuration for the Cortex component.
    :type config: Optional[CortexConfig]
    :param component_name: The name of this component.
    :type component_name: str

    Example usage:
    ```python
    from agents.components import Cortex
    from agents.components.cortex import CortexAction
    from agents.config import CortexConfig
    from agents.ros import Action, Topic, Launcher

    cortex = Cortex(
        inputs=[],
        outputs=[Topic(name="status", msg_type="String")],
        actions=[
            CortexAction(
                action=Action(method=nav.go_to),
                description="Navigate to a location",
            ),
            CortexAction(
                action=Action(method=arm.grasp),
                description="Grasp an object",
            ),
        ],
        model_client=my_client,
        config=CortexConfig(max_iterations=15),
        component_name="cortex",
    )
    ```
    """

    _SYSTEM_PROMPT = (
        "You are a task planning and execution agent on a robot. "
        "You have access to actions that you can call to accomplish tasks. "
        "Break down the given task into steps, execute them one at a time by "
        "calling the appropriate action, observe the result, and continue until "
        "the task is complete. When the task is finished, respond with a brief "
        "summary of what was accomplished. Do not call any actions if the task "
        "is already complete."
    )

    @validate_func_args
    def __init__(
        self,
        *,
        inputs: List[Topic],
        outputs: List[Topic],
        actions: List[CortexAction],
        model_client: Optional[ModelClient] = None,
        config: Optional[CortexConfig] = None,
        component_name: str,
        **kwargs,
    ):
        if not actions:
            raise ValueError(
                "Cortex requires at least one CortexAction. "
                "Provide a list of actions the planner can dispatch."
            )

        self.config: CortexConfig = config or CortexConfig()

        # Enforce config for ReAct loop
        self.config.chat_history = True
        self.config.stream = False
        self.config._system_prompt = self._SYSTEM_PROMPT

        self.allowed_inputs = {
            "Required": [],
            "Optional": [String],
        }
        self.handled_outputs = [String]

        if not model_client and not self.config.enable_local_model:
            raise RuntimeError(
                "Cortex component requires a model_client or "
                "enable_local_model=True in CortexConfig."
            )

        self.model_client = model_client

        # Initialize messages buffer with system prompt
        self.messages: List[Dict] = [{"role": "system", "content": self._SYSTEM_PROMPT}]

        # Behavioral actions: dispatched via event system
        self._action_event_topics: Dict[str, Topic] = {}
        self._action_events: Dict[Event, Union[Action, List[Action]]] = {}
        action_outputs = self._setup_action_events(actions, component_name)

        # System management tools: registered during activation when the
        # Launcher has populated Monitor-side attributes
        self._system_tools: Set[str] = set()

        # Monitor-side: Launcher populates these when it detects Cortex
        self._components_to_monitor: List[str] = []
        self._service_components = None
        self._action_components = None
        self._monitor_events_actions = None
        self._internal_events = None
        self._components_to_activate_on_start: List[str] = []
        self._update_parameter_srv_client: Dict = {}
        self._update_parameters_srv_client: Dict = {}
        self._topic_change_srv_client: Dict = {}
        self._configure_from_file_srv_client: Dict = {}
        self._main_srv_clients: Dict = {}
        self._main_action_clients: Dict = {}

        # Combine user outputs with internal action event topics
        all_outputs = list(outputs) + action_outputs

        # Action server mode
        self.run_type = ComponentRunType.ACTION_SERVER

        if "trigger" in kwargs:
            kwargs.pop("trigger")

        # Initialize via ModelComponent. The MRO will eventually reach
        # Monitor.__init__ which needs components_names — pass it via kwargs.
        ModelComponent.__init__(
            self,
            all_outputs,
            all_outputs,
            model_client,
            self.config,
            None,
            component_name,
            main_action_type=VisionLanguageAction,
            components_names=[],
            **kwargs,
        )

        # Wire event→action pairs after super().__init__
        for event, action in self._action_events.items():
            self._add_event_action_pair(event, action)

    def _setup_action_events(
        self, actions: List[CortexAction], component_name: str
    ) -> List[Topic]:
        """Create internal event topics and tool descriptions for each action.

        :returns: List of internal event topics (to be added as outputs)
        """
        event_topics = []

        for cortex_action in actions:
            targets = (
                cortex_action.action
                if isinstance(cortex_action.action, list)
                else [cortex_action.action]
            )
            action_targets = cast(List[Action], targets)
            name = action_targets[0].action_name

            event_topic = Topic(
                name=f"internal_cortex_event/{component_name}/{name}",
                msg_type="String",
            )
            event_topics.append(event_topic)
            self._action_event_topics[name] = event_topic

            event = Event(event_topic)
            self._action_events[event] = cast(
                Union[Action, List[Action]], cortex_action.action
            )

            tool_description = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": cortex_action.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }
            self.config._tool_descriptions.append(tool_description)
            self.config._tool_response_flags[name] = True

        return event_topics

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def custom_on_configure(self):
        if not self.model_client and self.config.enable_local_model:
            self._deploy_local_model()
        super().custom_on_configure()

    def custom_on_activate(self):
        super().custom_on_activate()

        # Activate Monitor capabilities if Launcher has populated them
        if self._components_to_monitor:
            Monitor.activate(self)
            self._register_system_tools()

    def _deploy_local_model(self):
        """Deploy local LLM model on demand."""
        if self.local_model is not None:
            return
        from ..utils.local_llm import LocalLLM

        self.local_model = LocalLLM(
            model_path=self.config.local_model_path,
            device=self.config.device_local_model,
            ncpu=self.config.ncpu_local_model,
        )

    # =========================================================================
    # System management tools (via Monitor)
    # =========================================================================

    def _register_system_tools(self):
        """Register system management capabilities as LLM tools.

        Called during activation after Monitor.activate() has created
        service clients for all managed components.
        """
        component_names_str = ", ".join(self._components_to_monitor)

        # update_parameter: change a config param on any component
        self._system_tools.add("update_parameter")
        self.config._tool_descriptions.append({
            "type": "function",
            "function": {
                "name": "update_parameter",
                "description": (
                    "Update a configuration parameter on a component. "
                    f"Available components: {component_names_str}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "component": {
                            "type": "string",
                            "description": "Component name",
                        },
                        "param_name": {
                            "type": "string",
                            "description": "Parameter name to update",
                        },
                        "new_value": {
                            "type": "string",
                            "description": "New value for the parameter",
                        },
                    },
                    "required": ["component", "param_name", "new_value"],
                },
            },
        })
        self.config._tool_response_flags["update_parameter"] = True

        # send_goal_to_component: send task to a component's action server
        if self._main_action_clients:
            action_server_names = ", ".join(self._main_action_clients.keys())
            self._system_tools.add("send_goal_to_component")
            self.config._tool_descriptions.append({
                "type": "function",
                "function": {
                    "name": "send_goal_to_component",
                    "description": (
                        "Send a task goal to a component's action server. "
                        f"Components with action servers: {action_server_names}"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "component": {
                                "type": "string",
                                "description": "Component name",
                            },
                            "task": {
                                "type": "string",
                                "description": "Task description to send as the goal",
                            },
                        },
                        "required": ["component", "task"],
                    },
                },
            })
            self.config._tool_response_flags["send_goal_to_component"] = True

    def _execute_system_tool(self, tool_name: str, args: Dict) -> str:
        """Execute a system management tool via inherited Monitor methods."""
        if tool_name == "update_parameter":
            return self._sys_update_parameter(
                args.get("component", ""),
                args.get("param_name", ""),
                args.get("new_value", ""),
            )
        elif tool_name == "send_goal_to_component":
            return self._sys_send_goal(
                args.get("component", ""),
                args.get("task", ""),
            )
        return f"Error: Unknown system tool '{tool_name}'."

    def _sys_update_parameter(
        self, component: str, param_name: str, new_value: str
    ) -> str:
        """Update a parameter on a managed component via ChangeParameter service."""
        client = self._update_parameter_srv_client.get(component)
        if not client:
            return (
                f"Error: Component '{component}' not found. "
                f"Available: {self._components_to_monitor}"
            )
        try:
            from automatika_ros_sugar.srv import ChangeParameter

            req = ChangeParameter.Request()
            req.name = param_name
            req.value = str(new_value)
            req.keep_alive = True
            client.send_request(req_msg=req, executor=self.executor)
            return (
                f"Parameter '{param_name}' on '{component}' updated to '{new_value}'."
            )
        except Exception as e:
            return f"Error updating parameter: {e}"

    def _sys_send_goal(self, component: str, task: str) -> str:
        """Send a task goal to a component's action server."""
        action_client = self._main_action_clients.get(component)
        if not action_client:
            available = list(self._main_action_clients.keys())
            return f"Error: No action server for '{component}'. Available: {available}"
        try:
            goal = VisionLanguageAction.Goal()
            goal.task = task
            action_client.send_request(goal)
            return f"Goal '{task}' sent to '{component}'."
        except Exception as e:
            return f"Error sending goal to '{component}': {e}"

    # =========================================================================
    # Behavioral action dispatch (via event system)
    # =========================================================================

    def _dispatch_action(self, name: str) -> str:
        """Dispatch an action by publishing to its internal event topic."""
        event_topic = self._action_event_topics.get(name)
        if not event_topic:
            available = list(self._action_event_topics.keys())
            return (
                f"Error: Action '{name}' does not exist. Available actions: {available}"
            )
        try:
            self.publishers_dict[event_topic.name].publish(f"cortex_dispatch:{name}")
            return f"Action '{name}' dispatched."
        except Exception as e:
            return f"Error dispatching action '{name}': {e}"

    # =========================================================================
    # ReAct planning loop
    # =========================================================================

    def main_action_callback(self, goal_handle):
        """Action server callback. Runs the ReAct planning loop for a task.

        :param goal_handle: Incoming action goal
        :type goal_handle: VisionLanguageAction.Goal
        :return: Action result
        :rtype: VisionLanguageAction.Result
        """
        task: str = goal_handle.request.task
        self.get_logger().info(f"Received task: {task}")

        feedback_msg = VisionLanguageAction.Feedback()
        result_msg = VisionLanguageAction.Result()
        result_msg.success = False

        # Reset conversation for this task
        self.messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        all_tools = list(self._action_event_topics.keys()) + list(self._system_tools)

        for iteration in range(self.config.max_iterations):
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Task cancelled by client.")
                with self._main_goal_lock:
                    goal_handle.canceled()
                return result_msg

            inference_input = {
                "query": self.messages,
                **self.config._get_inference_params(),
            }
            if self.config._tool_descriptions:
                inference_input["tools"] = self.config._tool_descriptions

            result = self._call_inference(inference_input)
            if not result:
                self.get_logger().error("Inference failed during planning loop.")
                break

            output = result.get("output") or ""
            if self.config.strip_think_tokens:
                output = strip_think_tokens(output)

            # No tool calls — task is complete
            if not result.get("tool_calls"):
                self.messages.append({"role": "assistant", "content": output})
                self._publish({"output": output})

                result_msg.success = True
                feedback_msg.timestep = iteration
                feedback_msg.completed = True
                feedback_msg.feedback = f"Task completed. {output}"
                goal_handle.publish_feedback(feedback_msg)

                self.get_logger().info(f"Task completed in {iteration + 1} iterations.")
                with self._main_goal_lock:
                    goal_handle.succeed()
                return result_msg

            # Has tool calls — dispatch via appropriate mechanism
            self.messages.append({"role": "assistant", "content": output})

            feedback_parts = []
            for tool_call in result["tool_calls"]:
                fn_name = tool_call["function"]["name"]
                fn_args = tool_call["function"].get("arguments", {})

                if fn_name in self._action_event_topics:
                    tool_result = self._dispatch_action(fn_name)
                elif fn_name in self._system_tools:
                    parsed_args = (
                        {
                            key: (json.loads(str(arg)) if isinstance(arg, str) else arg)
                            for key, arg in fn_args.items()
                        }
                        if fn_args
                        else {}
                    )
                    tool_result = self._execute_system_tool(fn_name, parsed_args)
                else:
                    tool_result = (
                        f"Error: Unknown tool '{fn_name}'. Available: {all_tools}"
                    )

                self.messages.append({"role": "tool", "content": tool_result})
                feedback_parts.append(f"{fn_name}: {tool_result}")
                self.get_logger().info(f"[Iteration {iteration + 1}] {tool_result}")

            feedback_msg.timestep = iteration
            feedback_msg.completed = False
            feedback_msg.feedback = " | ".join(feedback_parts)
            goal_handle.publish_feedback(feedback_msg)

        # Max iterations reached
        self.get_logger().warning(
            f"Task did not complete within {self.config.max_iterations} iterations."
        )
        with self._main_goal_lock:
            goal_handle.abort()
        return result_msg

    def _execution_step(self, *args, **kwargs):
        """Not used — Cortex runs as an action server."""
        pass

    def _warmup(self):
        """Warm up and verify model connectivity."""
        inference_input = {
            "query": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": "Hello"},
            ],
            **self.config._get_inference_params(),
        }
        self._call_inference(inference_input)

    def _handle_websocket_streaming(self):
        """Not used — streaming is disabled for Cortex."""
        pass

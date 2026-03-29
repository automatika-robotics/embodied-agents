from copy import copy
import json
import time
from typing import Optional, List, Dict, Set

from ..clients.model_base import ModelClient
from ..clients.db_base import DBClient
from ..config import CortexConfig
from ..ros import (
    String,
    Topic,
    Action,
    Event,
    ComponentRunType,
    VisionLanguageAction,
    Monitor,
    BaseComponent,
    BaseComponentConfig,
    ActionClientHandler,
    ServiceClientHandler,
)
from ..utils import validate_func_args, strip_think_tokens
from ..utils.actions import goal_type_to_json_properties
from .model_component import ModelComponent


class Cortex(ModelComponent, Monitor):
    """
    The Cortex component is an LLM-powered task planner and executor that
    also serves as the system monitor.

    Named after the cerebral cortex, the brain region responsible for
    higher-order planning, reasoning, and action sequencing, this component
    takes a high-level task, uses an LLM to decompose it into sub-tasks,
    and executes them by dispatching Actions registered on other components.

    Task execution follows a two-phase approach:

    1. **Planning** -- A single LLM call with all available actions as tools
       produces a step-by-step plan (returned as multiple tool_calls).
       Optional RAG context from a vector DB is injected during this phase.
    2. **Execution** -- Each planned step is executed sequentially. Before each
       step, a brief LLM confirmation call decides: EXECUTE, SKIP, or ABORT,
       based on the original plan and results so far.

    The component runs as a ROS2 action server, receiving task goals and
    providing feedback during execution.

    :param actions: The action palette -- a list of Action objects with
        descriptions, representing the actions available to the planner.
    :type actions: list[Action]
    :param outputs: Output topics for publishing task status/results.
    :type outputs: list[Topic]
    :param model_client: The model client for LLM inference.
        Optional if ``enable_local_model`` is set to True in the config.
    :type model_client: Optional[ModelClient]
    :param db_client: Optional database client for RAG context during planning.
    :type db_client: Optional[DBClient]
    :param config: Configuration for the Cortex component.
    :type config: Optional[CortexConfig]
    :param component_name: The name of this component.
    :type component_name: str

    Example usage:
    ```python
    from agents.components import Cortex
    from agents.config import CortexConfig
    from agents.ros import Action, Topic, Launcher

    cortex = Cortex(
        outputs=[Topic(name="status", msg_type="String")],
        actions=[
            Action(method=nav.go_to, description="Navigate to a location"),
            Action(method=arm.grasp, description="Grasp an object"),
        ],
        model_client=my_client,
        config=CortexConfig(max_iterations=15),
        component_name="cortex",
    )
    ```
    """

    _PLANNING_PROMPT = (
        "You are a task planning agent on a robot. "
        "Given a task, create a plan by calling the appropriate actions in sequence. "
        "Return ALL actions needed as tool calls in a single response. "
        "Each tool call is one step. Order them in execution sequence. "
        "If the task requires no actions, respond with text only."
    )

    _CONFIRMATION_PROMPT = (
        "You are monitoring task execution on a robot. "
        "Given the original plan and results so far, decide if the next step "
        "should proceed. Respond with exactly one of: EXECUTE, SKIP, or ABORT. "
        "Optionally follow with a brief reason after a colon."
    )

    @validate_func_args
    def __init__(
        self,
        *,
        actions: List[Action],
        outputs: List[Topic],
        model_client: Optional[ModelClient] = None,
        db_client: Optional[DBClient] = None,
        config: Optional[CortexConfig] = None,
        component_name: str,
        **kwargs,
    ):
        self._validate_actions(actions)

        self.config: CortexConfig = config or CortexConfig()

        # Enforce config for planning loop
        self.config.chat_history = True
        self.config.stream = False
        self.config._system_prompt = self._PLANNING_PROMPT

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
        self.db_client = db_client if db_client else None

        # Initialize messages buffer
        self.messages: List[Dict] = [
            {"role": "system", "content": self._PLANNING_PROMPT}
        ]

        # Behavioral actions: dispatched via event system
        self._action_event_topics: Dict[str, Topic] = {}
        self._action_events: Dict[Event, Action] = {}
        action_outputs = self._setup_action_events(actions, component_name)

        # System management and component action tools:
        # registered during activation
        self._system_tools: Set[str] = set()
        self._action_goal_tools: Dict[str, str] = {}

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
        self._main_srv_clients: Dict[str, ServiceClientHandler] = {}
        self._main_action_clients: Dict[str, ActionClientHandler] = {}

        # Combine user outputs with internal action event topics
        all_outputs = list(outputs) + action_outputs

        # Action server mode
        self.run_type = ComponentRunType.ACTION_SERVER

        if "trigger" in kwargs:
            kwargs.pop("trigger")

        if "inputs" in kwargs:
            kwargs.pop("inputs")

        ModelComponent.__init__(
            self,
            inputs=None,
            outputs=all_outputs,
            model_client=model_client,
            config=self.config,
            trigger=None,
            component_name=component_name,
            components_names=[],
            main_action_type=VisionLanguageAction,
            **kwargs,
        )

        # Wire event→action pairs
        for event, action in self._action_events.items():
            self._add_event_action_pair(event, action)

    @staticmethod
    def _validate_actions(actions: List[Action]):
        """Validate that all passed actions have descriptions."""
        if not actions:
            raise ValueError("Cortex must have at least one Action to execute.")
        for action in actions:
            if not action.description:
                raise ValueError(
                    "Each Cortex Action must have a description for the planner. "
                    f"Action '{action.action_name}' is missing a description."
                )

    def _init_internal_monitor(
        self,
        components_names: List[str],
        components: Optional[List[BaseComponent]] = None,
        events_actions: Optional[Dict[str, List[Action]]] = None,
        events_to_emit: Optional[List[Event]] = None,
        config: Optional[BaseComponentConfig] = None,
        services_components: Optional[List[BaseComponent]] = None,
        action_servers_components: Optional[List[BaseComponent]] = None,
        activate_on_start: Optional[List[str]] = None,
        activation_timeout: Optional[float] = None,
        activation_attempt_time: float = 1.0,
        **_,
    ):
        """Initialize Monitor capabilities. Called by the Launcher."""
        # Store component references for introspection by inspect_component
        self._managed_components: Dict[str, BaseComponent] = {}
        if components:
            for comp in components:
                self._managed_components[comp.node_name] = comp

        _config = copy(self.config)
        Monitor.__init__(
            self,
            component_name=self.node_name,
            components_names=components_names,
            events_actions=events_actions,
            events_to_emit=events_to_emit,
            config=config,
            services_components=services_components,
            action_servers_components=action_servers_components,
            activate_on_start=activate_on_start,
            activation_timeout=activation_timeout,
            activation_attempt_time=activation_attempt_time,
        )
        self.config = _config

    def _setup_action_events(
        self, actions: List[Action], component_name: str
    ) -> List[Topic]:
        """Create internal event topics and tool descriptions for each action."""
        event_topics = []

        for cortex_action in actions:
            name = cortex_action.action_name

            event_topic = Topic(
                name=f"internal_cortex_event/{component_name}/{name}",
                msg_type="String",
            )
            event_topics.append(event_topic)
            self._action_event_topics[name] = event_topic

            event = Event(event_topic)
            self._action_events[event] = cortex_action

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
        if self.db_client:
            self.db_client.check_connection()
            self.db_client.initialize()
        super().custom_on_configure()

    def custom_on_activate(self):
        super().custom_on_activate()
        if self._components_to_monitor:
            Monitor.activate(self)
            self._register_system_tools()

    def custom_on_deactivate(self):
        if self.db_client:
            self.db_client.check_connection()
            self.db_client.deinitialize()
        super().custom_on_deactivate()

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
    # RAG
    # =========================================================================

    def _handle_rag_query(self, query: str) -> Optional[str]:
        """Retrieve documents from vector DB for RAG context during planning."""
        if not self.db_client:
            return None
        db_input = {
            "collection_name": self.config.collection_name,
            "query": query,
            "n_results": self.config.n_results,
        }
        result = self.db_client.query(db_input)
        if result:
            return (
                "\n".join(
                    f"{str(meta)}, {doc}"
                    for meta, doc in zip(
                        result["output"]["metadatas"],
                        result["output"]["documents"],
                        strict=True,
                    )
                )
                if self.config.add_metadata
                else "\n".join(doc for doc in result["output"]["documents"])
            )
        return None

    def add_documents(
        self, ids: List[str], metadatas: List[Dict], documents: List[str]
    ) -> None:
        """Add documents to vector DB for RAG context during planning."""
        if not self.db_client:
            raise AttributeError("db_client needs to be set for add_documents to work")
        db_input = {
            "collection_name": self.config.collection_name,
            "distance_func": self.config.distance_func,
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        self.db_client.add(db_input)

    # =========================================================================
    # Phase 1: Planning
    # =========================================================================

    def _plan_task(self, task: str) -> Optional[List[Dict]]:
        """Single LLM call with tools to produce a step-by-step plan.

        :param task: The high-level task description
        :returns: List of tool_call dicts (the plan), or None if no actions needed
        """
        # Optional RAG context
        user_content = task
        if self.config.enable_rag and self.db_client:
            rag_context = self._handle_rag_query(task)
            if rag_context:
                user_content = f"Context:\n{rag_context}\n\nTask: {task}"

        planning_messages = [
            {"role": "system", "content": self._PLANNING_PROMPT},
            {"role": "user", "content": user_content},
        ]

        inference_input = {
            "query": planning_messages,
            **self.config._get_inference_params(),
        }
        if self.config._tool_descriptions:
            inference_input["tools"] = self.config._tool_descriptions

        result = self._call_inference(inference_input)
        if not result:
            self.get_logger().error("Inference failed during planning phase.")
            return None

        output = result.get("output") or ""
        if self.config.strip_think_tokens:
            output = strip_think_tokens(output)

        # Store for use if no tool calls (text-only response)
        self._planning_output = output

        if not result.get("tool_calls"):
            return None

        plan = result["tool_calls"]

        # Truncate if plan exceeds max_iterations
        if len(plan) > self.config.max_iterations:
            self.get_logger().warning(
                f"Plan has {len(plan)} steps, truncating to "
                f"{self.config.max_iterations}."
            )
            plan = plan[: self.config.max_iterations]

        return plan

    # =========================================================================
    # Phase 2: Execution with confirmation
    # =========================================================================

    def _confirm_step(
        self,
        plan: List[Dict],
        executed_results: List[Dict],
        step_index: int,
    ) -> str:
        """Ask the LLM whether the next planned step should be executed.

        :param plan: Full list of planned tool_calls
        :param executed_results: Results of already-executed steps
        :param step_index: Index of the next step to confirm
        :returns: "EXECUTE", "SKIP", or "ABORT"
        """
        # Build plan summary with status annotations
        plan_lines = []
        for i, step in enumerate(plan):
            name = step["function"]["name"]
            args = step["function"].get("arguments", {})
            args_str = f" ({args})" if args else ""

            if i < len(executed_results):
                status = f" [DONE: {executed_results[i]['result']}]"
            elif i == step_index:
                status = " [NEXT]"
            else:
                status = " [PENDING]"
            plan_lines.append(f"  {i + 1}. {name}{args_str}{status}")

        next_step = plan[step_index]
        fn_name = next_step["function"]["name"]
        fn_args = next_step["function"].get("arguments", {})

        user_message = (
            "Original plan:\n"
            + "\n".join(plan_lines)
            + f"\n\nNext action: {fn_name}"
            + (f" with arguments {fn_args}" if fn_args else "")
            + "\n\nRespond EXECUTE, SKIP, or ABORT."
        )

        inference_input = {
            "query": [
                {"role": "system", "content": self._CONFIRMATION_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": self.config.confirmation_temperature,
            "max_new_tokens": self.config.confirmation_max_tokens,
            "stream": False,
        }

        result = self._call_inference(inference_input)
        if not result:
            self.get_logger().warning(
                "Confirmation inference failed; defaulting to EXECUTE."
            )
            return "EXECUTE"

        output = (result.get("output") or "").strip()
        if self.config.strip_think_tokens:
            output = strip_think_tokens(output).strip()

        upper = output.upper()
        if upper.startswith("ABORT"):
            return "ABORT"
        elif upper.startswith("SKIP"):
            return "SKIP"
        return "EXECUTE"

    def _execute_action_step(self, step: Dict) -> str:
        """Execute a single planned step via the appropriate dispatch mechanism."""
        fn_name = step["function"]["name"]
        fn_args = step["function"].get("arguments", {})

        if fn_name in self._action_event_topics:
            return self._dispatch_action(fn_name)
        elif fn_name in self._system_tools:
            parsed_args = (
                {
                    key: (json.loads(str(arg)) if isinstance(arg, str) else arg)
                    for key, arg in fn_args.items()
                }
                if fn_args
                else {}
            )
            return self._execute_system_tool(fn_name, parsed_args)
        else:
            all_tools = list(self._action_event_topics.keys()) + list(
                self._system_tools
            )
            return f"Error: Unknown tool '{fn_name}'. Available: {all_tools}"

    # =========================================================================
    # System management tools (via Monitor)
    # =========================================================================

    def _register_system_tools(self):
        """Register system management capabilities and component actions as LLM tools.

        Called during activation after Monitor.activate() has created service
        clients. Discovers all @component_action and @component_fallback methods
        on managed components and registers them as callable tools.
        """
        component_names_str = ", ".join(self._components_to_monitor)

        # inspect_component: returns text context about a component
        self._system_tools.add("inspect_component")
        self.config._tool_descriptions.append({
            "type": "function",
            "function": {
                "name": "inspect_component",
                "description": (
                    "Get detailed information about a component: its input/output "
                    "topics, available actions, and additional model clients. "
                    "Use this to discover topic names or understand a component "
                    "before calling its actions. "
                    f"Available components: {component_names_str}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "component": {
                            "type": "string",
                            "description": "Component name to inspect.",
                        },
                    },
                    "required": ["component"],
                },
            },
        })
        self.config._tool_response_flags["inspect_component"] = True

        # update_parameter
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

        # Per-component action goal tools
        # Maps tool name -> component name for dispatch
        self._action_goal_tools: Dict[str, str] = {}
        for comp_name, action_client in self._main_action_clients.items():
            name = action_client.config.name.replace("/", "_")
            tool_name = f"send_goal_to_{name}"
            goal_type = self._get_component_action_request_message_type(comp_name)
            if goal_type is None:
                continue

            properties, required = goal_type_to_json_properties(goal_type)
            self._system_tools.add(tool_name)
            self._action_goal_tools[tool_name] = comp_name
            self.config._tool_descriptions.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": (
                        f"Send an action goal to the '{comp_name}' component's "
                        f"action server ({name})."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
            self.config._tool_response_flags[tool_name] = True

        # Discover and register component actions from all managed components
        self._register_component_actions()

    def _register_component_actions(self):
        """Discover @component_action/@component_fallback methods on all
        managed components and register them as callable LLM tools.

        Tool names are namespaced as ``{component_name}.{method_name}``.
        """
        for comp_name, comp in self._managed_components.items():
            for attr_name in dir(comp):
                try:
                    class_attr = getattr(type(comp), attr_name, None)
                    if not class_attr or not hasattr(class_attr, "_action_description"):
                        continue
                    desc_raw = class_attr._action_description
                    if not desc_raw:
                        continue

                    tool_name = f"{comp_name}.{attr_name}"

                    # Check if already registered
                    if tool_name in self._system_tools:
                        continue

                    # Build tool description from OpenAI-format JSON or docstring
                    try:
                        parsed = json.loads(desc_raw)
                        tool_desc = {
                            "type": "function",
                            "function": {
                                **parsed["function"],
                                "name": tool_name,
                            },
                        }
                    except (json.JSONDecodeError, TypeError, KeyError):
                        tool_desc = {
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "description": desc_raw[:200],
                                "parameters": {
                                    "type": "object",
                                    "properties": {},
                                    "required": [],
                                },
                            },
                        }

                    self._system_tools.add(tool_name)
                    self.config._tool_descriptions.append(tool_desc)
                    self.config._tool_response_flags[tool_name] = True
                except Exception:
                    continue

    def _send_action_goal_from_dict(
        self, component_name: str, goal_fields: Dict
    ) -> str:
        """Construct a Goal message from a dict and send it to a component's
        action server.

        :param component_name: Target component name
        :param goal_fields: Dict of goal field values from the LLM
        :returns: Result string for the execution log
        """
        if component_name not in self._main_action_clients:
            return f"Error: Component '{component_name}' has no action client."
        action_client = self._main_action_clients[component_name]
        try:
            sent = action_client.send_request_from_dict(goal_fields)
            if sent is None:
                return (
                    f"Failed to construct or send action goal to "
                    f"'{component_name}' from fields: {goal_fields}"
                )
            if sent:
                # NOTE: Currently execution will be blocked until the task (action) is done. To make it asynchronous: Action acceptance could be returned immediately, then Action feedback check could be implemented to update the LLM with the ongoing status of the action without blocking other tasks in parallel.

                # Block until action is done.
                while not action_client.action_returned:
                    time.sleep(1 / self.config.loop_rate)
                return f"Action for {component_name} completed and returned result {action_client.action_result}."

            return f"Action goal was rejected by '{component_name}'."
        except Exception as e:
            return f"Error sending action goal to '{component_name}': {e}"

    def _execute_system_tool(self, tool_name: str, args: Dict) -> str:
        """Execute a system management tool or a component action."""
        if tool_name == "inspect_component":
            return self._inspect_component(args.get("component", ""))
        elif tool_name == "update_parameter":
            return self.update_parameter(
                args.get("component", ""),
                args.get("param_name", ""),
                args.get("new_value", ""),
            )
        elif tool_name in self._action_goal_tools:
            comp_name = self._action_goal_tools[tool_name]
            return self._send_action_goal_from_dict(comp_name, args)
        # else: the tool is a component action
        return self._call_component_action(tool_name, args)

    def _call_component_action(self, tool_name: str, args: Dict) -> str:
        """Call a component action method directly, as a service."""
        try:
            comp_name, method_name = tool_name.split(".")
            response = self.execute_component_method(comp_name, method_name, args)
            if response.success:
                return f"{tool_name} executed successfully"
            return f"{tool_name} failed with error: {response.error_msg}"
        except Exception as e:
            return f"Error calling {tool_name}: {e}"

    def _inspect_component(self, component_name: str) -> str:
        """Return a text description of a component's structure.

        Provides input/output topics, registered actions (already available
        as tools), and additional model clients. This is context for the LLM
        to use the component's tools correctly (e.g., to discover topic names).
        """
        comp = self._managed_components.get(component_name)
        if not comp:
            available = list(self._managed_components.keys())
            return (
                f"Error: Component '{component_name}' not found. Available: {available}"
            )

        lines = [f"Component: {component_name}", f"Type: {type(comp).__name__}"]

        # Input topics
        if hasattr(comp, "in_topics") and comp.in_topics:
            lines.append("Input topics:")
            for t in comp.in_topics:
                msg_name = (
                    t.msg_type.__name__
                    if hasattr(t.msg_type, "__name__")
                    else t.msg_type
                )
                lines.append(f"  - {t.name} ({msg_name})")
        else:
            lines.append("Input topics: none")

        # Output topics
        if hasattr(comp, "out_topics") and comp.out_topics:
            lines.append("Output topics:")
            for t in comp.out_topics:
                msg_name = (
                    t.msg_type.__name__
                    if hasattr(t.msg_type, "__name__")
                    else t.msg_type
                )
                lines.append(f"  - {t.name} ({msg_name})")
        else:
            lines.append("Output topics: none")

        # List already-registered actions for this component
        prefix = f"{component_name}."
        comp_tools = [
            name for name in self._component_action_callables if name.startswith(prefix)
        ]
        if comp_tools:
            lines.append("Actions (available as tools):")
            for tool_name in comp_tools:
                # Find the tool description
                for td in self.config._tool_descriptions:
                    if td["function"]["name"] == tool_name:
                        fn = td["function"]
                        params = fn.get("parameters", {}).get("properties", {})
                        param_str = ", ".join(
                            f"{k}: {v.get('type', '?')}" for k, v in params.items()
                        )
                        lines.append(
                            f"  - {tool_name}({param_str}): {fn.get('description', '')}"
                        )
                        break
        else:
            lines.append("Actions: none")

        # Additional model clients
        if (
            hasattr(comp, "_additional_model_clients")
            and comp._additional_model_clients
        ):
            lines.append(
                f"Additional model clients: {list(comp._additional_model_clients.keys())}"
            )

        return "\n".join(lines)

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
            self.publishers_dict[event_topic.name].publish(name)
            return f"Action '{name}' dispatched."
        except Exception as e:
            return f"Error dispatching action '{name}': {e}"

    # =========================================================================
    # Main action server callback
    # =========================================================================

    def main_action_callback(self, goal_handle):
        """Action server callback. Runs two-phase planning and execution.

        Phase 1: Single LLM call produces a plan (multiple tool_calls).
        Phase 2: Each step is confirmed then executed sequentially.

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

        # Phase 1: Planning
        plan = self._plan_task(task)

        # Send feedback
        feedback_msg.timestep = 0
        feedback_msg.completed = False
        feedback_msg.feedback = "Creating a plan ..."
        goal_handle.publish_feedback(feedback_msg)
        self.get_logger().info("Creating a plan ...")

        if plan is None:
            text_output = getattr(self, "_planning_output", "") or ""
            if text_output:
                self._publish({"output": text_output})
                result_msg.success = True
                feedback_msg.timestep = 0
                feedback_msg.completed = True
                feedback_msg.feedback = f"No actions needed. {text_output}"
                goal_handle.publish_feedback(feedback_msg)
                with self._main_goal_lock:
                    goal_handle.succeed()
            else:
                feedback_msg.timestep = 0
                feedback_msg.completed = True
                feedback_msg.feedback = "Planning failed: no response from model."
                goal_handle.publish_feedback(feedback_msg)
                with self._main_goal_lock:
                    goal_handle.abort()
            return result_msg

        # Publish planning feedback
        plan_description = ", ".join(step["function"]["name"] for step in plan)
        feedback_msg.timestep = 0
        feedback_msg.completed = False
        feedback_msg.feedback = (
            f"Plan created with {len(plan)} steps: {plan_description}"
        )
        goal_handle.publish_feedback(feedback_msg)
        self.get_logger().info(f"Plan: {plan_description}")

        # Phase 2: Execution
        executed_results: List[Dict] = []
        aborted = False

        for i, step in enumerate(plan):
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Task cancelled by client.")
                with self._main_goal_lock:
                    goal_handle.canceled()
                return result_msg

            fn_name = step["function"]["name"]

            # Confirm step with LLM
            decision = self._confirm_step(plan, executed_results, i)
            self.get_logger().info(
                f"[Step {i + 1}/{len(plan)}] {fn_name} -> {decision}"
            )

            if decision == "ABORT":
                feedback_msg.timestep = i + 1
                feedback_msg.completed = False
                feedback_msg.feedback = f"Plan aborted at step {i + 1} ({fn_name})."
                goal_handle.publish_feedback(feedback_msg)
                aborted = True
                break

            if decision == "SKIP":
                executed_results.append({
                    "step": i,
                    "action": fn_name,
                    "result": "SKIPPED",
                })
                feedback_msg.timestep = i + 1
                feedback_msg.completed = False
                feedback_msg.feedback = f"Step {i + 1} ({fn_name}): skipped."
                goal_handle.publish_feedback(feedback_msg)
                continue

            # EXECUTE action step
            step_result = self._execute_action_step(step)
            executed_results.append({
                "step": i,
                "action": fn_name,
                "result": step_result,
            })

            feedback_msg.timestep = i + 1
            feedback_msg.completed = False
            feedback_msg.feedback = f"Step {i + 1} ({fn_name}): {step_result}"
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f"[Step {i + 1}] {step_result}")

        # Final result handling
        summary = "; ".join(f"{r['action']}: {r['result']}" for r in executed_results)

        if aborted:
            self._publish({"output": f"Plan aborted. Completed: {summary}"})
            with self._main_goal_lock:
                goal_handle.abort()
        else:
            self._publish({"output": f"Plan completed. {summary}"})
            result_msg.success = True

            feedback_msg.timestep = len(plan)
            feedback_msg.completed = True
            feedback_msg.feedback = f"All {len(plan)} steps completed."
            goal_handle.publish_feedback(feedback_msg)

            self.get_logger().info(
                f"Task completed: {len(executed_results)} steps executed."
            )
            with self._main_goal_lock:
                goal_handle.succeed()

        return result_msg

    # =========================================================================
    # Unused/overridden methods (action server mode)
    # =========================================================================

    def _create_input(self, *args, **kwargs) -> Optional[Dict]:
        """Not used -- Cortex builds inputs in _plan_task and _confirm_step."""
        return None

    def _execution_step(self, *args, **kwargs):
        """Not used -- Cortex runs as an action server."""
        pass

    def _warmup(self):
        """Warm up and verify model connectivity."""
        self._call_inference({
            "query": [
                {"role": "system", "content": self._PLANNING_PROMPT},
                {"role": "user", "content": "Hello"},
            ],
            **self.config._get_inference_params(),
        })

    def _handle_websocket_streaming(self):
        """Not used -- streaming is disabled for Cortex."""
        pass

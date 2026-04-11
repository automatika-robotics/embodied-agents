from copy import copy
import json
import os
import time
from typing import Optional, List, Dict, Set, Any, Tuple

from ..clients.model_base import ModelClient
from ..clients.db_base import DBClient
from ..config import CortexConfig
from ..ros import (
    String,
    StreamingString,
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
    ros_msg_to_str,
    get_methods_with_decorator,
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
    :param output: Output topic for publishing results for tasks where an action is not required or a plan is not generated.
    :type output: Topic
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
        actions=[
            Action(method=nav.go_to, description="Navigate to a location"),
            Action(method=arm.grasp, description="Grasp an object"),
        ],
        model_client=my_client,
        config=CortexConfig(max_planning_steps=10, max_execution_steps=15),
        component_name="cortex",
    )
    ```
    """

    _PLANNING_PROMPT = (
        "You are a task planning agent on a robot. "
        "Given a task, first use inspect_component to research the available "
        "components and discover their capabilities, topics, and actions. "
        "Once you have enough information, create a plan by calling the "
        "appropriate actions in sequence. "
        "Return ALL actions needed as tool calls in a single response. "
        "Each tool call is one step. Order them in execution sequence. "
        "If the task requires no actions, respond with text only."
    )

    _CONFIRMATION_PROMPT = (
        "You are monitoring task execution on a robot. "
        "Given the original plan and results so far, decide what to do next. "
        "Respond with exactly one of:\n"
        "  EXECUTE - proceed with the next step\n"
        "  SKIP - skip the next step\n"
        "  ABORT - abort the entire plan\n"
        "  CONTINUE - wait for ongoing async actions to complete before proceeding\n"
        "Use CONTINUE when there are active async actions that should finish "
        "before moving on. Optionally follow with a brief reason after a colon."
    )

    @validate_func_args
    def __init__(
        self,
        *,
        actions: Optional[List[Action]] = None,
        output: Optional[Topic] = None,
        model_client: Optional[ModelClient] = None,
        db_client: Optional[DBClient] = None,
        config: Optional[CortexConfig] = None,
        component_name: str,
        **kwargs,
    ):
        self.handled_outputs = [String, StreamingString]
        self._validate_actions(actions)

        self.config: CortexConfig = config or CortexConfig()

        # Enforce config for planning loop
        self.config.chat_history = True
        self.config.stream = False
        self.config._system_prompt = self._PLANNING_PROMPT

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

        # Tool registries — separated into planning and execution phases.
        # Planning tools (e.g. inspect_component) gather information.
        # Execution tools (actions, system tools) are what the plan consists of.
        self._planning_tools: Set = set()
        self._planning_tool_descriptions: List[Dict] = []
        self._execution_tools: Set = set()
        self._execution_tool_descriptions: List[Dict] = []

        # Action server goals and Service request tools
        self._action_goal_tools: Dict[str, Tuple[str, str, Any]] = {}
        self._service_request_tools: Dict[str, Tuple[str, str, Any]] = {}

        # Behavioral actions: dispatched via internal event system
        self._pure_internal_events = []
        self._additional_internal_actions = {}
        self._setup_internal_action_events(actions)

        # Planning output buffer for failed plans
        self._planning_output: Optional[str] = None

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
        self._active_action_clients: Dict[
            str, ActionClientHandler
        ] = {}  # Register action clients with active ongoing goals to manage feedback and request cancellation

        # Action server mode
        self.run_type = ComponentRunType.ACTION_SERVER

        for kwarg in ["inputs", "trigger", "outputs"]:
            if kwarg in kwargs:
                kwargs.pop(kwarg)

        ModelComponent.__init__(
            self,
            inputs=None,
            outputs=[output] if output else None,
            model_client=model_client,
            config=self.config,
            trigger=None,
            component_name=f"{component_name}_{os.getpid()}",
            components_names=[],
            main_action_type=VisionLanguageAction,
            **kwargs,
        )

        # set the cortex action name
        self.main_action_name = "cortex_input_command"

    @staticmethod
    def _validate_actions(actions: Optional[List[Action]]):
        """Validate that all passed actions have descriptions."""
        if not actions:
            return
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
        events_actions: Optional[Dict[Event, List[Action]]] = None,
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

    def _setup_internal_action_events(self, actions: Optional[List[Action]]) -> None:
        """Create internal event topics and tool descriptions for each action."""
        if not actions:
            return
        for cortex_action in actions:
            name = cortex_action.action_name
            Monitor.add_internal_event_action_pair(
                self, event_id=name, action=cortex_action
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
            self._execution_tools.add(name)
            self._execution_tool_descriptions.append(tool_description)

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

        # Display all the tools registered
        planning_names = [
            t["function"]["name"] for t in self._planning_tool_descriptions
        ]
        execution_names = [
            t["function"]["name"] for t in self._execution_tool_descriptions
        ]
        self.get_logger().debug(f"Cortex planning tools: {planning_names}")
        self.get_logger().debug(f"Cortex execution tools: {execution_names}")

    def custom_on_deactivate(self):
        if self.db_client:
            self.db_client.check_connection()
            self.db_client.deinitialize()
        # cancel any active action clients held by cortex
        self._cancel_all_active_clients()
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
    # Helpers
    # =========================================================================

    def _parse_tool_args(self, fn_args: Dict) -> Dict:
        """Parse tool arguments, deserializing JSON strings where needed."""
        parsed_args = {}
        for key, arg in fn_args.items():
            if isinstance(arg, str):
                arg_str = arg.strip()
                if not arg_str:
                    parsed_args[key] = ""
                    continue
                try:
                    parsed_args[key] = json.loads(arg_str)
                except json.JSONDecodeError:
                    parsed_args[key] = arg_str
            else:
                parsed_args[key] = arg
        return parsed_args

    def __register_action_client_as_tool(
        self, component_name: str, action_name: str, action_type: Any
    ) -> None:
        """Helper method to register a component Action Server as a system tool

        :param component_name: Component name
        :type component_name: str
        :param action_name: Action server name
        :type action_name: str
        :param action_type: Action server type
        :type action_type: Any
        """
        name = action_name.replace("/", "_")
        tool_name = f"send_goal_to_{name}"
        goal_type = action_type.Goal
        properties, required = goal_type_to_json_properties(goal_type)
        self._execution_tools.add(tool_name)
        self._action_goal_tools[tool_name] = (component_name, action_name, action_type)
        self._execution_tool_descriptions.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": (
                    f"Send an action goal to the '{component_name}' component's "
                    f"action server ({name})."
                ),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })

    def __register_service_client_as_tool(
        self, component_name: str, srv_name: str, srv_type: Any
    ) -> None:
        """Helper method to register a component Service as a system tool

        :param component_name: Component name
        :type component_name: str
        :param srv_name: Server name
        :type srv_name: str
        :param srv_type: Server type
        :type srv_type: Any
        """
        name = srv_name.replace("/", "_")
        tool_name = f"send_request_to_{name}"
        req_type = srv_type.Request
        properties, required = goal_type_to_json_properties(req_type)
        self._execution_tools.add(tool_name)
        self._service_request_tools[tool_name] = (component_name, srv_name, srv_type)
        self._execution_tool_descriptions.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": (
                    f"Send a service request to the '{component_name}' component's "
                    f"server ({name})."
                ),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })

    # =========================================================================
    # Phase 1: Planning (multi-step loop)
    # =========================================================================

    def _build_planning_messages(self, task: str) -> List[Dict]:
        """Build the initial message list for the planning loop.

        Injects optional RAG context from the vector DB.
        """
        user_content = task
        if self.config.enable_rag and self.db_client:
            rag_context = self._handle_rag_query(task)
            if rag_context:
                user_content = f"Context:\n{rag_context}\n\nTask: {task}"

        return [
            {"role": "system", "content": self._PLANNING_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _process_planning_calls(
        self,
        planning_calls: List[Dict],
        messages: List[Dict],
        output: str,
        step: int,
    ) -> None:
        """Execute planning tool calls and append results to the message history."""
        assistant_msg = {
            "role": "assistant",
            "content": output,
            "tool_calls": [
                {
                    "id": f"plan_{step}_{i}",
                    "type": "function",
                    "function": tc["function"],
                }
                for i, tc in enumerate(planning_calls)
            ],
        }
        messages.append(assistant_msg)

        for i, tc in enumerate(planning_calls):
            fn_name = tc["function"]["name"]
            fn_args = self._parse_tool_args(tc["function"].get("arguments", {}))
            tool_result = self._execute_planning_tool(fn_name, fn_args)
            self.get_logger().debug(
                f"[Planning step {step + 1}] {fn_name} -> {tool_result[:200]}"
            )
            messages.append({
                "role": "tool",
                "tool_call_id": f"plan_{step}_{i}",
                "content": tool_result,
            })

    def _finalize_plan(self, execution_calls: List[Dict]) -> List[Dict]:
        """Truncate the execution plan to max_execution_steps if needed."""
        plan = execution_calls
        if len(plan) > self.config.max_execution_steps:
            self.get_logger().warning(
                f"Plan has {len(plan)} steps, truncating to "
                f"{self.config.max_execution_steps}."
            )
            plan = plan[: self.config.max_execution_steps]
        self.get_logger().info(f"Got plan: {plan}")
        return plan

    def _plan_task(self, task: str) -> Optional[List[Dict]]:
        """Multi-step planning loop that researches components before producing a plan.

        The LLM is given both planning tools and execution tools.
        Each iteration it may:

        - Call planning tools to gather information (results are fed back).
        - Call execution tools — these become the plan and end the loop.
        - Respond with text only — no actions needed, loop ends.

        :param task: The high-level task description
        :returns: List of tool_call dicts (the plan), or None if no actions needed
        """
        all_tools = self._planning_tool_descriptions + self._execution_tool_descriptions
        messages = self._build_planning_messages(task)
        output = ""

        for step in range(self.config.max_planning_steps):
            inference_input = {
                "query": messages,
                **self.config._get_inference_params(),
            }
            if all_tools:
                inference_input["tools"] = all_tools

            result = self._call_inference(inference_input)
            if not result:
                self.get_logger().error(
                    f"Inference failed during planning step {step + 1}."
                )
                return None

            output = result.get("output") or ""
            if self.config.strip_think_tokens:
                output = strip_think_tokens(output)

            tool_calls = result.get("tool_calls")
            if not tool_calls:
                self._planning_output = output
                return None

            # Separate planning tool calls from execution tool calls
            planning_calls = [
                tc
                for tc in tool_calls
                if tc["function"]["name"] in self._planning_tools
            ]
            execution_calls = [
                tc
                for tc in tool_calls
                if tc["function"]["name"] not in self._planning_tools
            ]
            if planning_calls:
                self._process_planning_calls(planning_calls, messages, output, step)

            if execution_calls:
                return self._finalize_plan(execution_calls)

        self.get_logger().warning(
            f"Planning reached max steps ({self.config.max_planning_steps}) "
            "without producing an execution plan."
        )
        self._planning_output = output
        return None

    def _execute_planning_tool(self, tool_name: str, args: Dict) -> str:
        """Execute a planning-phase tool (e.g. inspect_component)."""
        if tool_name == "inspect_component":
            return self._inspect_component(args.get("component", ""))
        return f"Error: Unknown planning tool '{tool_name}'."

    # =========================================================================
    # Phase 2: Execution with confirmation
    # =========================================================================

    def _build_confirmation_message(
        self,
        plan: List[Dict],
        executed_results: List[Dict],
        step_index: int,
    ) -> str:
        """Build the user message for a confirmation LLM call.

        Includes the plan with status annotations and active async action
        status if any actions are currently running.
        """
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

        message = "Original plan:\n" + "\n".join(plan_lines)

        if step_index < len(plan):
            next_step = plan[step_index]
            fn_name = next_step["function"]["name"]
            fn_args = next_step["function"].get("arguments", {})

            message += f"\n\nNext action: {fn_name}" + (
                f" with arguments {fn_args}" if fn_args else ""
            )

        # Include active async action status if any
        active_status = self._monitor_active_clients()
        if active_status:
            message += f"\n\n{active_status}"

        message += "\n\nRespond EXECUTE, SKIP, ABORT, or CONTINUE."
        return message

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
        :returns: "EXECUTE", "SKIP", "ABORT", or "CONTINUE"
        """
        user_message = self._build_confirmation_message(
            plan, executed_results, step_index
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
        self.get_logger().debug(f"Got confirm step result = {result}")
        if not result:
            self.get_logger().warning(
                "Confirmation inference failed; defaulting to EXECUTE."
            )
            return "EXECUTE"

        output = (result.get("output") or "").strip()
        if self.config.strip_think_tokens:
            output = strip_think_tokens(output).strip()

        upper = output.upper()
        for token in ("ABORT", "SKIP", "CONTINUE", "EXECUTE"):
            if upper.startswith(token):
                return token
        return "EXECUTE"

    def _execute_action_step(self, step: Dict) -> str:
        """Execute a single planned step via the appropriate dispatch mechanism."""
        fn_name = step["function"]["name"]
        fn_args = step["function"].get("arguments", {})

        if fn_name in self.emit_internal_event_methods:
            return self._dispatch_action(fn_name)
        elif fn_name in self._execution_tools:
            parsed_args = self._parse_tool_args(fn_args)
            return self._execute_system_tool(fn_name, parsed_args)
        else:
            all_tools = list(self.emit_internal_event_methods.keys()) + list(
                self._execution_tools
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

        Tools are separated into two categories:
        - **Planning tools** (``inspect_component``): used during the planning
          loop to research components before building a plan.
        - **Execution tools** (``update_parameter``, ``send_goal_to_*``,
          component actions): used in the execution plan.
        """
        component_names_str = ", ".join(self._components_to_monitor)

        # inspect_component: planning-only tool
        inspect_desc = {
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
        }
        self._planning_tools.add("inspect_component")
        self._planning_tool_descriptions.append(inspect_desc)

        # update_parameter: execution tool
        update_param_desc = {
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
        }
        self._execution_tools.add("update_parameter")
        self._execution_tool_descriptions.append(update_param_desc)

        # Per-component action goal tools: execution tools
        for comp_name, action_client in self._main_action_clients.items():
            self.__register_action_client_as_tool(
                component_name=comp_name,
                action_name=action_client.config.name,
                action_type=action_client.config.action_type,
            )

        # Discover and register component actions from all managed components
        self._register_component_actions()

    # Lifecycle management methods that should not be exposed as LLM tools.
    # These are handled by the Monitor / Launcher
    _LIFECYCLE_METHODS = frozenset({
        "start",
        "stop",
        "restart",
        "reconfigure",
        "set_param",
        "set_params",
        "broadcast_status",
    })

    def _register_component_actions(self):
        """Discover @component_action/@component_fallback methods on all
        managed components and register them as callable LLM tools.

        Uses ``get_methods_with_decorator`` from sugarcoat to find decorated
        methods. Tool names are namespaced as ``{component_name}.{method_name}``.
        These are execution-phase tools. Lifecycle management methods
        (start, stop, restart, etc.) are excluded.
        """
        for comp_name, comp in self._managed_components.items():
            # Collect all action/fallback method names
            action_methods = get_methods_with_decorator(comp, "component_action")
            fallback_methods = get_methods_with_decorator(comp, "component_fallback")

            for attr_name in action_methods + fallback_methods:
                if attr_name in self._LIFECYCLE_METHODS:
                    continue

                try:
                    tool_name = f"{comp_name}.{attr_name}"
                    # Check if already registered
                    if tool_name in self._execution_tools:
                        continue

                    # Get the description from the decorator attribute
                    class_attr = getattr(type(comp), attr_name, None)
                    desc_raw = (
                        getattr(class_attr, "_action_description", None)
                        if class_attr
                        else None
                    )

                    # Not the decorator we were looking for
                    if not desc_raw:
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

                    self._execution_tools.add(tool_name)
                    self._execution_tool_descriptions.append(tool_desc)
                except Exception:
                    continue
            # Register component additional entry points (Actions & Services)
            entrypoints: Dict[str, Dict] = comp.get_ros_entrypoints()
            # Register additional services as tools
            srv_entrypoints = entrypoints.get("services", {})
            for srv_name, srv_type in srv_entrypoints.items():
                self.__register_service_client_as_tool(
                    component_name=comp_name, srv_name=srv_name, srv_type=srv_type
                )

            # Register additional Action Servers as tools
            actions_entrypoints = entrypoints.get("actions", {})
            for action_name, action_type in actions_entrypoints.items():
                self.__register_action_client_as_tool(
                    component_name=comp_name,
                    action_name=action_name,
                    action_type=action_type,
                )

    def _send_action_goal_from_dict(
        self,
        tool_name: str,
        component_name: str,
        action_name: str,
        action_type: Any,
        goal_fields: Dict,
    ) -> str:
        """Construct a Goal message from a dict and send it to a component's
        action server.

        :param component_name: Target component name
        :type component_name: str
        :param action_name: Target action server name
        :type action_name: str
        :param action_type: Target action server type
        :type action_type: Any
        :param goal_fields:  Dict of goal field values from the LLM
        :type goal_fields: Dict
        :return: Result string for the execution log
        :rtype: str
        """
        action_client = self._get_action_client(action_name, action_type)
        try:
            sent = action_client.send_request_from_dict(goal_fields)
            if not sent:
                return (
                    f"Error: Failed to construct or send action goal to "
                    f"'{component_name}' from fields: {goal_fields}"
                )
            if sent:
                self._active_action_clients[tool_name] = action_client
                return (
                    f"Action '{tool_name}' has been dispatched to '{component_name}' "
                    f"and is now running asynchronously."
                )
            return f"Error: Action goal was rejected by '{component_name}'."
        except Exception as e:
            return f"Error sending action goal to '{component_name}': {e}"

    def _send_service_request_from_dict(
        self, component_name: str, srv_name: str, srv_type: Any, req_fields: Dict
    ) -> str:
        """Construct a Request message from a dict and send it to a component's
        service.

        :param component_name: Target component name
        :type component_name: str
        :param srv_name: Target server name
        :type srv_name: str
        :param srv_type: Target server type
        :type srv_type: Any
        :param req_fields: Dict of request field values from the LLM
        :type req_fields: Dict
        :return: Result string for the execution log
        :rtype: str
        """
        srv_client: ServiceClientHandler = self._get_srv_client(srv_name, srv_type)
        try:
            result = srv_client.send_request_from_dict(req_fields)
            if result is None:
                return (
                    f"Error: Failed to construct or send service request to "
                    f"'{component_name}' service '{srv_name}' from fields: {req_fields}"
                )
            return f"Service {srv_name} for {component_name} completed and returned result {result}."

        except Exception as e:
            return f"Error sending service request to '{component_name}' service {srv_name}: {e}"

    def _execute_system_tool(self, tool_name: str, args: Dict) -> str:
        """Execute an execution-phase system tool or a component action."""
        try:
            if tool_name == "update_parameter":
                self.get_logger().info(
                    f"Calling component action {tool_name} with args: {args}"
                )
                response = self.update_parameter(
                    args.get("component", ""),
                    args.get("param_name", ""),
                    args.get("new_value", ""),
                )
                if response.success:
                    return f"{tool_name} executed successfully"
                return f"Error: {tool_name} failed with error: {response.error_msg}"
            elif tool_name in self._action_goal_tools:
                comp_name, action_name, action_type = self._action_goal_tools[tool_name]
                return self._send_action_goal_from_dict(
                    tool_name, comp_name, action_name, action_type, args
                )
            elif tool_name in self._service_request_tools:
                comp_name, srv_name, srv_type = self._service_request_tools[tool_name]
                return self._send_service_request_from_dict(
                    comp_name, srv_name, srv_type, args
                )
            # else: the tool is a component action
            return self._call_component_action(tool_name, args)
        except Exception as e:
            return f"Error calling {tool_name}: {e}"

    def _call_component_action(self, tool_name: str, args: Dict) -> str:
        """Call a component action method directly, as a service."""
        self.get_logger().info(
            f"Calling component action {tool_name} with args: {args}"
        )
        try:
            comp_name, method_name = tool_name.split(".")
            response = self.execute_component_method(comp_name, method_name, args)
            if response.success:
                return f"{tool_name} executed successfully"
            return f"Error: {tool_name} failed with error: {response.error_msg}"
        except Exception as e:
            return f"Error: Could not parse tool name for {tool_name}. Failed with error: {e}"

    def _inspect_component(self, component_name: str) -> str:
        """Return a text description of a component's structure.

        Delegates to the component's own ``inspect_component()`` for base
        info (inputs, outputs, config, additional model clients), then
        appends the Cortex-registered execution tools for that component.
        """
        comp = self._managed_components.get(component_name)
        if not comp:
            available = list(self._managed_components.keys())
            return (
                f"Error: Component '{component_name}' not found. Available: {available}"
            )

        result = comp.inspect_component()

        # Append Cortex-registered execution tools for this component
        prefix = f"{component_name}."
        comp_tools = [name for name in self._execution_tools if name.startswith(prefix)]
        if comp_tools:
            lines = ["Actions (available as tools):"]
            for tool_name in comp_tools:
                for td in self._execution_tool_descriptions:
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
            result += "\n" + "\n".join(lines)
        else:
            result += "\nActions: none"

        return result

    # =========================================================================
    # Behavioral action dispatch (via event system)
    # =========================================================================

    def _dispatch_action(self, name: str) -> str:
        """Dispatch an action by publishing to its internal event topic."""
        dispatch_method = self.emit_internal_event_methods.get(name, None)
        if not dispatch_method:
            available = list(self.emit_internal_event_methods.keys())
            return (
                f"Error: Action '{name}' does not exist. Available actions: {available}"
            )
        try:
            dispatch_method()
            return f"Action '{name}' dispatched."
        except Exception as e:
            return f"Error dispatching action '{name}': {e}"

    # =========================================================================
    # Action client helpers
    # =========================================================================

    def _monitor_active_clients(self) -> Optional[str]:
        """Helper method to get a status update on the ongoing Action clients

        :return: Active tools status feedback
        :rtype: Optional[str]
        """
        if not self._active_action_clients:
            # Not active clients -> Nothing to monitor
            return None
        completed_actions = []
        feedback_lines = "[Active Tools Status]\n"
        for tool_name, action_client in self._active_action_clients.items():
            if action_client.action_returned:
                # Action is done and returned result
                result = action_client.action_result
                status = action_client._status
                completed_actions.append(tool_name)
                feedback_lines += (
                    f"- {tool_name}: {status.upper()} | Result: {result}\n"
                )
                continue
            updates_dict = action_client.get_ui_elements()
            feedback_lines += f"- {tool_name}: {updates_dict['status']} (running for {updates_dict['duration_secs']}s)"
            if updates_dict["feedback"]:
                feedback_lines += (
                    f" | Latest feedback: {ros_msg_to_str(updates_dict['feedback'])}"
                )
            if updates_dict["feedback_timeout"]:
                feedback_lines += " [WARNING: No new feedback received — the tool may be stalled or waiting on an external process.]"
            feedback_lines += "\n"
        feedback_lines += "[End Of Tools Status Update]\n"
        # Remove completed actions from the active clients registry
        for tool in completed_actions:
            self._active_action_clients.pop(tool)
        return feedback_lines

    def _cancel_all_active_clients(self):
        """Helper method to cancel all action Action clients

        :return: Cancellation error message if errors occurred
        :rtype: Optional[str]
        """
        if not self._active_action_clients:
            # Not active clients to cancel
            return
        successful_cancellation = []
        for tool_name, action_client in self._active_action_clients.items():
            cancelled, _ = action_client.cancel_request()
            if not cancelled:
                self.get_logger().error(
                    f"Error: Failed to cancel the following ongoing tool: {tool_name}"
                )
            else:
                successful_cancellation.append(tool_name)
        for key in successful_cancellation:
            self._active_action_clients.pop(key)

    # =========================================================================
    # Main action server callback
    # =========================================================================

    def _send_feedback(
        self,
        goal_handle,
        feedback_msg,
        timestep: int,
        text: str,
        completed: bool = False,
    ) -> None:
        """Publish feedback on the action server."""
        feedback_msg.timestep = timestep
        feedback_msg.completed = completed
        feedback_msg.feedback = text
        goal_handle.publish_feedback(feedback_msg)

    def _wait_for_active_clients(
        self, goal_handle, feedback_msg, plan, executed_results, step_index, label: str
    ) -> str:
        """Poll active async actions until the LLM stops returning CONTINUE.

        :returns: Final decision (EXECUTE, SKIP, or ABORT)
        """
        decision = self._confirm_step(plan, executed_results, step_index)
        while decision == "CONTINUE":
            if goal_handle.is_cancel_requested:
                return "ABORT"
            self._send_feedback(
                goal_handle,
                feedback_msg,
                step_index,
                f"{label}: waiting for async actions to complete...",
            )
            time.sleep(self.config.monitoring_interval)
            decision = self._confirm_step(plan, executed_results, step_index)
            self.get_logger().info(f"[{label}] re-check -> {decision}")
        return decision

    def _execute_plan(self, plan, goal_handle, feedback_msg) -> Tuple[List[Dict], bool]:
        """Execute plan steps with per-step confirmation.

        :returns: (executed_results, aborted)
        """
        executed_results: List[Dict] = []
        total = len(plan)

        for i, step in enumerate(plan):
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Task cancelled by client.")
                with self._main_goal_lock:
                    self._cancel_all_active_clients()
                    goal_handle.canceled()
                return executed_results, True

            fn_name = step["function"]["name"]
            fn_args = step["function"].get("arguments", {})
            label = f"Step {i + 1}/{total} ({fn_name})"

            # Confirm (may wait for async actions via CONTINUE)
            decision = self._wait_for_active_clients(
                goal_handle, feedback_msg, plan, executed_results, i, label
            )
            self.get_logger().info(f"[{label}] -> {decision}")

            if decision == "ABORT":
                self._send_feedback(
                    goal_handle, feedback_msg, i + 1, f"Plan aborted at {label}."
                )
                return executed_results, True

            if decision == "SKIP":
                executed_results.append({
                    "step": i,
                    "action": fn_name,
                    "result": "SKIPPED",
                })
                self._send_feedback(
                    goal_handle, feedback_msg, i + 1, f"{label}: skipped."
                )
                continue

            # EXECUTE
            args_str = f" with {fn_args}" if fn_args else ""
            self._send_feedback(
                goal_handle, feedback_msg, i + 1, f"Executing {label}{args_str}"
            )

            step_result = self._execute_action_step(step)
            executed_results.append({
                "step": i,
                "action": fn_name,
                "result": step_result,
                "failed": step_result.startswith("Error"),
            })
            self._send_feedback(
                goal_handle, feedback_msg, i + 1, f"{label} completed: {step_result}"
            )
            self.get_logger().info(f"[{label}] {step_result}")

        # Wait for any remaining async actions after the last step
        if self._monitor_active_clients():
            decision = self._wait_for_active_clients(
                goal_handle,
                feedback_msg,
                plan,
                executed_results,
                len(plan),
                "Post-execution",
            )
            if decision == "ABORT":
                self._send_feedback(
                    goal_handle,
                    feedback_msg,
                    total,
                    "Plan aborted while waiting for async actions.",
                )
                return executed_results, True

        return executed_results, False

    def _finalize_goal(
        self, goal_handle, feedback_msg, result_msg, executed_results, plan_len, aborted
    ) -> None:
        """Set the final goal status based on execution results."""
        summary = "; ".join(f"{r['action']}: {r['result']}" for r in executed_results)
        has_failures = any(r.get("failed") for r in executed_results)

        if aborted:
            with self._main_goal_lock:
                self._cancel_all_active_clients()
                goal_handle.abort()
        elif has_failures:
            self._send_feedback(
                goal_handle,
                feedback_msg,
                plan_len,
                f"Plan finished with errors. {summary}",
                completed=True,
            )
            self.get_logger().warning(f"Task finished with errors: {summary}")
            with self._main_goal_lock:
                self._cancel_all_active_clients()
                goal_handle.abort()
        else:
            result_msg.success = True
            self._send_feedback(
                goal_handle,
                feedback_msg,
                plan_len,
                f"All {plan_len} steps completed.",
                completed=True,
            )
            self.get_logger().info(
                f"Task completed: {len(executed_results)} steps executed."
            )
            with self._main_goal_lock:
                self._cancel_all_active_clients()
                goal_handle.succeed()

    def main_action_callback(self, goal_handle):
        """Action server callback. Runs two-phase planning and execution.

        :param goal_handle: Incoming action goal
        :return: Action result
        """
        task: str = goal_handle.request.task
        self.get_logger().info(f"Received task: {task}")

        feedback_msg = VisionLanguageAction.Feedback()
        result_msg = VisionLanguageAction.Result()
        result_msg.success = False

        self._send_feedback(
            goal_handle, feedback_msg, 0, f"Received task. Creating a plan for: {task}"
        )

        # Phase 1: Planning
        plan = self._plan_task(task)

        if plan is None:
            text_output = self._planning_output or ""
            if text_output:
                result_msg.success = True
                self._send_feedback(
                    goal_handle,
                    feedback_msg,
                    0,
                    f"[No actions needed]. {text_output}",
                    completed=True,
                )
                # publish planning output if there is an output topic
                self._publish(result={"output": text_output})
                with self._main_goal_lock:
                    goal_handle.succeed()
            else:
                self._send_feedback(
                    goal_handle,
                    feedback_msg,
                    0,
                    "Planning failed: no response from model.",
                    completed=True,
                )
                with self._main_goal_lock:
                    goal_handle.abort()
            return result_msg

        plan_description = ", ".join(s["function"]["name"] for s in plan)
        self._send_feedback(
            goal_handle,
            feedback_msg,
            0,
            f"Plan created with {len(plan)} steps: {plan_description}",
        )
        self.get_logger().info(f"Plan: {plan_description}")

        # Phase 2: Execution
        executed_results, aborted = self._execute_plan(plan, goal_handle, feedback_msg)

        # Finalize
        self._finalize_goal(
            goal_handle, feedback_msg, result_msg, executed_results, len(plan), aborted
        )
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

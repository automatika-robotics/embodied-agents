import json
import os
import re
import time
import uuid
from copy import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from ..clients.db_base import DBClient
from ..clients.model_base import ModelClient
from ..config import CortexConfig
from ..ros import (
    COMPONENT_ACTION_SERVER,
    COMPONENT_METHOD,
    COMPONENT_SERVICE,
    MONITOR_OWNER,
    Action,
    ActionClientHandler,
    BaseComponent,
    BaseComponentConfig,
    ComponentRunType,
    Event,
    Monitor,
    RegisteredAction,
    Routine,
    RoutineStatus,
    ServiceClientHandler,
    StreamingString,
    String,
    SystemActionRegistry,
    Topic,
    VisionLanguageAction,
    actions,
    get_logger,
    get_ros_msg_fields_dict,
    ros_msg_to_str,
)
from ..utils import strip_think_tokens, validate_func_args
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
    :param routines: Routines to host and offer to the planner as skills, for a
        recipe where no event triggers them and no UI is enabled. Each needs a
        description that the planner can read. Routines triggered by
        events or given to the UI are given to cortex and don't have to be listed here.
    :type routines: Optional[List[Routine]]
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
        "You are the task planning agent of a robot. You receive a request and "
        "turn it into tool calls.\n\n"
        "How to work:\n"
        "1. Research first. Use inspect_component to learn each component's "
        "topics, the fields of their messages, and its actions, until you know "
        "enough.\n"
        "2. Decide what the request is:\n"
        "   - A task to carry out now: return the plan as tool calls.\n"
        "   - A question, or nothing to do: respond with text only.\n\n"
        "Writing a plan:\n"
        "- Return ALL the steps in a single response, one tool call per step, "
        "in execution order. Never return fewer tool calls than needed, even "
        "when some arguments are not known yet.\n"
        "- Fill in the arguments you already know, such as topic names from "
        "inspection. For an argument that depends on the output of an earlier "
        'step, write exactly "<output from step N>", with N the number of that '
        "step. This is expected and correct: it is resolved at execution time "
        "from that step's result.\n\n"
        "How the tools behave:\n"
        "- A send_goal_to_* tool waits for its goal to finish before the next "
        "step. Pass wait_to_finish=false only when the following steps should "
        "run while the goal runs, such as speaking while navigating.\n"
        "- A component's action server runs one goal at a time. If a goal is "
        "rejected because the server is busy with a goal you did not send, and "
        "the task calls for it, stop that goal with the component's "
        "cancel_main_goal tool, then send yours.\n"
        "- The wait tool holds for a number of seconds when a step needs time "
        "to take effect. Do not use it to wait for a running goal: its progress "
        "is reported to you as it runs."
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
        "before moving on. Optionally follow with a brief reason after a colon.\n\n"
        "When you respond EXECUTE, you may also return a tool call for the next "
        "action with updated arguments based on the results of previous steps. "
        "For example, if a previous step produced text output, use that text as "
        "the argument for the next step instead of the placeholder from the plan."
        " But this is not just limited to text outputs, you can also use structured"
        " outputs from previous steps to fill input parameters of next steps."
    )

    @validate_func_args
    def __init__(
        self,
        *,
        actions: Optional[List[Action]] = None,
        output: Optional[Topic] = None,
        model_client: Optional[ModelClient] = None,
        db_client: Optional[DBClient] = None,
        routines: Optional[List[Routine]] = None,
        config: Optional[CortexConfig] = None,
        component_name: str,
        **kwargs,
    ):
        self.handled_outputs = [String, StreamingString]
        self._validate_actions(actions)
        for routine in routines or []:
            if not routine.description:
                raise ValueError(
                    "Each routine given to Cortex must have a description for the "
                    f"planner. Routine '{routine.name}' is missing one."
                )
        self._routines = routines or []

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

        # Effective planning prompt; augmented in custom_on_activate
        # based on discovered managed components (e.g. to handle Memory)
        self._effective_planning_prompt = self._PLANNING_PROMPT

        # Planning-prompt addenda. The effective planning prompt is always
        # _PLANNING_PROMPT + _robot_description + _sensors_description
        # + _memory_addendum
        self._robot_description = ""  # set by set_robot_description()
        self._sensors_description = ""  # set by set_sensor_descriptions()
        self._memory_addendum = ""  # set by _augment_planning_prompt_for_memory()
        self._events_addendum = ""  # set by _register_event_tools()

        # Initialize messages buffer
        self.messages: List[Dict] = [
            {"role": "system", "content": self._effective_planning_prompt}
        ]

        # Tool registries separated into planning and execution phases.
        # Planning tools (e.g. inspect_component) gather information.
        # Execution tools (actions, system tools) are what the plan consists of.
        self._planning_tools: Set = set()
        self._planning_tool_descriptions: List[Dict] = []
        self._execution_tools: Set = set()
        self._execution_tool_descriptions: List[Dict] = []

        # The action registry reference behind each tool built from it
        self._tool_refs: Dict[str, str] = {}

        # Routine tools: tool name -> routine name
        self._routine_tools: Dict[str, str] = {}
        # Plugin action tools: tool name -> (plugin, action name, tool parameters)
        self._plugin_action_tools: Dict[str, Tuple[Any, str, Dict]] = {}
        # Behavioral actions: dispatched via internal event system
        self._behavioral_actions = actions
        self._pure_internal_events = []
        self._additional_internal_actions = {}

        # Planning output buffer for failed plans
        self._planning_output: Optional[str] = None
        # Started routines which are followed till the end
        self._active_routines: Set[str] = set()

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

    # =========================================================================
    # Monitor setup (called by the Launcher)
    # =========================================================================

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
        action_registry: Optional[SystemActionRegistry] = None,
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
            action_registry=action_registry,
        )
        self.config = _config
        self._setup_internal_action_events(self._behavioral_actions)
        if self._routines:
            # Hosted by the Monitor (similar to when no UI is given)
            self.host_routines(self._routines)

    # =========================================================================
    # Tools: recipe actions (dispatched via internal events)
    # =========================================================================

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
    # Tools: plugin actions (run in the launcher process)
    # =========================================================================

    @staticmethod
    def _plugin_namespace(plugin: Any) -> str:
        """Namespace a plugin's actions are registered under: the plugin's id.

        The id is unique within a recipe, so two sensors of the same kind get
        distinct tools. An object without one falls back to its metadata name
        and to "robot" if that is empty.
        """
        plugin_id = getattr(plugin, "id", None)
        if isinstance(plugin_id, str) and plugin_id:
            return plugin_id
        metadata_name = getattr(getattr(plugin, "metadata", None), "name", "") or ""
        return metadata_name.strip().lower().replace(" ", "_") or "robot"

    def add_plugin_actions(self, plugin: Any) -> None:
        """Expose a plugin's actions to cortex as execution tools.

        Works for the robot plugin and for sensor plugins alike. Each
        `robot.ActionRegistry` factory on ``plugin`` is registered as
        ``{plugin_id}.{action_name}``.

        :param plugin: A `robot.Plugin` instance with an ``actions`` registry.
            Plugins without actions are a no-op.
        """
        if plugin is None or not getattr(plugin, "actions", None):
            return

        ns = self._plugin_namespace(plugin)

        tool_descriptions = plugin.actions.tool_descriptions(namespace=ns)
        if not tool_descriptions:
            return

        registered: List[str] = []
        for tool_desc in tool_descriptions:
            tool_name = tool_desc["function"]["name"]
            if tool_name in self._execution_tools:
                get_logger("cortex").warning(
                    f"Plugin action '{tool_name}' collides with an existing "
                    "tool; skipping."
                )
                continue

            local_name = tool_name.split(".", 1)[1]
            parameters = tool_desc["function"].get("parameters") or {}
            self._plugin_action_tools[tool_name] = (plugin, local_name, parameters)
            self._execution_tools.add(tool_name)
            self._execution_tool_descriptions.append(tool_desc)
            registered.append(tool_name)

        if registered:
            get_logger("cortex").info(
                f"Registered {len(registered)} plugin action(s) from '{ns}' "
                f"as Cortex execution tools: {registered}"
            )

    def _call_plugin_action(self, tool_name: str, args: Dict) -> str:
        """Build a plugin action from the call's arguments and run it.

        Only the arguments the tool declares reach the factory. Anything else
        would land in the ``Action``'s own keyword arguments. A call missing a
        required argument is refused.

        :param tool_name: The plugin tool, ``{plugin_id}.{action_name}``
        :param args: Parsed tool-call arguments
        :return: The action's message, or an error line on failure
        """
        plugin, local_name, parameters = self._plugin_action_tools[tool_name]
        declared = parameters.get("properties") or {}
        if missing := [k for k in parameters.get("required", []) if k not in args]:
            return (
                f"Error: {tool_name} failed with error: missing required "
                f"argument(s) {missing}"
            )
        if ignored := sorted(set(args) - set(declared)):
            self.get_logger().warning(
                f"Ignoring arguments {ignored} that {tool_name} does not declare"
            )
        call_args = {k: v for k, v in args.items() if k in declared}

        self.get_logger().info(
            f"Calling plugin action {tool_name} with args: {call_args}"
        )
        action = getattr(plugin.actions, local_name)(**call_args)
        action.action_name = tool_name
        success, message = action()
        if not success:
            return f"Error: {tool_name} failed with error: {message}"
        return message or f"{tool_name} executed successfully"

    # =========================================================================
    # Tools: registration
    # =========================================================================

    # Tools that install, remove and list runtime events. Each is the Monitor
    # method of the same name
    _EVENT_TOOLS = ("add_event", "remove_event", "list_events")

    # The comparisons a planner may write in an event condition
    _CONDITION_OPERATORS = (
        "equals",
        "not_equals",
        "greater_than",
        "greater_or_equal",
        "less_than",
        "less_or_equal",
        "contains",
        "not_contains",
        "is_in",
        "not_in",
        "contains_any",
        "contains_all",
    )

    # Tools that control a hosted routine by name. Each is the Monitor method
    # of the same name.
    _ROUTINE_CONTROLS = {
        "pause_routine": "Pause a running routine at its current step",
        "resume_routine": "Resume a paused routine from that step",
        "abort_routine": "Abort a running or paused routine",
    }

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

    def _register_system_tools(self):
        """Register system management capabilities and component actions as LLM tools.

        Called during activation after Monitor.activate() has created service
        clients. Everything a component offers is read from the action
        registry the Launcher built.

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

        # wait: execution tool, the one step that takes time on purpose
        wait_desc = {
            "type": "function",
            "function": {
                "name": "wait",
                "description": (
                    "Wait for a number of seconds before the next step, for "
                    "something started earlier to take effect. Not for waiting "
                    "on a running action goal, whose progress is reported to "
                    "you as it runs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "duration": {
                            "type": "number",
                            "description": "Seconds to wait",
                        },
                    },
                    "required": ["duration"],
                },
            },
        }
        self._execution_tools.add("wait")
        self._execution_tool_descriptions.append(wait_desc)

        if self.config.enable_events:
            self._register_event_tools()

        # Register all the tools the monitor has gathered from components
        self._register_component_tools()
        # Register all routines
        self._register_routine_tools()

    def _register_event_tools(self) -> None:
        """Offer the runtime events as tools, and tell the planner about
        standing instructions."""
        condition = {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic name"},
                "field": {
                    "type": "string",
                    "description": (
                        "Dotted path to a field of the message, as listed by "
                        "inspect_component, e.g. labels or pose.position.x. "
                        "Empty to fire on any message"
                    ),
                },
                "operator": {"type": "string", "enum": list(self._CONDITION_OPERATORS)},
                "value": {"description": "What the field is compared with"},
            },
            "required": ["topic"],
        }
        event_action = {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": (
                        "A component action, a send_goal_to_* tool or a routine tool"
                    ),
                },
                "arguments": {"type": "object"},
            },
            "required": ["tool"],
        }
        event_tools = {
            "add_event": {
                "description": (
                    "Install a standing event: when the condition on a topic "
                    "holds, run the actions. The event outlives the task and "
                    "fires on its own."
                ),
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "Name to list and remove the event by",
                    },
                    "conditions": {"type": "array", "items": condition},
                    "match": {
                        "type": "string",
                        "enum": ["all", "any"],
                        "description": "Whether all conditions must hold, or any one. Default all",
                    },
                    "actions": {"type": "array", "items": event_action},
                    "once": {
                        "type": "boolean",
                        "description": (
                            "Fire once and remove the event, the default. False "
                            "keeps it, firing each time the condition becomes true"
                        ),
                    },
                },
                "required": ["event_id", "conditions", "actions"],
            },
            "remove_event": {
                "description": "Remove a standing event by its id.",
                "properties": {"event_id": {"type": "string"}},
                "required": ["event_id"],
            },
            "list_events": {
                "description": "The standing events installed, by id.",
                "properties": {},
                "required": [],
            },
        }
        for tool_name, schema in event_tools.items():
            description = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": schema["description"],
                    "parameters": {
                        "type": "object",
                        "properties": schema["properties"],
                        "required": schema["required"],
                    },
                },
            }
            self._execution_tools.add(tool_name)
            self._execution_tool_descriptions.append(description)
            if tool_name == "list_events":
                # Planning may look before it decides between a plan and an event
                self._planning_tools.add(tool_name)
                self._planning_tool_descriptions.append(description)

        self._events_addendum = (
            "\n\n=== Standing Instructions ===\n"
            "Besides a task and a question, a request can include a standing "
            "instruction conditioned on an event happening: 'whenever X happens, "
            "do Y', 'if the battery drops below 20 percent, dock'. Install it with "
            "add_event instead of adding simple planning steps. Its condition "
            "names a topic and a field listed by inspect_component. An event outlives "
            "this task and fires on its own. list_events shows what is installed "
            "and remove_event removes one."
        )

    def _register_routine_tools(self) -> None:
        """Offer every routine the Monitor hosts as a skill.

        Each routine becomes a start tool, described in the recipe author's
        words, plus generic pause, resume and abort tools taking the routine
        name.
        """
        routines = self.get_routines()
        for routine in routines:
            self._register_routine(routine)
        if not routines or "pause_routine" in self._execution_tools:
            return
        names = ", ".join(routine["name"] for routine in routines)
        for tool_name, text in self._ROUTINE_CONTROLS.items():
            self._execution_tools.add(tool_name)
            self._execution_tool_descriptions.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"{text}. Routines: {names}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "routine_name": {
                                "type": "string",
                                "description": "Name of the routine",
                            },
                        },
                        "required": ["routine_name"],
                    },
                },
            })

    def _register_routine(self, routine: Dict) -> None:
        """One start tool for one hosted routine, named ``routine.<name>``.

        :param routine: The routine's cursor and description, as the Monitor
            lists them
        """
        name = routine["name"]
        tool_name = f"routine.{name}"
        if tool_name in self._routine_tools:
            return
        self._routine_tools[tool_name] = name
        described = (routine.get("description") or f"Runs routine '{name}'").rstrip(".")
        self._execution_tools.add(tool_name)
        self._execution_tool_descriptions.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": (
                    f"{described}. Steps: {', '.join(routine['steps'])}. Starts "
                    "the routine and returns; its progress is reported to you "
                    "as it runs."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        })

    def _register_component_tools(self) -> None:
        """Register what the action registry lists, as LLM tools.

        The registry is the Launcher's catalogue of everything the stack can
        be asked to do by name, built from every component whatever process
        it runs in. Cortex's own entries are left out (no spooky behaviour).
        """
        for entry in self._action_registry.list():
            # Skip monitors own methods
            if entry.owner in (self.node_name, MONITOR_OWNER):
                continue
            # Create tools for component methods, action servers and service requests
            tool = self._tool_from_entry(entry)
            if tool is None:
                continue
            tool_name, function, phase = tool
            if tool_name in self._tool_refs:
                # Registered on an earlier activation
                continue
            self._tool_refs[tool_name] = entry.ref
            description = {
                "type": "function",
                "function": {**function, "name": tool_name},
            }
            # segregate by planning and execution
            if phase in ("planning", "both"):
                self._planning_tools.add(tool_name)
                self._planning_tool_descriptions.append(description)
            if phase in ("execution", "both"):
                self._execution_tools.add(tool_name)
                self._execution_tool_descriptions.append(description)

    def _tool_from_entry(
        self, entry: RegisteredAction
    ) -> Optional[Tuple[str, Dict, str]]:
        """The tool a registry entry becomes with its name, its function
        description and the phase it is registered in. None for an entry
        that is not offered to the planner."""
        if entry.kind == COMPONENT_METHOD:
            # Skip lifecycle methods
            if entry.name in self._LIFECYCLE_METHODS or not entry.schema:
                return None
            function = entry.schema.get("function", entry.schema)
            return (
                f"{entry.owner}.{entry.name}",
                function,
                entry.schema.get("phase", "execution"),
            )

        interface = self._action_registry.interface_for(entry.ref)
        if interface is None or not entry.server_name:
            return None
        # NOTE: The short name has the owner prefix stripped, so putting the owner
        # back keeps the name a node-prefixed server had before
        name = f"{entry.owner}_{entry.name}"
        if entry.kind == COMPONENT_ACTION_SERVER:
            verb, message = "goal", interface.Goal
            text = (
                f"Send an action goal to the '{entry.owner}' component's "
                f"action server ({entry.server_name})."
            )
        else:
            verb, message = "request", interface.Request
            text = (
                f"Send a service request to the '{entry.owner}' component's "
                f"server ({entry.server_name})."
            )
        properties, required = goal_type_to_json_properties(message)
        if verb == "goal" and "wait_to_finish" not in properties:
            # Planner can explicitly choose concurrent execution for action servers
            properties["wait_to_finish"] = {
                "type": "boolean",
                "description": (
                    "Wait for the goal to finish before the next step. Default "
                    "true. False carries out the next steps in the plan while it runs."
                ),
            }
        function = {
            "description": text,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
        return f"send_{verb}_to_{name}", function, "execution"

    # =========================================================================
    # Tools: components (reached through the Monitor)
    # =========================================================================

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

        # The fields of each topic's message, required only for event conditions
        if self.config.enable_events:
            result += self._topic_fields(comp)

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

    def _call_component_action(self, tool_name: str, args: Dict) -> str:
        """Run a component action through the Monitor.

        The registry entry behind the tool is resolved to the Monitor's own
        callable for it, which sends the call over the component's
        ExecuteMethod service and reads the response back into the action
        contract. Returns the action's message, suitable to feed back to an
        LLM as a tool result, prefixed with ``"Error:"`` when the action failed.
        """
        self.get_logger().info(
            f"Calling component action {tool_name} with args: {args}"
        )
        ref = self._tool_refs.get(tool_name)
        if ref is None:
            return f"Error: Unknown tool '{tool_name}'"
        try:
            entry = self._action_registry.get(ref)
            # execute via the monitor
            success, message = self._executable_for(entry)(**args)
        except Exception as e:
            return f"Error calling {tool_name}: {e}"

        if not success:
            return f"Error: {tool_name} failed with error: {message}"
        return message or f"{tool_name} executed successfully"

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

        A server runs one goal at a time and rejects a new one while it does,
        so a goal this tool already has running is replaced: canceled, and
        waited on, before the new one is sent.

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
        action_client = self.get_action_client(action_name, action_type)
        try:
            if action_client.goal_accepted and not action_client.action_returned:
                canceled, why = action_client.cancel_request()
                if not canceled:
                    return (
                        f"Error: '{tool_name}' still has a goal running and it "
                        f"could not be canceled to make room for the new one: {why}"
                    )
                self._active_action_clients.pop(tool_name, None)
                self.get_logger().info(
                    f"Canceled the running goal of '{tool_name}' to send a new one"
                )
            sent = action_client.send_request_from_dict(goal_fields)
            if sent:
                self._active_action_clients[tool_name] = action_client
                return (
                    f"Action '{tool_name}' has been dispatched to '{component_name}' "
                    f"and is now running asynchronously."
                )
            if action_client.goal_rejected:
                return (
                    f"Error: '{component_name}' rejected the goal because it is "
                    "busy with another one. Stop that one first with the "
                    f"'{component_name}.cancel_main_goal' tool."
                )
            return (
                f"Error: Failed to construct or send action goal to "
                f"'{component_name}' from fields: {goal_fields}"
            )
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

    # =========================================================================
    # Tools: Cortex's own (parameter updates, routine control, wait)
    # =========================================================================

    def _update_parameter_tool(self, args: Dict) -> str:
        """The update_parameter tool, through the Monitor"""
        self.get_logger().info(
            f"Calling component action update_parameter with args: {args}"
        )
        success, message = self.update_parameter(
            args.get("component", ""),
            args.get("param_name", ""),
            args.get("new_value", ""),
        )
        if success:
            return "update_parameter executed successfully"
        return f"Error: update_parameter failed with error: {message}"

    def _run_routine_tool(self, tool_name: str, args: Dict) -> str:
        """Start, pause, resume or abort a routine.

        A started routine is followed by `_monitor_active_routines` until it
        ends, and aborted with the task if it is still running then.
        """
        if tool_name in self._routine_tools:
            name = self._routine_tools[tool_name]
            success, message = self.start_routine(name)
            if not success:
                return f"Error: {tool_name} failed with error: {message}"
            self._active_routines.add(name)
            return (
                f"Routine '{name}' started; its progress is reported to you as it runs."
            )
        success, message = getattr(self, tool_name)(args.get("routine_name", ""))
        return (
            message if success else f"Error: {tool_name} failed with error: {message}"
        )

    def _wait(self, duration: Any) -> str:
        """The wait tool dwells before the next step."""
        try:
            seconds = float(duration)
        except (TypeError, ValueError):
            return f"Error: wait takes a number of seconds, got {duration!r}"
        if seconds < 0:
            return f"Error: cannot wait for {seconds} seconds"
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0:
            handle = self._main_goal_handle
            if handle is not None and handle.is_cancel_requested:
                return (
                    f"Wait stopped after {seconds - remaining:.1f}s: the task "
                    "was cancelled"
                )
            time.sleep(min(self.config.monitoring_interval, remaining))
        return f"Waited {seconds:g}s"

    # =========================================================================
    # Tools: runtime events (standing instructions kept by the Monitor)
    # =========================================================================

    @staticmethod
    def _message_fields(topic: Topic) -> Dict:
        """The fields of a topic's message, nested as the message is"""
        ros_type = getattr(topic.msg_type, "_ros_type", None)
        return get_ros_msg_fields_dict(ros_type) if ros_type else {}

    @classmethod
    def _field_paths(cls, fields: Dict, prefix: str = "") -> List[str]:
        """Dotted field paths with their types, out of the nested fields"""
        paths: List[str] = []
        for name, kind in fields.items():
            path = f"{prefix}{name}"
            if isinstance(kind, dict):
                paths.extend(cls._field_paths(kind, f"{path}."))
            elif isinstance(kind, list):
                paths.extend(cls._field_paths(kind[0], f"{path}[]."))
            else:
                paths.append(f"{path}: {kind}")
        return paths

    def _topic_fields(self, comp: Any) -> str:
        """The fields of each of a component's topics, for inspection"""
        topics = [
            *(getattr(comp, "in_topics", None) or []),
            *(getattr(comp, "out_topics", None) or []),
        ]
        described = []
        for topic in topics:
            fields = self._message_fields(topic)
            if fields:
                type_name = getattr(topic.msg_type, "__name__", str(topic.msg_type))
                described.append(
                    f"  - {topic.name} ({type_name}): "
                    + ", ".join(self._field_paths(fields))
                )
        if not described:
            return ""
        return "\nTopic fields, for event conditions:\n" + "\n".join(described)

    def _condition_spec(self, condition: Dict) -> Dict:
        """One structured condition as the JSON leaf sugarcoat reads.

        The topic must be one a managed component reads or writes, and the
        field one the message has.

        :raises ValueError: Naming what was wrong and what is available
        """
        topics = {
            topic.name: topic
            for comp in self._managed_components.values()
            for topic in [
                *(getattr(comp, "in_topics", None) or []),
                *(getattr(comp, "out_topics", None) or []),
            ]
        }
        topic = topics.get(condition.get("topic", ""))
        if topic is None:
            raise ValueError(
                f"Unknown topic '{condition.get('topic', '')}'. Known topics: "
                f"{sorted(topics)}"
            )
        fields = self._message_fields(topic)
        path = [part for part in str(condition.get("field") or "").split(".") if part]
        node: Any = fields
        for part in path:
            if isinstance(node, list):
                node = node[0]
            if not isinstance(node, dict) or part not in node:
                raise ValueError(
                    f"Topic '{topic.name}' has no field '{'.'.join(path)}'. Fields: "
                    f"{', '.join(self._field_paths(fields))}"
                )
            node = node[part]
        operator = condition.get("operator") if path else None
        if path and operator not in self._CONDITION_OPERATORS:
            raise ValueError(
                f"A condition on a field needs an operator, one of "
                f"{', '.join(self._CONDITION_OPERATORS)}; got {operator!r}"
            )
        return {
            "type": "simple",
            "topic_name": topic.name,
            "topic_msg_type": getattr(topic.msg_type, "__name__", str(topic.msg_type)),
            "topic_qos_config": topic.qos_profile.to_dict(),
            "topic_use_plugin": topic.use_plugin,
            "attribute_path": path,
            "operator": operator or "none",
            "ref_value": condition.get("value") if path else None,
        }

    def _event_action_spec(self, call: Dict) -> Dict:
        """One of an event's actions, a tool call, as an action spec by
        registry reference.

        :raises ValueError: For a tool an event cannot run
        """
        tool = call.get("tool", "")
        args = self._parse_tool_args(call.get("arguments", {}))
        if tool in self._routine_tools:
            return {
                "ref": f"{MONITOR_OWNER}/start_routine",
                "kwargs": {"routine_name": self._routine_tools[tool]},
            }
        if tool in self._ROUTINE_CONTROLS:
            return {
                "ref": f"{MONITOR_OWNER}/{tool}",
                "kwargs": {"routine_name": args.get("routine_name", "")},
            }
        ref = self._tool_refs.get(tool)
        if ref is None:
            raise ValueError(
                f"An event cannot run '{tool}'. It runs component actions, "
                "send_goal_to_* tools and routine tools"
            )
        return self._action_spec(ref, args)

    def _event_spec(self, args: Dict) -> Tuple[Dict, List[Dict]]:
        """The event and its actions as the Monitor's registration reads them.

        :raises ValueError: If anything in the call cannot be installed
        """
        if not args.get("event_id"):
            raise ValueError("An event needs an event_id")
        conditions = [self._condition_spec(c) for c in args.get("conditions") or []]
        if not conditions:
            raise ValueError("An event needs at least one condition")
        actions = [self._event_action_spec(call) for call in args.get("actions") or []]
        if not actions:
            raise ValueError("An event needs at least one action")
        condition = conditions[0]
        if len(conditions) > 1:
            # NOTE: sugarcoat's logic operators: AND is 1, OR is 2
            condition = {
                "type": "composite",
                "logic_operator": 2 if args.get("match") == "any" else 1,
                "sub_conditions": conditions,
            }
        once = bool(args.get("once", True))
        event = {
            "name": args["event_id"],
            "condition": condition,
            "handle_once": once,
            # A kept event fires when the condition becomes true, not on
            # every message that satisfies it
            "on_change": not once,
            "keep_event_delay": 0.0,
        }
        return event, actions

    def _run_event_tool(self, tool_name: str, args: Dict) -> str:
        """Install, remove or list runtime events through the Monitor"""
        if tool_name == "list_events":
            return self.list_events()[1]
        if tool_name == "remove_event":
            success, message = self.remove_event(args.get("event_id", ""))
            return message if success else f"Error: {message}"
        try:
            event, event_actions = self._event_spec(args)
        except ValueError as e:
            return f"Error: {e}"
        success, message = self._add_event_from_spec(
            event=event, actions=event_actions, event_id=args["event_id"]
        )
        return message if success else f"Error: {message}"

    # =========================================================================
    # Planning prompt
    # =========================================================================

    def set_robot_description(self, plugin: Any) -> None:
        """Augment the planning prompt with the attached robot's identity.

        When a robot plugin is attached, Cortex is embodied in a specific
        physical robot. This builds a compact identity addendum from the
        plugin's metadata and prepends it to the planning prompt, so the agent
        can answer "who/what are you" questions about its own body instead of
        hallucinating a generic robot.

        :param plugin: A `robot.RobotPlugin` instance, or ``None``.
        """
        if plugin is None:
            return
        try:
            desc = plugin.describe()
        except Exception as e:
            get_logger("cortex").error(f"Failed to read robot plugin description: {e}")
            return

        meta = desc.get("metadata", {})
        name = meta.get("name", "") or "Unknown"
        vendor = meta.get("vendor", "") or "unknown vendor"
        version = meta.get("version", "") or "unspecified"
        blurb = meta.get("description", "") or "(no description provided)"

        feedback_keys = [f["key"] for f in desc.get("feedbacks", [])]
        command_keys = [c["key"] for c in desc.get("commands", [])]
        action_names = [a["name"] for a in desc.get("actions", [])]

        # Namespace the plugin actions get registered under as execution tools
        ns = self._plugin_namespace(plugin)

        def _join(items: List[str]) -> str:
            return ", ".join(items) if items else "(none)"

        if action_names:
            action_hint = (
                f"The robot actions above are available to you as execution "
                f"tools (named '{ns}.<action>', e.g. '{ns}.{action_names[0]}'); "
                "use them when a task calls for a physical behaviour."
            )
        else:
            action_hint = ""

        self._robot_description = (
            "\n\n=== Robot Identity ===\n"
            "You are not a disembodied assistant -- you are the intelligence "
            "of a physical robot. Answer questions about who or what you are "
            "based on the following, never on guesswork:\n"
            f"  - Name: {name}\n"
            f"  - Vendor: {vendor}\n"
            f"  - Version: {version}\n"
            f"  - Description: {blurb}\n"
            "Capabilities exposed through your robot body:\n"
            f"  - Sensor feedback you receive: {_join(feedback_keys)}\n"
            f"  - Commands you can issue: {_join(command_keys)}\n"
            f"  - Built-in robot actions: {_join(action_names)}\n" + action_hint
        )

        self._compose_planning_prompt()
        get_logger("cortex").info(
            f"Planning prompt augmented with robot identity for '{name}'."
        )

    def set_sensor_descriptions(self, plugins: List[Any]) -> None:
        """Augment the planning prompt with the sensors attached to the robot.

        Each attached sensor is listed under the id its tools are namespaced by,
        with what its plugin says about it, so the agent knows what each sensor
        is and can tell two of the same kind apart.

        :param plugins: The attached `robot.SensorPlugin` instances.
        """
        entries: List[str] = []
        for plugin in plugins or []:
            try:
                desc = plugin.describe()
            except Exception as e:
                get_logger("cortex").error(
                    f"Failed to read sensor plugin description: {e}"
                )
                continue
            ns = self._plugin_namespace(plugin)
            meta = desc.get("metadata", {})
            entry = f"  - {ns}: {meta.get('name', '') or ns}"
            if vendor := meta.get("vendor", ""):
                entry += f", by {vendor}"
            if blurb := meta.get("description", ""):
                entry += f"\n    {blurb}"
            if feedback_keys := [f["key"] for f in desc.get("feedbacks", [])]:
                entry += f"\n    Feedback: {', '.join(feedback_keys)}"
            if action_names := [a["name"] for a in desc.get("actions", [])]:
                tools = ", ".join(f"{ns}.{a}" for a in action_names)
                entry += f"\n    Tools: {tools}"
            entries.append(entry)

        if entries:
            self._sensors_description = (
                "\n\n=== Attached Sensors ===\n"
                "Besides your robot body, these sensors are attached. Each is "
                "named by the id its execution tools are namespaced under:\n"
                + "\n".join(entries)
            )
            get_logger("cortex").info(
                f"Planning prompt augmented with {len(entries)} attached sensor(s)."
            )
        else:
            self._sensors_description = ""
        self._compose_planning_prompt()

    def _augment_planning_prompt_for_memory(self) -> None:
        """Append memory-aware guidance to the planning prompt when a
        Memory component is present in the managed recipe.

        Detects the Memory component by class name, verifies it exposes
        the expected episode/body-status tools, and builds an addendum
        instructing the planner to wrap tasks in episodes, check body
        status during planning, and use memory retrieval tools.
        """
        # Find a managed Memory component
        memory_comp = None
        for comp in self._managed_components.values():
            if type(comp).__name__ == "Memory":
                memory_comp = comp
                break

        if memory_comp is None:
            return

        prefix = f"{memory_comp.node_name}."
        start_ep_tool = f"{prefix}start_episode"
        end_ep_tool = f"{prefix}end_episode"
        body_status_tool = f"{prefix}body_status"
        store_note_tool = f"{prefix}store_specific_memory"

        # Only augment if the expected tools are actually registered
        required = {start_ep_tool, end_ep_tool}
        if not required.issubset(self._execution_tools):
            return

        # Perception retrieval tools (planning phase), excluding body_status
        perception_retrieval_tools = sorted(
            name
            for name in self._planning_tools
            if name.startswith(prefix) and name != body_status_tool
        )
        perception_str = (
            ", ".join(perception_retrieval_tools)
            if perception_retrieval_tools
            else "(none)"
        )

        # Pre-compute memory component inspection so layer names are in
        # the prompt directly — the planner doesn't need to spend a round
        # trip calling inspect_component on memory
        try:
            memory_inspection = memory_comp.inspect_component()
        except Exception as e:
            self.get_logger().warning(
                f"Failed to inspect memory component for prompt augmentation: {e}"
            )
            memory_inspection = ""

        addendum = (
            "\n\n=== Memory Guidance ===\n"
            f"A spatio-temporal memory component '{memory_comp.node_name}' is "
            "available. It has TWO distinct retrieval surfaces you must "
            "distinguish between:\n"
            f"  - Perception memory: past external observations (what the "
            f"robot saw, detected, or was told). Retrieved via: {perception_str}.\n"
            f"  - Interoception / body state: the robot's own internal "
            f"readings (battery, temperature, joint health, fault flags). "
            f"Retrieved via '{body_status_tool}' ONLY — internal state is "
            f"NOT returned by perception retrieval tools.\n\n"
            + (
                f"Memory component inspection (use these layer names as "
                f"'layer' / 'layers' filters on retrieval tools):\n"
                f"{memory_inspection}\n\n"
                if memory_inspection
                else ""
            )
            + "TASK CLASSIFICATION — decide which type of task you have:\n"
            "  (A) PERCEPTION QUERY: user is asking about past external "
            "observations (e.g. 'what happened in the last episode?', "
            "'where did you see the cup?', 'what did you do today?').\n"
            "  (B) BODY QUERY: user is asking about the robot's internal "
            "state (e.g. 'are you low on battery?', 'is anything wrong?', "
            "'what is your current temperature?').\n"
            "  (C) ACTION TASK: user wants the robot to do something in "
            "the world (e.g. 'take a picture and describe it', "
            "'navigate to the kitchen').\n\n"
            "FOR PERCEPTION QUERY TASKS:\n"
            f"  - Call one or more perception retrieval tools during "
            f"planning: {perception_str}.\n"
            "  - Respond with a text answer summarizing what you found.\n"
            "  - DO NOT return any execution tool calls. DO NOT start or "
            "end an episode — query tasks do not generate new observations.\n\n"
            "FOR BODY QUERY TASKS:\n"
            f"  - Call '{body_status_tool}' during planning (optionally "
            f"filtered by an internal-state layer name listed above).\n"
            "  - Respond with a text answer summarizing the readings.\n"
            "  - DO NOT return execution tool calls. DO NOT start or end "
            "an episode.\n\n"
            "FOR ACTION TASKS:\n"
            f"  1. Call '{body_status_tool}' during planning to read the "
            "robot's current internal state. If readings indicate a "
            "problem (very low battery, fault, overheating), abort the task "
            "with a clear text explanation instead of proceeding.\n"
            "  2. Optionally call perception retrieval tools during "
            f"planning to ground the task in past memory (e.g. use "
            f"'{memory_comp.node_name}.locate' to find a known object's "
            "position).\n"
            f"  3. Begin your execution plan with '{start_ep_tool}' using a "
            "short, descriptive episode name derived from the task.\n"
            "  4. Execute the task steps.\n"
            f"  5. If during the task you derive a fact worth remembering "
            f"(e.g. a VLM description you want to persist, an operator "
            f"instruction, a decision), include '{store_note_tool}' in the "
            f"plan to record it.\n"
            f"  6. End your execution plan with '{end_ep_tool}' to "
            "consolidate observations from this task into long-term memory.\n"
            f"  Episodes are mandatory for ACTION tasks — always wrap the "
            f"plan between '{start_ep_tool}' and '{end_ep_tool}'."
        )

        self._memory_addendum = addendum
        self._compose_planning_prompt()
        self.get_logger().info(
            f"Planning prompt augmented for Memory component '{memory_comp.node_name}'."
        )

    def _compose_planning_prompt(self) -> None:
        """Rebuild the effective planning prompt from the base prompt plus
        every active addendum (robot identity, attached sensors, memory
        guidance).

        Single source of truth and Idempotent.
        """
        self._effective_planning_prompt = (
            self._PLANNING_PROMPT
            + self._robot_description
            + self._sensors_description
            + self._memory_addendum
            + self._events_addendum
        )
        self.config._system_prompt = self._effective_planning_prompt
        self.messages = [{"role": "system", "content": self._effective_planning_prompt}]

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
            self._augment_planning_prompt_for_memory()
        # Always (re)compose so the planning prompt reflects every addendum
        # set so far -- robot identity, memory guidance, or neither --
        # regardless of which augmentation paths ran.
        self._compose_planning_prompt()

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

    def _warmup(self):
        """Warm up and verify model connectivity."""
        self._call_inference({
            "query": [
                {"role": "system", "content": self._PLANNING_PROMPT},
                {"role": "user", "content": "Hello"},
            ],
            **self.config._get_inference_params(),
        })

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
            {"role": "system", "content": self._effective_planning_prompt},
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
            self.get_logger().debug(
                f"[Planning step {step + 1}] calling {fn_name}({fn_args})"
            )
            tool_result = self._execute_planning_tool(fn_name, fn_args)
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

    def _plan_task(
        self, task: str, messages: Optional[List[Dict]] = None
    ) -> Tuple[Optional[List[Dict]], List[Dict]]:
        """Multi-step planning loop that researches components before producing a plan.

        The LLM is given both planning tools and execution tools.
        Each iteration it may:

        - Call planning tools to gather information (results are fed back).
        - Call execution tools — these become the plan and end the loop.
        - Respond with text only — no actions needed, loop ends.

        :param task: The high-level task description
        :param messages: Optional pre-existing planning messages to continue
            from (e.g. after feeding back execution results). If None, a
            fresh conversation is started.
        :returns: Tuple of (plan, planning_messages). plan is a list of
            tool_call dicts or None if no actions needed. planning_messages
            is the conversation history, preserved so execution results can
            be fed back for continued planning.
        """
        all_tools = self._planning_tool_descriptions + self._execution_tool_descriptions
        if messages is None:
            messages = self._build_planning_messages(task)
        output = ""

        self.get_logger().debug(
            f"[Planning] starting task={task!r} with "
            f"{len(self._planning_tool_descriptions)} planning tools, "
            f"{len(self._execution_tool_descriptions)} execution tools, "
            f"max_steps={self.config.max_planning_steps}"
        )

        for step in range(self.config.max_planning_steps):
            inference_input = {
                "query": messages,
                **self.config._get_inference_params(),
            }
            if all_tools:
                inference_input["tools"] = all_tools

            self.get_logger().debug(
                f"[Planning step {step + 1}/{self.config.max_planning_steps}] "
                "invoking planner LLM"
            )
            result = self._call_inference(inference_input)
            if not result:
                self.get_logger().error(
                    f"Inference failed during planning step {step + 1}."
                )
                return None, messages

            output = result.get("output") or ""
            if self.config.strip_think_tokens:
                output = strip_think_tokens(output)

            tool_calls = result.get("tool_calls")
            if not tool_calls:
                self.get_logger().debug(
                    f"[Planning step {step + 1}] planner returned no tool calls; "
                    "ending loop"
                )
                self._planning_output = output
                return None, messages

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

            self.get_logger().debug(
                f"[Planning step {step + 1}] planner requested "
                f"{len(planning_calls)} planning call(s) "
                f"{[tc['function']['name'] for tc in planning_calls]} and "
                f"{len(execution_calls)} execution call(s) "
                f"{[tc['function']['name'] for tc in execution_calls]}"
            )

            if planning_calls:
                self._process_planning_calls(planning_calls, messages, output, step)

            if execution_calls:
                self.get_logger().debug(
                    f"[Planning] producing plan with {len(execution_calls)} "
                    f"step(s) after {step + 1} planning iteration(s)"
                )
                return self._finalize_plan(execution_calls), messages

        self.get_logger().warning(
            f"Planning reached max steps ({self.config.max_planning_steps}) "
            "without producing an execution plan."
        )
        self._planning_output = output
        return None, messages

    def _append_execution_results_to_planning(
        self,
        messages: List[Dict],
        plan: List[Dict],
        executed_results: List[Dict],
    ) -> None:
        """Append executed plan steps and their results to the planning
        conversation so the LLM can continue planning with context.

        Each executed step is added as an assistant tool_call followed by
        a tool response message with the result.
        """
        tool_calls_msg = []
        for i, step in enumerate(plan):
            tool_calls_msg.append({
                "id": f"exec_{i}",
                "type": "function",
                "function": step["function"],
            })

        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls_msg,
        })

        for i, result in enumerate(executed_results):
            messages.append({
                "role": "tool",
                "tool_call_id": f"exec_{i}",
                "content": result["result"],
            })

    def _execute_planning_tool(self, tool_name: str, args: Dict) -> str:
        """Execute a planning-phase tool.

        Handles the built-in ``inspect_component`` tool plus any component
        action that was registered into the planning toolset (actions
        decorated with ``phase=ActionPhase.PLANNING`` or ``BOTH``).
        """
        if tool_name == "inspect_component":
            return self._inspect_component(args.get("component", ""))
        if tool_name == "list_events":
            return self._run_event_tool(tool_name, args)
        if tool_name in self._planning_tools and tool_name in self._tool_refs:
            parsed_args = self._parse_tool_args(args)
            return self._call_component_action(tool_name, parsed_args)
        return f"Error: Unknown planning tool '{tool_name}'."

    # =========================================================================
    # Phase 2: Execution with confirmation
    # =========================================================================

    def _parse_tool_args(self, fn_args) -> Dict:
        """Parse tool arguments, deserializing JSON strings where needed."""
        # OpenAI-compatible endpoints return tool-call arguments as a
        # JSON string and Ollama returns a dict. Normalize to a dict first.
        if isinstance(fn_args, str):
            fn_args = fn_args.strip()
            try:
                fn_args = json.loads(fn_args) if fn_args else {}
            except json.JSONDecodeError:
                fn_args = {}
        if not isinstance(fn_args, dict):
            return {}
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
    ) -> Tuple[str, Optional[Dict]]:
        """Ask the LLM whether the next planned step should be executed.

        The confirmation call includes execution tools so the LLM can
        return a tool call with resolved arguments (e.g. using output
        from a previous step as input to the next).

        :param plan: Full list of planned tool_calls
        :param executed_results: Results of already-executed steps
        :param step_index: Index of the next step to confirm
        :returns: Tuple of (decision, resolved_step) where decision is
            one of "EXECUTE", "SKIP", "ABORT", "CONTINUE" and
            resolved_step is a tool_call dict with updated arguments
            (or None to use the pre-planned arguments)
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

        # Include execution tools so the LLM can return a tool call
        # with arguments resolved from prior step results
        if self._execution_tool_descriptions:
            inference_input["tools"] = self._execution_tool_descriptions

        result = self._call_inference(inference_input)
        if not result:
            self.get_logger().warning(
                "Confirmation inference failed; defaulting to EXECUTE."
            )
            return "EXECUTE", None

        output = (result.get("output") or "").strip()
        if self.config.strip_think_tokens:
            output = strip_think_tokens(output).strip()

        # Check for a tool call with resolved arguments
        resolved_step = None
        if tool_calls := result.get("tool_calls"):
            resolved_step = tool_calls[0]
            self.get_logger().debug(
                f"Confirmation returned resolved tool call: {resolved_step}"
            )

        upper = output.upper()
        for token in ("ABORT", "SKIP", "CONTINUE"):
            if upper.startswith(token):
                return token, None
        # EXECUTE, either explicitly stated or implied by a tool call
        return "EXECUTE", resolved_step

    def _wait_for_active_clients(
        self, goal_handle, feedback_msg, plan, executed_results, step_index, label: str
    ) -> Tuple[str, Optional[Dict]]:
        """Poll active async actions until the LLM stops returning CONTINUE.

        :returns: Tuple of (decision, resolved_step)
        """
        decision, resolved_step = self._confirm_step(plan, executed_results, step_index)
        while decision == "CONTINUE":
            if goal_handle.is_cancel_requested:
                return "ABORT", None
            self._send_feedback(
                goal_handle,
                feedback_msg,
                step_index,
                f"{label}: waiting for async actions to complete...",
            )
            time.sleep(self.config.monitoring_interval)
            decision, resolved_step = self._confirm_step(
                plan, executed_results, step_index
            )
            self.get_logger().info(f"[{label}] re-check -> {decision}")
        return decision, resolved_step

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

    def _execute_system_tool(self, tool_name: str, args: Dict) -> str:
        """Execute an execution-phase system tool or a component action.

        A tool built from the action registry is dispatched by the kind of
        its entry: a goal to an action server, a request to a service, and
        anything else as a component method run through the Monitor.
        """
        try:
            if tool_name == "update_parameter":
                return self._update_parameter_tool(args)
            if tool_name in self._EVENT_TOOLS:
                return self._run_event_tool(tool_name, args)
            if tool_name == "wait":
                return self._wait(args.get("duration"))
            if tool_name in self._routine_tools or tool_name in self._ROUTINE_CONTROLS:
                return self._run_routine_tool(tool_name, args)
            if tool_name in self._plugin_action_tools:
                return self._call_plugin_action(tool_name, args)
            ref = self._tool_refs.get(tool_name)
            if ref is None:
                return f"Error: Unknown tool '{tool_name}'"
            entry = self._action_registry.get(ref)
            interface = self._action_registry.interface_for(ref)
            if entry.kind == COMPONENT_ACTION_SERVER:
                goal_fields = {
                    key: value for key, value in args.items() if key != "wait_to_finish"
                }
                return self._send_action_goal_from_dict(
                    tool_name, entry.owner, entry.server_name, interface, goal_fields
                )
            if entry.kind == COMPONENT_SERVICE:
                return self._send_service_request_from_dict(
                    entry.owner, entry.server_name, interface, args
                )
            return self._call_component_action(tool_name, args)
        except Exception as e:
            return f"Error calling {tool_name}: {e}"

    # =========================================================================
    # Compiled execution: a run of known steps as one routine
    # =========================================================================

    # How the planner marks an argument that depends on an earlier result
    _PLACEHOLDER = re.compile(r"<output from step \d+>")

    @classmethod
    def _has_placeholder(cls, value: Any) -> bool:
        """Whether an argument waits for an earlier step's output"""
        if isinstance(value, str):
            return cls._PLACEHOLDER.search(value) is not None
        if isinstance(value, dict):
            return any(cls._has_placeholder(item) for item in value.values())
        if isinstance(value, list):
            return any(cls._has_placeholder(item) for item in value)
        return False

    def _compiled_ref(self, step: Dict) -> Optional[str]:
        """The registry reference a plan step compiles to, or None.

        A step compiles when it is a component action, an action goal or a
        wait, and every argument is known now.
        """
        function = step["function"]
        args = self._parse_tool_args(function.get("arguments", {}))
        if self._has_placeholder(args):
            return None
        if function["name"] == "wait":
            ref = f"{MONITOR_OWNER}/wait"
            return ref if ref in self._action_registry else None
        ref = self._tool_refs.get(function["name"])
        if ref is None:
            return None
        kind = self._action_registry.get(ref).kind
        if kind == COMPONENT_ACTION_SERVER:
            # A goal the planner does not wait is not dispatched as a routine, and
            # the steps after it run alongside it
            return ref if args.get("wait_to_finish", True) else None
        return ref if kind == COMPONENT_METHOD else None

    def _compilable_run(self, plan: List[Dict], start: int) -> Dict[int, str]:
        """The consecutive compilable steps from `start`, as plan index to
        registry reference, maybe none.

        A single step is a run only when it is a goal: awaited, cancellable
        and with the server's outcome as the planner's result, that is worth
        a routine. A lone component action runs directly, since a routine of
        one would buy it nothing but a registration and a wait for its cursor.
        """
        run: Dict[int, str] = {}
        if not self.config.compile_routines:
            return run
        for index in range(start, len(plan)):
            ref = self._compiled_ref(plan[index])
            if ref is None:
                break
            run[index] = ref
        # Only an action server call can be a one step routine, that too if its
        # wait_to_finish param is True
        if len(run) == 1:
            ((_, ref),) = run.items()
            if self._action_registry.get(ref).kind != COMPONENT_ACTION_SERVER:
                return {}
        return run

    def _compilable_runs(self, plan: List[Dict]) -> Dict[int, Dict[int, str]]:
        """Every run in a plan, keyed by the index of its first step.

        Decided once, before anything runs: a run is decided by the planned
        arguments, which execution does not change. A run that later cannot
        be built has no run recorded inside it, so its steps run directly
        without the routine being attempted again from each of them.
        """
        runs: Dict[int, Dict[int, str]] = {}
        index = 0
        while index < len(plan):
            run = self._compilable_run(plan, index)
            if run:
                runs[index] = run
            index += len(run) or 1
        return runs

    def _action_spec(self, ref: str, args: Dict) -> Dict:
        """An action by registry reference with its arguments, as the Monitor's
        spec builder reads it.
        """
        if self._action_registry.get(ref).kind == COMPONENT_ACTION_SERVER:
            goal = {
                key: value for key, value in args.items() if key != "wait_to_finish"
            }
            return {"ref": ref, "goal": goal}
        return {"ref": ref, "kwargs": args}

    def _routine_spec(self, plan: List[Dict], run: Dict[int, str]) -> Dict:
        """Create a routine spec from a run of plan steps."""
        steps = []
        for index, ref in run.items():
            function = plan[index]["function"]
            args = self._parse_tool_args(function.get("arguments", {}))
            spec = self._action_spec(ref, args)
            spec["name"] = f"{index + 1}_{function['name']}"
            if self._action_registry.get(ref).kind == COMPONENT_METHOD:
                spec["timeout"] = self.config.step_timeout
                spec["on_timeout"] = "fail"
            steps.append(spec)
        return {
            "name": f"cortex_plan_{uuid.uuid4().hex[:8]}",
            "description": f"Steps {min(run) + 1} to {max(run) + 1} of the current task",
            "steps": steps,
        }

    def _compiled_step(self, spec: Dict) -> Action:
        """Build one step of a compiled routine."""
        if spec["ref"] == f"{MONITOR_OWNER}/wait":
            return actions.wait(
                duration=float(spec["kwargs"]["duration"]), name=spec["name"]
            )
        return self._action_from_spec(spec)

    def _execute_compiled(
        self, plan, run, goal_handle, feedback_msg, executed_results
    ) -> Optional[Tuple[int, Optional[bool]]]:
        """Run a compilable run of plan steps as one routine.

        :returns: None when the routine could not be built or started. Otherwise
            the plan index to go on from and the verdict: None to carry on, True
            when the task is aborted, False to return to planning with a result
            for every step
        """
        first, last = min(run), max(run)
        spec = self._routine_spec(plan, run)
        name = spec["name"]
        try:
            routine = Routine.from_spec(spec, self._compiled_step)
            added, why = self.add_routine(routine, replace=True)
            if not added:
                raise RuntimeError(why)
            started, why = self.start_routine(name)
            if not started:
                raise RuntimeError(why)
        except Exception as e:
            self.get_logger().warning(
                f"Steps {first + 1} to {last + 1} run one at a time, their "
                f"routine could not be started: {e}"
            )
            self.remove_routine(name, force=True)
            return None

        self._send_feedback(
            goal_handle,
            feedback_msg,
            first + 1,
            f"Steps {first + 1}-{last + 1}/{len(plan)}: running as routine '{name}'",
        )
        status = self._follow_routine(routine, first, goal_handle, feedback_msg)
        executed_results.extend(self._compiled_results(plan, run, routine, status))
        self.remove_routine(name, force=True)

        next_index = last + 1
        if status == RoutineStatus.COMPLETED:
            return next_index, None
        if goal_handle.is_cancel_requested:
            self._cancel_task(goal_handle)
            return next_index, True
        if status == RoutineStatus.ABORTED:
            # Stopped from outside, by the UI or an event (operator intervention)
            self._send_feedback(
                goal_handle,
                feedback_msg,
                next_index,
                f"Plan aborted: {executed_results[-1]['result']}",
            )
            return next_index, True
        # Failed at a step. Back to planning with the failure in view
        for index in range(next_index, len(plan)):
            executed_results.append({
                "step": index,
                "action": plan[index]["function"]["name"],
                "result": "NOT RUN: an earlier step failed",
                "failed": False,
            })
        return len(plan), False

    def _follow_routine(
        self, routine: Routine, first: int, goal_handle, feedback_msg
    ) -> RoutineStatus:
        """Report a compiled routine's cursor as task feedback until it ends.
        A cancelled task aborts it.

        :param first: Plan index of the routine's first step; the steps are
            consecutive, so the cursor's index counts on from it
        """
        while True:
            state = routine.state
            status = RoutineStatus(state["status"])
            if status.is_terminal():
                return status
            if goal_handle.is_cancel_requested:
                self.abort_routine(routine.name, reason="the task was cancelled")
                continue
            plan_step = first + state["index"] + 1
            text = f"Routine step '{state['active_step']}' {status}"
            if state["step_message"]:
                text += f": {state['step_message']}"
            self._send_feedback(goal_handle, feedback_msg, plan_step, text)
            time.sleep(self.config.monitoring_interval)

    @staticmethod
    def _compiled_results(
        plan, run: Dict[int, str], routine: Routine, status: RoutineStatus
    ) -> List[Dict]:
        """One result per step of the run, from what the steps returned"""
        messages = {entry["step"]: entry for entry in routine.step_messages()}
        results = []
        for index in run:
            name = plan[index]["function"]["name"]
            entry = messages.get(f"{index + 1}_{name}")
            if entry is None:
                why = (
                    f"the routine was aborted ({routine.state['abort_reason']})"
                    if status == RoutineStatus.ABORTED
                    else "an earlier step failed"
                )
                result = f"NOT RUN: {why}"
            elif entry["succeeded"]:
                result = entry["message"] or f"{name} executed successfully"
            else:
                result = f"Error: {name} failed with error: {entry['message']}"
            results.append({
                "step": index,
                "action": name,
                "result": result,
                "failed": result.startswith("Error"),
            })
        return results

    # =========================================================================
    # Executing a plan
    # =========================================================================

    def _execute_plan(self, plan, goal_handle, feedback_msg) -> Tuple[List[Dict], bool]:
        """Execute a plan: compilable runs as routines, the rest step by step
        with a confirmation call each.

        The confirmation call includes execution tools so the LLM can
        return a tool call with arguments resolved from prior step results.

        :returns: (executed_results, aborted)
        """
        executed_results: List[Dict] = []
        total = len(plan)
        # Check for compilable routines
        runs = self._compilable_runs(plan)
        i = 0
        while i < total:
            if goal_handle.is_cancel_requested:
                self._cancel_task(goal_handle)
                return executed_results, True

            run = runs.get(i)
            compiled = (
                self._execute_compiled(
                    plan, run, goal_handle, feedback_msg, executed_results
                )
                if run
                else None
            )
            if compiled is not None:
                i, verdict = compiled
                if verdict is None:
                    continue
                return executed_results, verdict

            step = plan[i]
            fn_name = step["function"]["name"]
            label = f"Step {i + 1}/{total} ({fn_name})"

            # Confirm (may wait for async actions via CONTINUE).
            # The LLM may also return a tool call with resolved arguments.
            decision, resolved_step = self._wait_for_active_clients(
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
                i += 1
                continue

            # Use resolved arguments from the confirmation call if available,
            # otherwise fall back to the pre-planned arguments
            effective_step = resolved_step if resolved_step else step
            fn_args = effective_step["function"].get("arguments", {})

            # EXECUTE
            args_str = f" with {fn_args}" if fn_args else ""
            self._send_feedback(
                goal_handle, feedback_msg, i + 1, f"Executing {label}{args_str}"
            )

            step_result = self._execute_action_step(effective_step)
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
            i += 1

        # Wait for any remaining async actions after the last step
        if self._monitor_active_clients():
            decision, _ = self._wait_for_active_clients(
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

    def _cancel_task(self, goal_handle) -> None:
        """End the task as cancelled, stopping everything it started"""
        self.get_logger().info("Task cancelled by client.")
        with self._main_goal_lock:
            self._cancel_all_active_clients()
            goal_handle.canceled()

    # =========================================================================
    # Following running goals and routines
    # =========================================================================

    def _monitor_active_clients(self) -> Optional[str]:
        """Helper method to get a status update on the ongoing Action clients
        and routines

        :return: Active tools status feedback
        :rtype: Optional[str]
        """
        if not self._active_action_clients and not self._active_routines:
            # Nothing to monitor
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
        feedback_lines += self._monitor_active_routines()
        feedback_lines += "[End Of Tools Status Update]\n"
        # Remove completed actions from the active clients registry
        for tool in completed_actions:
            self._active_action_clients.pop(tool)
        return feedback_lines

    def _monitor_active_routines(self) -> str:
        """Status lines for the routines this task started, from their
        cursors. A routine that has ended is reported once and dropped.

        :rtype: str
        """
        if not self._active_routines:
            return ""
        cursors = {routine["name"]: routine for routine in self.get_routines()}
        lines = ""
        for name in sorted(self._active_routines):
            cursor = cursors.get(name)
            if cursor is None:
                self._active_routines.discard(name)
                lines += f"- routine.{name}: GONE | The routine was removed\n"
                continue
            status = RoutineStatus(cursor["status"])
            if status.is_terminal():
                self._active_routines.discard(name)
                detail = (
                    cursor["abort_reason"]
                    if status == RoutineStatus.ABORTED
                    else cursor["step_message"]
                )
                lines += f"- routine.{name}: {status.upper()} | {detail}\n"
                continue
            lines += (
                f"- routine.{name}: {status} at step '{cursor['active_step']}' "
                f"(running for {cursor['elapsed']}s)"
            )
            if cursor["step_message"]:
                lines += f" | Last step said: {cursor['step_message']}"
            lines += "\n"
        return lines

    def _cancel_all_active_clients(self):
        """Helper method to cancel all action Action clients and abort the
        routines this task started

        :return: Cancellation error message if errors occurred
        :rtype: Optional[str]
        """
        for name in list(self._active_routines):
            aborted, why = self.abort_routine(name, reason="the task ended")
            if not aborted:
                self.get_logger().info(f"Routine '{name}' was not aborted: {why}")
        self._active_routines.clear()
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

    def _finalize_goal(
        self,
        goal_handle,
        feedback_msg,
        result_msg,
        executed_results,
        plan_len,
        aborted,
        voluntarily_stopped: bool = False,
    ) -> None:
        """Set the final goal status based on execution results.

        Success criterion: the planning loop ended voluntarily (the LLM
        returned no more tool calls), meaning the LLM considered the task
        complete. Earlier per-step errors are tolerated as long as the
        loop recovered — the LLM had the chance to continue if needed.
        """
        has_failures = any(r.get("failed") for r in executed_results)
        had_recovery = has_failures and voluntarily_stopped

        # If the goal is already in a terminal state e.g on client cancellation,
        # don't attempt another state transition.
        if not goal_handle.is_active:
            self._cancel_all_active_clients()
            return

        if aborted:
            with self._main_goal_lock:
                self._cancel_all_active_clients()
                goal_handle.abort()
        elif voluntarily_stopped:
            result_msg.success = True
            if had_recovery:
                feedback = "Plan finished despite errors along the way."
                self.get_logger().info(feedback)
            else:
                feedback = f"All {plan_len} steps completed."
                self.get_logger().info(
                    f"Task completed: {len(executed_results)} steps executed."
                )
            self._send_feedback(
                goal_handle, feedback_msg, plan_len, feedback, completed=True
            )
            with self._main_goal_lock:
                self._cancel_all_active_clients()
                goal_handle.succeed()
        else:
            # Loop exhausted without the LLM voluntarily stopping
            self._send_feedback(
                goal_handle,
                feedback_msg,
                plan_len,
                f"Plan did not finish: reached max_execution_steps "
                f"({self.config.max_execution_steps}).",
                completed=True,
            )
            self.get_logger().warning(
                "Task exhausted max_execution_steps without voluntary completion."
            )
            with self._main_goal_lock:
                self._cancel_all_active_clients()
                goal_handle.abort()

    def main_action_callback(self, goal_handle):
        """Action server callback. Iterative plan-execute loop.

        Plans a batch of steps, executes them, then feeds the results back
        into the planning conversation so the LLM can continue with
        additional steps if needed. The loop ends when the planner returns
        no more tool calls or an abort/failure occurs.

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

        all_executed_results: List[Dict] = []
        total_steps = 0
        planning_messages = None
        voluntarily_stopped = False

        for _ in range(self.config.max_execution_steps):
            # Plan (or continue planning with prior results)
            plan, planning_messages = self._plan_task(task, planning_messages)

            if plan is None:
                # No more actions needed
                if all_executed_results:
                    # We already executed some steps — LLM signalled done
                    voluntarily_stopped = True
                    break
                # First iteration, no plan at all
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
                total_steps,
                f"Plan: {plan_description}",
            )
            self.get_logger().info(f"Plan: {plan_description}")

            # Execute this batch
            executed_results, aborted = self._execute_plan(
                plan, goal_handle, feedback_msg
            )
            all_executed_results.extend(executed_results)
            total_steps += len(plan)

            if aborted:
                self._finalize_goal(
                    goal_handle,
                    feedback_msg,
                    result_msg,
                    all_executed_results,
                    total_steps,
                    aborted=True,
                )
                return result_msg

            # Feed execution results back into the planning conversation
            # so the LLM can continue with the next steps
            self._append_execution_results_to_planning(
                planning_messages, plan, executed_results
            )

        # Loop exited voluntarily (LLM stopped) or exhausted
        self._finalize_goal(
            goal_handle,
            feedback_msg,
            result_msg,
            all_executed_results,
            total_steps,
            aborted=False,
            voluntarily_stopped=voluntarily_stopped,
        )
        return result_msg

    # =========================================================================
    # Not used in action server mode
    # =========================================================================

    def _create_input(self, *args, **kwargs) -> Optional[Dict]:
        """Not used -- Cortex builds inputs in _plan_task and _confirm_step."""
        return None

    def _execution_step(self, *args, **kwargs):
        """Not used -- Cortex runs as an action server."""
        pass

    def _handle_websocket_streaming(self):
        """Not used -- streaming is disabled for Cortex."""
        pass

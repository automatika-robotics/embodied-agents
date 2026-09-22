"""Tests for Cortex component — requires rclpy."""

import json
import pytest
from typing import Tuple
from unittest.mock import MagicMock

from ros_sugar.robot import ActionRegistry, PluginMetadata, SensorPlugin, plugin_action
from std_srvs.srv import SetBool

from agents.config import CortexConfig
from agents.ros import (
    COMPONENT_ACTION_SERVER,
    COMPONENT_METHOD,
    COMPONENT_SERVICE,
    MONITOR_METHOD,
    MONITOR_OWNER,
    ActionPhase,
    ActionReturnType,
    Topic,
    Action,
    ComponentRunType,
    RegisteredAction,
    SystemActionRegistry,
    VisionLanguageAction,
    component_action,
)
from agents.components.cortex import Cortex
from tests.conftest import mock_component_internals


def _registry(*entries):
    """A registry holding the given (entry, interface) pairs"""
    registry = SystemActionRegistry()
    for entry, interface in entries:
        registry.add(entry, interface=interface)
    return registry


def _method(owner, name, described=True, phase=None):
    """A component method entry, described the way the decorator stores it"""
    schema = None
    if described:
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} on {owner}",
                "parameters": {
                    "type": "object",
                    "properties": {"topic": {"type": "string"}},
                    "required": ["topic"],
                },
            },
        }
        if phase:
            schema["phase"] = phase
    entry = RegisteredAction(
        ref=f"{owner}/{name}",
        owner=owner,
        name=name,
        kind=COMPONENT_METHOD,
        schema=schema,
    )
    return entry, None


def _server(owner, server_name, kind=COMPONENT_ACTION_SERVER, interface=None):
    """An action server or service entry, named as the registry shortens it"""
    short = server_name.strip("/")
    if short.startswith(f"{owner}/"):
        short = short[len(owner) + 1 :]
    entry = RegisteredAction(
        ref=f"{owner}/{short.replace('/', '_')}",
        owner=owner,
        name=short.replace("/", "_"),
        kind=kind,
        server_name=server_name,
    )
    return entry, interface or VisionLanguageAction


def _make_mock_action(name="test_action", description="A test action"):
    """Create a mock Action with the given name and description."""
    action = MagicMock(spec=Action)
    action.action_name = name
    action.description = description
    return action


def _make_cortex(actions, mock_model_client, component_name, **cortex_kwargs):
    """Construct a Cortex and run the action-registration step that the
    Launcher normally triggers through ``_init_internal_monitor``.
    Tests that assert on ``_execution_tools`` /
    ``_execution_tool_descriptions`` need this because registration was
    deliberately moved out of ``__init__`` so that Monitor init cannot
    overwrite the registry (see commit 8d7eb95)."""
    comp = Cortex(
        outputs=[Topic(name="out", msg_type="String")],
        actions=actions,
        model_client=mock_model_client,
        config=cortex_kwargs.pop("config", CortexConfig()),
        component_name=component_name,
        **cortex_kwargs,
    )
    comp._setup_internal_action_events(comp._behavioral_actions)
    return comp


class TestCortexConstruction:
    def test_with_model_client(self, rclpy_init, mock_model_client):
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_mock_action()],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex",
        )
        assert comp.model_client is mock_model_client

    def test_with_local_model(self, rclpy_init):
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_mock_action()],
            config=CortexConfig(enable_local_model=True),
            component_name="test_cortex_local",
        )
        assert comp.config.enable_local_model is True

    def test_no_client_no_local_raises(self, rclpy_init):
        with pytest.raises(RuntimeError):
            Cortex(
                outputs=[Topic(name="out", msg_type="String")],
                actions=[_make_mock_action()],
                config=CortexConfig(),
                component_name="test_cortex_fail",
            )

    def test_empty_actions_allowed(self, rclpy_init, mock_model_client):
        """Empty actions list is valid — Cortex can still use system tools."""
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_no_actions",
        )
        assert len(comp._execution_tool_descriptions) == 0

    def test_action_without_description_raises(self, rclpy_init, mock_model_client):
        action = _make_mock_action(name="bad_action", description="")
        with pytest.raises(ValueError):
            Cortex(
                outputs=[Topic(name="out", msg_type="String")],
                actions=[action],
                model_client=mock_model_client,
                config=CortexConfig(),
                component_name="test_cortex_no_desc",
            )

    def test_config_enforced(self, rclpy_init, mock_model_client):
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_mock_action()],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_config",
        )
        assert comp.config.chat_history is True
        assert comp.config.stream is False

    def test_action_server_run_type(self, rclpy_init, mock_model_client):
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_mock_action()],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_runtype",
        )
        assert comp.run_type == ComponentRunType.ACTION_SERVER

    def test_with_db_client(self, rclpy_init, mock_model_client, mock_db_client):
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_mock_action()],
            model_client=mock_model_client,
            db_client=mock_db_client,
            config=CortexConfig(enable_rag=True, collection_name="test_col"),
            component_name="test_cortex_rag",
        )
        assert comp.db_client is mock_db_client


class TestCortexActions:
    def test_action_registers_tool_description(self, rclpy_init, mock_model_client):
        action = _make_mock_action(name="navigate", description="Go somewhere")
        comp = _make_cortex(
            actions=[action],
            mock_model_client=mock_model_client,
            component_name="test_cortex_tools",
        )

        assert len(comp._execution_tool_descriptions) == 1
        tool_desc = comp._execution_tool_descriptions[0]
        assert tool_desc["function"]["name"] == "navigate"
        assert tool_desc["function"]["description"] == "Go somewhere"

    def test_action_registers_in_execution_tools(self, rclpy_init, mock_model_client):
        action = _make_mock_action(name="grasp", description="Grasp object")
        comp = _make_cortex(
            actions=[action],
            mock_model_client=mock_model_client,
            component_name="test_cortex_events",
        )

        assert "grasp" in comp._execution_tools

    def test_multiple_actions(self, rclpy_init, mock_model_client):
        actions = [
            _make_mock_action(name="navigate", description="Go to location"),
            _make_mock_action(name="grasp", description="Grasp object"),
            _make_mock_action(name="release", description="Release object"),
        ]
        comp = _make_cortex(
            actions=actions,
            mock_model_client=mock_model_client,
            component_name="test_cortex_multi",
        )

        assert len(comp._execution_tool_descriptions) == 3
        assert len(comp._execution_tools) == 3

    def test_dispatch_action_unknown(self, rclpy_init, mock_model_client):
        action = _make_mock_action(name="real_action", description="Exists")
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[action],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_dispatch",
        )
        mock_component_internals(comp)
        # Simulate what Monitor.__init__ would populate
        comp.emit_internal_event_methods = {"real_action": MagicMock()}

        result = comp._dispatch_action("nonexistent")
        assert "does not exist" in result


class TestCortexPlanning:
    def test_plan_task_returns_tool_calls(self, rclpy_init, mock_model_client):
        mock_model_client.inference.return_value = {
            "output": "I'll navigate then grasp.",
            "tool_calls": [
                {"function": {"name": "navigate", "arguments": {}}},
                {"function": {"name": "grasp", "arguments": {}}},
            ],
        }
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[
                _make_mock_action(name="navigate", description="Go"),
                _make_mock_action(name="grasp", description="Grab"),
            ],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_plan",
        )
        mock_component_internals(comp)

        plan, _ = comp._plan_task("fetch a cup")
        assert plan is not None
        assert len(plan) == 2
        assert plan[0]["function"]["name"] == "navigate"
        assert plan[1]["function"]["name"] == "grasp"

    def test_plan_task_no_tool_calls_returns_none(self, rclpy_init, mock_model_client):
        mock_model_client.inference.return_value = {
            "output": "I don't need to do anything.",
        }
        comp = Cortex(
            actions=[_make_mock_action()],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_plan_none",
        )
        mock_component_internals(comp)

        plan, _ = comp._plan_task("just say hello")
        assert plan is None
        assert comp._planning_output == "I don't need to do anything."

    def test_plan_truncated_to_max_execution_steps(self, rclpy_init, mock_model_client):
        mock_model_client.inference.return_value = {
            "output": "",
            "tool_calls": [
                {"function": {"name": f"step_{i}", "arguments": {}}} for i in range(20)
            ],
        }
        comp = Cortex(
            actions=[_make_mock_action()],
            model_client=mock_model_client,
            config=CortexConfig(max_execution_steps=5),
            component_name="test_cortex_truncate",
        )
        mock_component_internals(comp)

        plan, _ = comp._plan_task("big task")
        assert len(plan) == 5

    def test_plan_with_inspect_then_execute(self, rclpy_init, mock_model_client):
        """Planning loop: first call inspects, second call returns action tools."""
        mock_model_client.inference.side_effect = [
            # Step 1: LLM calls inspect_component (planning tool)
            {
                "output": "Let me check the vision component.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "inspect_component",
                            "arguments": {"component": "vision"},
                        }
                    },
                ],
            },
            # Step 2: LLM returns action tool calls (execution tools)
            {
                "output": "Now I know what to do.",
                "tool_calls": [
                    {"function": {"name": "navigate", "arguments": {}}},
                ],
            },
        ]
        comp = Cortex(
            actions=[
                _make_mock_action(name="navigate", description="Go"),
            ],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_plan_loop",
        )
        mock_component_internals(comp)
        comp._planning_tools.add("inspect_component")
        comp._managed_components = {}

        plan, _ = comp._plan_task("find an object")
        assert plan is not None
        assert len(plan) == 1
        assert plan[0]["function"]["name"] == "navigate"
        assert mock_model_client.inference.call_count == 2

    def test_plan_exhausts_max_planning_steps(self, rclpy_init, mock_model_client):
        """Planning loop exits when max_planning_steps is reached."""
        mock_model_client.inference.return_value = {
            "output": "Still researching...",
            "tool_calls": [
                {
                    "function": {
                        "name": "inspect_component",
                        "arguments": {"component": "x"},
                    }
                },
            ],
        }
        comp = Cortex(
            actions=[_make_mock_action()],
            model_client=mock_model_client,
            config=CortexConfig(max_planning_steps=3),
            component_name="test_cortex_plan_exhaust",
        )
        mock_component_internals(comp)
        comp._planning_tools.add("inspect_component")
        comp._managed_components = {}

        plan, _ = comp._plan_task("complex task")
        assert plan is None
        assert mock_model_client.inference.call_count == 3


class TestCortexConfirmation:
    def test_confirm_execute(self, rclpy_init, mock_model_client):
        mock_model_client.inference.return_value = {"output": "EXECUTE"}
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_mock_action()],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_confirm_exec",
        )
        mock_component_internals(comp)

        plan = [{"function": {"name": "navigate", "arguments": {}}}]
        decision, resolved = comp._confirm_step(plan, [], 0)
        assert decision == "EXECUTE"
        assert resolved is None

    def test_confirm_skip(self, rclpy_init, mock_model_client):
        mock_model_client.inference.return_value = {"output": "SKIP: already done"}
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_mock_action()],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_confirm_skip",
        )
        mock_component_internals(comp)

        plan = [{"function": {"name": "navigate", "arguments": {}}}]
        decision, resolved = comp._confirm_step(plan, [], 0)
        assert decision == "SKIP"
        assert resolved is None

    def test_confirm_abort(self, rclpy_init, mock_model_client):
        mock_model_client.inference.return_value = {"output": "ABORT: unsafe condition"}
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_mock_action()],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_confirm_abort",
        )
        mock_component_internals(comp)

        plan = [{"function": {"name": "navigate", "arguments": {}}}]
        decision, resolved = comp._confirm_step(plan, [], 0)
        assert decision == "ABORT"
        assert resolved is None

    def test_confirm_defaults_to_execute(self, rclpy_init, mock_model_client):
        mock_model_client.inference.return_value = {"output": "Sure, go ahead!"}
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_mock_action()],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_confirm_default",
        )
        mock_component_internals(comp)

        plan = [{"function": {"name": "navigate", "arguments": {}}}]
        decision, _ = comp._confirm_step(plan, [], 0)
        assert decision == "EXECUTE"

    def test_confirm_execute_with_resolved_args(self, rclpy_init, mock_model_client):
        """When the LLM returns EXECUTE with a tool call, the resolved step is returned."""
        mock_model_client.inference.return_value = {
            "output": "EXECUTE",
            "tool_calls": [
                {
                    "function": {
                        "name": "tts.say",
                        "arguments": {"text": "A red cup on the table"},
                    }
                },
            ],
        }
        comp = Cortex(
            actions=[_make_mock_action()],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_confirm_resolved",
        )
        mock_component_internals(comp)

        plan = [{"function": {"name": "tts.say", "arguments": {"text": "placeholder"}}}]
        decision, resolved = comp._confirm_step(plan, [], 0)
        assert decision == "EXECUTE"
        assert resolved is not None
        assert resolved["function"]["arguments"]["text"] == "A red cup on the table"


class TestNoLLMMethods:
    def test_no_llm_methods(self, rclpy_init, mock_model_client):
        """Cortex extends ModelComponent, not LLM."""
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_mock_action()],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_no_llm",
        )
        assert not hasattr(comp, "register_tool")
        assert not hasattr(comp, "set_component_prompt")


def _make_mock_plugin(name="Lite3", action_names=("sit_stand", "stop")):
    """Mock a `~ros_sugar.robot.RobotPlugin` exposing two zero-arg actions.

    Mirrors the real surface that `Cortex.add_plugin_actions` consumes:
    ``plugin.metadata.name`` for the namespace, ``plugin.actions`` with a
    ``tool_descriptions(namespace=...)`` method and per-name factories
    accessed via ``getattr(plugin.actions, name)``.
    """
    plugin = MagicMock()
    plugin.metadata = MagicMock()
    plugin.metadata.name = name

    plugin.actions = MagicMock()
    plugin.actions.tool_descriptions.side_effect = lambda namespace=None: [
        {
            "type": "function",
            "function": {
                "name": f"{namespace}.{n}" if namespace else n,
                "description": f"Do {n}",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
        for n in action_names
    ]
    # Each factory call returns a fresh mock Action (so action_name can be set
    # independently per registration).
    plugin.actions.configure_mock(**{
        n: MagicMock(return_value=_make_mock_action(name=n, description=f"Do {n}"))
        for n in action_names
    })
    return plugin


class TestCortexPluginActions:
    """The plugin-action bridge: factories on a plugin registered as
    namespaced execution tools, built from the tool call's arguments and run
    by cortex when called."""

    def test_register_plugin_actions(self, rclpy_init, mock_model_client):
        plugin = _make_mock_plugin(name="Lite3", action_names=("sit_stand", "stop"))
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_plugin",
        )

        mock_component_internals(comp)
        comp.add_plugin_actions(plugin)

        assert "lite3.sit_stand" in comp._execution_tools
        assert "lite3.stop" in comp._execution_tools
        assert "lite3.sit_stand" in comp._plugin_action_tools
        # Run by cortex itself, not dispatched through an internal event that
        # could carry no arguments
        assert "lite3.sit_stand" not in comp._additional_internal_actions
        names = [t["function"]["name"] for t in comp._execution_tool_descriptions]
        assert "lite3.sit_stand" in names
        assert "lite3.stop" in names

    def test_namespace_falls_back_to_robot_when_metadata_name_empty(
        self, rclpy_init, mock_model_client
    ):
        plugin = _make_mock_plugin(name="", action_names=("dock",))
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_plugin_default_ns",
        )

        mock_component_internals(comp)
        comp.add_plugin_actions(plugin)

        assert "robot.dock" in comp._execution_tools

    def test_namespace_normalizes_whitespace(self, rclpy_init, mock_model_client):
        plugin = _make_mock_plugin(name="My Robot", action_names=("dock",))
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_plugin_ns_norm",
        )

        mock_component_internals(comp)
        comp.add_plugin_actions(plugin)

        assert "my_robot.dock" in comp._execution_tools

    def test_does_not_double_register_on_collision(
        self, rclpy_init, mock_model_client
    ):
        plugin = _make_mock_plugin(name="Lite3", action_names=("stop",))
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_plugin_collision",
        )

        mock_component_internals(comp)
        comp.add_plugin_actions(plugin)
        comp.add_plugin_actions(plugin)  # second call should skip with a warning

        assert sum(
            1
            for t in comp._execution_tool_descriptions
            if t["function"]["name"] == "lite3.stop"
        ) == 1

    def test_none_plugin_is_noop(self, rclpy_init, mock_model_client):
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_plugin_none",
        )

        comp.add_plugin_actions(None)

        assert len(comp._execution_tools) == 0
        assert len(comp._execution_tool_descriptions) == 0

    def test_plugin_without_actions_is_noop(self, rclpy_init, mock_model_client):
        plugin = MagicMock()
        plugin.actions = None
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_plugin_no_actions",
        )

        mock_component_internals(comp)
        comp.add_plugin_actions(plugin)

        assert len(comp._execution_tools) == 0
        assert len(comp._execution_tool_descriptions) == 0

    def test_factory_failure_is_reported_as_a_tool_error(
        self, rclpy_init, mock_model_client
    ):
        """Factories are only called when the tool is, with its arguments, so
        one that cannot build its action fails that call, not registration."""
        plugin = _make_mock_plugin(name="Lite3", action_names=("ok", "broken"))
        plugin.actions.broken.side_effect = RuntimeError("boom")
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_plugin_factory_fail",
        )

        mock_component_internals(comp)
        # Simulate what Monitor.__init__ would populate
        comp.emit_internal_event_methods = {}
        comp.add_plugin_actions(plugin)

        assert "lite3.ok" in comp._execution_tools
        assert "lite3.broken" in comp._execution_tools
        result = comp._execute_action_step(_tool_call("lite3.broken"))
        assert result.startswith("Error")
        assert "boom" in result


class _PtzCamera(SensorPlugin):
    """A sensor plugin with a parametric action and one that takes nothing."""

    def __init__(self):
        self.metadata = PluginMetadata(
            name="PTZ Camera", vendor="Acme", description="A pan-tilt camera."
        )
        self.aimed = []
        self.refuse = False
        self.actions = ActionRegistry(
            {"look_at": self._make_look_at, "stop": self._make_stop}
        )

    @plugin_action(
        description={
            "name": "look_at",
            "description": "Aim the camera.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pan_deg": {"type": "number"},
                    "tilt_deg": {"type": "number"},
                },
                "required": ["pan_deg", "tilt_deg"],
            },
        }
    )
    def _make_look_at(self, pan_deg=0.0, tilt_deg=0.0, **action_kwargs):
        def look_at(pan_deg, tilt_deg) -> Tuple[bool, str]:
            if self.refuse:
                return False, "the camera refused the move"
            self.aimed.append((pan_deg, tilt_deg))
            return True, f"aimed at pan {pan_deg}, tilt {tilt_deg}"

        return Action(
            method=look_at,
            kwargs={"pan_deg": pan_deg, "tilt_deg": tilt_deg},
            **action_kwargs,
        )

    @plugin_action(description="Stop moving the camera.")
    def _make_stop(self, **action_kwargs):
        def stop() -> Tuple[bool, str]:
            return True, "stopped"

        return Action(method=stop, **action_kwargs)


def _tool_call(name, arguments=None):
    return {"function": {"name": name, "arguments": arguments or {}}}


class TestCortexRunsPluginActions:
    """A plugin tool builds its action from the LLM's arguments and reports
    the action's real outcome."""

    def _cortex(self, mock_model_client, component_name, *plugins):
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name=component_name,
        )
        mock_component_internals(comp)
        # Simulate what Monitor.__init__ would populate
        comp.emit_internal_event_methods = {}
        for plugin in plugins:
            comp.add_plugin_actions(plugin)
        return comp

    def test_tools_are_namespaced_by_plugin_id(self, rclpy_init, mock_model_client):
        """Two sensors of the same kind share a metadata name; their ids tell
        their tools apart."""
        front, rear = _PtzCamera(id="front_cam"), _PtzCamera(id="rear_cam")
        comp = self._cortex(mock_model_client, "test_cortex_plugin_ids", front, rear)

        assert {"front_cam.look_at", "rear_cam.look_at"} <= comp._execution_tools

        comp._execute_action_step(
            _tool_call("rear_cam.look_at", {"pan_deg": 10, "tilt_deg": 0})
        )
        assert rear.aimed == [(10, 0)]
        assert front.aimed == []

    def test_the_llms_arguments_reach_the_action(self, rclpy_init, mock_model_client):
        camera = _PtzCamera(id="front_cam")
        comp = self._cortex(mock_model_client, "test_cortex_plugin_args", camera)

        # OpenAI-compatible endpoints send arguments as a JSON string
        result = comp._execute_action_step(
            _tool_call("front_cam.look_at", '{"pan_deg": 90, "tilt_deg": 30}')
        )

        assert camera.aimed == [(90, 30)]
        # The action's own message, not a bare "dispatched"
        assert result == "aimed at pan 90, tilt 30"

    def test_a_call_missing_a_required_argument_is_refused(
        self, rclpy_init, mock_model_client
    ):
        """The factory would otherwise fill the gap with its default, which
        for an aiming action is a real position."""
        camera = _PtzCamera(id="front_cam")
        comp = self._cortex(mock_model_client, "test_cortex_plugin_required", camera)

        result = comp._execute_action_step(
            _tool_call("front_cam.look_at", {"pan_deg": 90})
        )

        assert result.startswith("Error")
        assert "tilt_deg" in result
        assert camera.aimed == []

    def test_undeclared_arguments_are_ignored(self, rclpy_init, mock_model_client):
        """An argument the tool does not declare would otherwise land in the
        Action's own keyword arguments."""
        camera = _PtzCamera(id="front_cam")
        comp = self._cortex(mock_model_client, "test_cortex_plugin_extra", camera)

        result = comp._execute_action_step(
            _tool_call("front_cam.stop", {"max_retries": 3, "speed": "fast"})
        )

        assert result == "stopped"

    def test_a_failed_action_is_an_error_result(self, rclpy_init, mock_model_client):
        camera = _PtzCamera(id="front_cam")
        camera.refuse = True
        comp = self._cortex(mock_model_client, "test_cortex_plugin_refused", camera)

        result = comp._execute_action_step(
            _tool_call("front_cam.look_at", {"pan_deg": 90, "tilt_deg": 30})
        )

        assert result.startswith("Error")
        assert "the camera refused the move" in result


class TestCortexSensorDescriptions:
    """``set_sensor_descriptions`` tells the planner what each attached sensor
    is, under the id its tools are named by."""

    def test_sensors_augment_planning_prompt(self, rclpy_init, mock_model_client):
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_sensor_desc",
        )
        mock_component_internals(comp)

        comp.set_sensor_descriptions([_PtzCamera(id="front_cam")])

        prompt = comp._effective_planning_prompt
        assert "Attached Sensors" in prompt
        assert "front_cam: PTZ Camera, by Acme" in prompt
        assert "A pan-tilt camera." in prompt
        assert "front_cam.look_at" in prompt
        assert comp._PLANNING_PROMPT in prompt
        assert comp.messages[0]["content"] == prompt

    def test_composes_with_robot_identity(self, rclpy_init, mock_model_client):
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_sensor_desc_robot",
        )
        mock_component_internals(comp)

        comp.set_sensor_descriptions([_PtzCamera(id="front_cam")])
        comp.set_robot_description(_make_mock_plugin_with_describe(name="Lite3"))

        prompt = comp._effective_planning_prompt
        assert "Robot Identity" in prompt
        assert "Attached Sensors" in prompt

    def test_no_sensors_is_noop(self, rclpy_init, mock_model_client):
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_sensor_desc_none",
        )
        mock_component_internals(comp)

        comp.set_sensor_descriptions([])

        assert comp._sensors_description == ""
        assert comp._effective_planning_prompt == comp._PLANNING_PROMPT


def _make_mock_plugin_with_describe(
    name="Lite3",
    vendor="DeepRobotics",
    version="1.0",
    description="A four-legged quadruped robot.",
):
    """Mock a `RobotPlugin` exposing the ``describe()`` surface that
    ``Cortex.set_robot_description`` consumes."""
    plugin = MagicMock()
    plugin.describe.return_value = {
        "metadata": {
            "name": name,
            "vendor": vendor,
            "version": version,
            "description": description,
        },
        "feedbacks": [{"key": "Odometry"}, {"key": "Imu"}, {"key": "Float64"}],
        "commands": [{"key": "Twist"}],
        "actions": [{"name": "sit_stand"}, {"name": "stop"}],
        "events": [{"name": "low_battery"}],
    }
    return plugin


class TestCortexRobotDescription:
    """``set_robot_description`` augments the planning prompt with the
    attached robot's identity so the agent answers "who are you" correctly."""

    def test_description_augments_planning_prompt(
        self, rclpy_init, mock_model_client
    ):
        plugin = _make_mock_plugin_with_describe(
            name="Lite3", description="A nimble quadruped."
        )
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_robot_desc",
        )
        mock_component_internals(comp)

        comp.set_robot_description(plugin)

        prompt = comp._effective_planning_prompt
        assert "Robot Identity" in prompt
        assert "Lite3" in prompt
        assert "DeepRobotics" in prompt
        assert "A nimble quadruped." in prompt
        # Capability overview is summarized
        assert "Odometry" in prompt and "Twist" in prompt
        # The base planning prompt is preserved
        assert comp._PLANNING_PROMPT in prompt
        # config + messages buffer kept in sync
        assert comp.config._system_prompt == prompt
        assert comp.messages[0]["content"] == prompt

    def test_none_plugin_is_noop(self, rclpy_init, mock_model_client):
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_robot_desc_none",
        )
        mock_component_internals(comp)

        comp.set_robot_description(None)

        assert comp._robot_description == ""
        assert comp._effective_planning_prompt == comp._PLANNING_PROMPT

    def test_memory_augmentation_preserves_robot_description(
        self, rclpy_init, mock_model_client
    ):
        """``_augment_planning_prompt_for_memory`` composes on top of the
        robot identity rather than clobbering it."""
        plugin = _make_mock_plugin_with_describe(name="Lite3")
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_robot_desc_memory",
        )
        mock_component_internals(comp)
        comp.set_robot_description(plugin)

        # No Memory component -> _augment_planning_prompt_for_memory early-returns;
        # the robot identity must survive untouched.
        comp._managed_components = {}
        comp._augment_planning_prompt_for_memory()
        assert "Robot Identity" in comp._effective_planning_prompt

    def test_compose_is_single_source_of_truth(
        self, rclpy_init, mock_model_client
    ):
        """Both addendum slots compose into the prompt regardless of which
        augmentation ran -- the composer always rebuilds from both."""
        plugin = _make_mock_plugin_with_describe(name="Lite3")
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_compose",
        )
        mock_component_internals(comp)

        # Memory addendum set first, robot description second
        comp._memory_addendum = "\n\n=== Memory Guidance ===\nstub"
        comp._compose_planning_prompt()
        comp.set_robot_description(plugin)

        prompt = comp._effective_planning_prompt
        assert comp._PLANNING_PROMPT in prompt
        assert "Robot Identity" in prompt
        assert "Memory Guidance" in prompt


class TestStandingInForTheMonitor:
    """The Launcher builds the registry of what the stack can be asked to do
    and hands it to whichever node monitors the stack. Cortex takes it either
    way it can arrive"""

    def test_a_registry_passed_at_construction_is_kept(
        self, rclpy_init, mock_model_client
    ):
        registry = SystemActionRegistry.from_components([])
        comp = _make_cortex(
            [_make_mock_action()], mock_model_client, "test_cortex_registry"
        )

        comp._init_internal_monitor(components_names=[], action_registry=registry)

        assert comp._action_registry is registry
        assert comp._registry_given

    def test_a_registry_handed_over_later_is_installed(
        self, rclpy_init, mock_model_client
    ):
        """What the Launcher does: Cortex is built without the registry, and
        the one rebuilt with Cortex removed is handed over before activation"""
        registry = SystemActionRegistry.from_components([])
        comp = _make_cortex(
            [_make_mock_action()], mock_model_client, "test_cortex_registry_later"
        )
        comp._init_internal_monitor(components_names=[])
        assert not comp._registry_given

        comp.set_action_registry(registry)

        assert comp._action_registry is registry
        assert comp._registry_given


class TestDispatchingAGoal:
    """A component's action server runs one goal at a time and rejects a new
    one while it does, so Cortex replaces a goal it has running on that
    server before sending another"""

    TOOL = "send_goal_to_vla_run"

    def _dispatch(self, comp, client):
        mock_component_internals(comp)
        comp.get_action_client = MagicMock(return_value=client)
        return comp._send_action_goal_from_dict(
            self.TOOL, "vla", "vla/run", MagicMock(), {"task": "go to the kitchen"}
        )

    def _client(self, running, sent=True, canceled=(True, "ok")):
        client = MagicMock()
        client.goal_accepted = running
        client.action_returned = not running
        client.goal_rejected = False
        client.cancel_request.return_value = canceled
        client.send_request_from_dict.return_value = sent
        return client

    def test_an_idle_server_gets_the_goal_straight_away(
        self, rclpy_init, mock_model_client
    ):
        comp = _make_cortex([], mock_model_client, "test_cortex_dispatch_idle")
        client = self._client(running=False)

        result = self._dispatch(comp, client)

        client.cancel_request.assert_not_called()
        assert "dispatched" in result
        assert comp._active_action_clients[self.TOOL] is client

    def test_a_running_goal_is_canceled_before_the_new_one_is_sent(
        self, rclpy_init, mock_model_client
    ):
        comp = _make_cortex([], mock_model_client, "test_cortex_dispatch_replace")
        client = self._client(running=True)
        comp._active_action_clients[self.TOOL] = client
        order = MagicMock()
        order.attach_mock(client.cancel_request, "cancel")
        order.attach_mock(client.send_request_from_dict, "send")

        result = self._dispatch(comp, client)

        assert [c[0] for c in order.mock_calls] == ["cancel", "send"]
        assert "dispatched" in result
        assert comp._active_action_clients[self.TOOL] is client

    def test_a_goal_that_will_not_cancel_blocks_the_new_one(
        self, rclpy_init, mock_model_client
    ):
        comp = _make_cortex([], mock_model_client, "test_cortex_dispatch_stuck")
        client = self._client(running=True, canceled=(False, "Failed to cancel goal"))
        comp._active_action_clients[self.TOOL] = client

        result = self._dispatch(comp, client)

        client.send_request_from_dict.assert_not_called()
        assert result.startswith("Error:")
        assert "could not be canceled" in result
        # the running goal is still tracked
        assert comp._active_action_clients[self.TOOL] is client

    def test_a_rejection_is_reported_as_one(self, rclpy_init, mock_model_client):
        comp = _make_cortex([], mock_model_client, "test_cortex_dispatch_rejected")
        client = self._client(running=False, sent=False)
        client.goal_rejected = True

        result = self._dispatch(comp, client)

        assert result.startswith("Error:")
        assert "rejected" in result and "busy" in result
        assert self.TOOL not in comp._active_action_clients


class TestToolsFromTheRegistry:
    """Everything a component offers reaches Cortex through the action
    registry the Launcher built: described methods with their schema whole,
    action servers and services with their message fields"""

    def _register(self, comp, *entries):
        mock_component_internals(comp)
        comp._action_registry = _registry(*entries)
        comp._register_system_tools()

    def _tool(self, comp, name, phase="execution"):
        described = (
            comp._execution_tool_descriptions
            if phase == "execution"
            else comp._planning_tool_descriptions
        )
        return next(t["function"] for t in described if t["function"]["name"] == name)

    def test_a_described_method_is_a_tool_with_its_schema(
        self, rclpy_init, mock_model_client
    ):
        comp = _make_cortex([], mock_model_client, "test_cortex_registry_method")

        self._register(comp, _method("vision", "take_picture"))

        tool = self._tool(comp, "vision.take_picture")
        assert tool["description"] == "take_picture on vision"
        assert tool["parameters"]["required"] == ["topic"]
        assert comp._tool_refs["vision.take_picture"] == "vision/take_picture"

    def test_the_phase_in_the_schema_routes_the_tool(
        self, rclpy_init, mock_model_client
    ):
        comp = _make_cortex([], mock_model_client, "test_cortex_registry_phase")

        self._register(
            comp,
            _method("memory", "recall", phase="planning"),
            _method("memory", "body_status", phase="both"),
            _method("memory", "start_episode"),
        )

        assert "memory.recall" in comp._planning_tools
        assert "memory.recall" not in comp._execution_tools
        assert "memory.body_status" in comp._planning_tools
        assert "memory.body_status" in comp._execution_tools
        assert "memory.start_episode" not in comp._planning_tools
        assert "memory.start_episode" in comp._execution_tools
        # The phase is Cortex's business, not the model's
        assert "phase" not in self._tool(comp, "memory.recall", "planning")

    def test_what_is_not_offered(self, rclpy_init, mock_model_client):
        """Lifecycle methods, methods without a description, Cortex's own
        actions and the Monitor's methods are not tools"""
        comp = _make_cortex([], mock_model_client, "test_cortex_registry_filter")
        monitor_entry = RegisteredAction(
            ref=f"{MONITOR_OWNER}/start_routine",
            owner=MONITOR_OWNER,
            name="start_routine",
            kind=MONITOR_METHOD,
        )

        self._register(
            comp,
            _method("vision", "restart"),
            _method("vision", "undescribed", described=False),
            _method(comp.node_name, "cancel_main_goal"),
            (monitor_entry, None),
            _method("vision", "track"),
        )

        assert comp._tool_refs == {"vision.track": "vision/track"}

    def test_an_action_server_is_a_goal_tool(self, rclpy_init, mock_model_client):
        comp = _make_cortex([], mock_model_client, "test_cortex_registry_server")

        self._register(comp, _server("vla", "vla/run"))

        tool = self._tool(comp, "send_goal_to_vla_run")
        assert "task" in tool["parameters"]["properties"]
        assert "'vla' component's action server" in tool["description"]
        assert comp._tool_refs["send_goal_to_vla_run"] == "vla/run"

    def test_a_service_is_a_request_tool(self, rclpy_init, mock_model_client):
        """Any service the registry lists, a component's main one included"""
        comp = _make_cortex([], mock_model_client, "test_cortex_registry_service")

        self._register(
            comp, _server("planner", "save_plan", COMPONENT_SERVICE, SetBool)
        )

        tool = self._tool(comp, "send_request_to_planner_save_plan")
        assert "data" in tool["parameters"]["properties"]
        assert (
            comp._tool_refs["send_request_to_planner_save_plan"] == "planner/save_plan"
        )

    def test_two_components_with_one_server_name_get_two_tools(
        self, rclpy_init, mock_model_client
    ):
        """A server tool is named from its registry reference, owner and
        server, so a bare server name two components share cannot collide"""
        comp = _make_cortex([], mock_model_client, "test_cortex_registry_shared")

        self._register(comp, _server("planner", "run"), _server("vla", "run"))

        assert comp._tool_refs["send_goal_to_planner_run"] == "planner/run"
        assert comp._tool_refs["send_goal_to_vla_run"] == "vla/run"

    def test_registering_twice_registers_once(self, rclpy_init, mock_model_client):
        """A component activated again must not offer every tool twice"""
        comp = _make_cortex([], mock_model_client, "test_cortex_registry_twice")

        self._register(comp, _server("vla", "vla/run"), _method("vision", "track"))
        comp._register_system_tools()

        names = [t["function"]["name"] for t in comp._execution_tool_descriptions]
        assert names.count("send_goal_to_vla_run") == 1
        assert names.count("vision.track") == 1

    def test_a_goal_tool_is_dispatched_by_its_entry(
        self, rclpy_init, mock_model_client
    ):
        comp = _make_cortex([], mock_model_client, "test_cortex_registry_goal")
        self._register(comp, _server("vla", "vla/run"))
        comp._send_action_goal_from_dict = MagicMock(return_value="sent")

        result = comp._execute_system_tool("send_goal_to_vla_run", {"task": "go"})

        assert result == "sent"
        comp._send_action_goal_from_dict.assert_called_once_with(
            "send_goal_to_vla_run",
            "vla",
            "vla/run",
            VisionLanguageAction,
            {"task": "go"},
        )

    def test_a_request_tool_is_dispatched_by_its_entry(
        self, rclpy_init, mock_model_client
    ):
        comp = _make_cortex([], mock_model_client, "test_cortex_registry_request")
        self._register(
            comp, _server("planner", "save_plan", COMPONENT_SERVICE, SetBool)
        )
        comp._send_service_request_from_dict = MagicMock(return_value="done")

        result = comp._execute_system_tool(
            "send_request_to_planner_save_plan", {"data": True}
        )

        assert result == "done"
        comp._send_service_request_from_dict.assert_called_once_with(
            "planner", "save_plan", SetBool, {"data": True}
        )


class TestThePhaseTravelsWithTheDescription:
    """The wrapper writes the phase into the description the decorator
    stores, so it reaches Cortex with the schema through the registry"""

    def test_a_given_phase_and_the_default(self):
        described = {
            "type": "function",
            "function": {"name": "probe", "description": "Probe", "parameters": {}},
        }

        class Probe:
            @component_action(description=described, phase=ActionPhase.PLANNING)
            def probe(self) -> ActionReturnType:
                return True, "probed"

            @component_action(description=described)
            def act(self) -> ActionReturnType:
                return True, "acted"

        probe = json.loads(Probe.probe._action_description)
        act = json.loads(Probe.act._action_description)
        assert probe["phase"] == "planning"
        assert act["phase"] == "execution"
        # The description itself is untouched
        assert probe["function"] == described["function"]


class TestCallingAComponentAction:
    """Cortex reaches a component's actions through the Monitor, which reads
    the service response into the (success, message) action contract. What
    the LLM sees as the tool result is that message, or an error line"""

    def _call(self, comp, returns=None, raises=None):
        """Through the real resolver, down to the method service call"""
        mock_component_internals(comp)
        comp._action_registry = _registry(_method("memory", "start_episode"))
        comp._tool_refs["memory.start_episode"] = "memory/start_episode"
        comp._execute_component_method_srv_client = {"memory": MagicMock()}
        comp.execute_component_method = MagicMock(
            return_value=returns, side_effect=raises
        )
        return comp._call_component_action("memory.start_episode", {"name": "tidy"})

    def test_the_actions_message_is_the_tool_result(
        self, rclpy_init, mock_model_client
    ):
        comp = _make_cortex([], mock_model_client, "test_cortex_call_message")

        result = self._call(comp, (True, "Episode 'tidy' started"))

        called = comp.execute_component_method.call_args.args
        assert called[:3] == ("memory", "start_episode", {"name": "tidy"})
        assert result == "Episode 'tidy' started"

    def test_a_failure_is_an_error_line(self, rclpy_init, mock_model_client):
        comp = _make_cortex([], mock_model_client, "test_cortex_call_failure")

        result = self._call(comp, (False, "no such layer"))

        assert result.startswith("Error:")
        assert "no such layer" in result

    def test_an_empty_message_is_a_confirmation(self, rclpy_init, mock_model_client):
        comp = _make_cortex([], mock_model_client, "test_cortex_call_empty")

        assert "executed successfully" in self._call(comp, (True, ""))

    def test_a_raised_error_is_reported_not_raised(self, rclpy_init, mock_model_client):
        comp = _make_cortex([], mock_model_client, "test_cortex_call_raises")

        result = self._call(comp, raises=KeyError("memory"))

        assert result.startswith("Error calling memory.start_episode")

    def test_a_tool_the_registry_does_not_know_is_refused(
        self, rclpy_init, mock_model_client
    ):
        comp = _make_cortex([], mock_model_client, "test_cortex_call_unknown")
        mock_component_internals(comp)

        result = comp._call_component_action("ghost.action", {})

        assert result.startswith("Error: Unknown tool")

    def test_the_parameter_tool_reads_the_contract_too(
        self, rclpy_init, mock_model_client
    ):
        comp = _make_cortex([], mock_model_client, "test_cortex_call_parameter")
        mock_component_internals(comp)
        comp.update_parameter = MagicMock(return_value=(True, "updated"))
        args = {"component": "vision", "param_name": "threshold", "new_value": "0.5"}

        assert "executed successfully" in comp._execute_system_tool(
            "update_parameter", args
        )
        comp.update_parameter.assert_called_once_with("vision", "threshold", "0.5")

        comp.update_parameter.return_value = (False, "unknown parameter")
        result = comp._execute_system_tool("update_parameter", args)
        assert result.startswith("Error:") and "unknown parameter" in result

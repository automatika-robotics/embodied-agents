"""Tests for Cortex component — requires rclpy."""

import json
import threading
import time
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
    BaseComponent,
    Topic,
    Action,
    ComponentRunType,
    RegisteredAction,
    Routine,
    RoutineStatus,
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
    """A plugin's actions reach Cortex through the action registry like a
    component's: named by the plugin id, built by the plugin's own factory
    from the LLM's arguments, and run in Cortex's process"""

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
        comp._action_registry = SystemActionRegistry.from_components(
            [], plugins=plugins
        )
        comp.get_routines = MagicMock(return_value=[])
        comp._register_system_tools()
        return comp

    def _tool(self, comp, name):
        return next(
            t["function"]
            for t in comp._execution_tool_descriptions
            if t["function"]["name"] == name
        )

    def test_tools_are_named_by_plugin_id(self, rclpy_init, mock_model_client):
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

    def test_the_tool_carries_the_plugins_own_schema(
        self, rclpy_init, mock_model_client
    ):
        """Described with a dict or with a string, every plugin action has a
        whole schema in the registry, and a name a lifecycle method shares is
        still the plugin's own"""
        comp = self._cortex(
            mock_model_client, "test_cortex_plugin_schema", _PtzCamera(id="front_cam")
        )

        look_at = self._tool(comp, "front_cam.look_at")
        assert look_at["parameters"]["required"] == ["pan_deg", "tilt_deg"]
        stop = self._tool(comp, "front_cam.stop")
        assert stop["description"] == "Stop moving the camera."
        assert stop["parameters"]["properties"] == {}

    def test_the_llms_arguments_reach_the_action(self, rclpy_init, mock_model_client):
        camera = _PtzCamera(id="front_cam")
        comp = self._cortex(mock_model_client, "test_cortex_plugin_args", camera)

        # OpenAI-compatible endpoints send arguments as a JSON string
        result = comp._execute_action_step(
            _tool_call("front_cam.look_at", '{"pan_deg": 90, "tilt_deg": 30}')
        )

        assert camera.aimed == [(90, 30)]
        assert result == "aimed at pan 90, tilt 30"

    def test_a_call_missing_a_required_argument_is_refused(
        self, rclpy_init, mock_model_client
    ):
        """The factory would default it silently; the tool said it is required"""
        camera = _PtzCamera(id="front_cam")
        comp = self._cortex(mock_model_client, "test_cortex_plugin_missing", camera)

        result = comp._execute_action_step(
            _tool_call("front_cam.look_at", {"pan_deg": 90})
        )

        assert result.startswith("Error:") and "missing required" in result
        assert camera.aimed == []

    def test_undeclared_arguments_are_ignored(self, rclpy_init, mock_model_client):
        """An argument the tool never declared would reach the factory as the
        Action's own keyword argument"""
        camera = _PtzCamera(id="front_cam")
        comp = self._cortex(mock_model_client, "test_cortex_plugin_extra", camera)

        result = comp._execute_action_step(
            _tool_call("front_cam.look_at", {"pan_deg": 1, "tilt_deg": 2, "speed": 9})
        )

        assert camera.aimed == [(1, 2)]
        assert result == "aimed at pan 1, tilt 2"
        comp.get_logger().warning.assert_called()

    def test_a_failed_action_is_an_error_result(self, rclpy_init, mock_model_client):
        camera = _PtzCamera(id="front_cam")
        camera.refuse = True
        comp = self._cortex(mock_model_client, "test_cortex_plugin_failed", camera)

        result = comp._execute_action_step(
            _tool_call("front_cam.look_at", {"pan_deg": 0, "tilt_deg": 0})
        )

        assert result.startswith("Error:") and "refused" in result


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
    plugin.id = name.lower()
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
        # The way out is named: the planner can stop a goal it did not send
        assert "vla.cancel_main_goal" in result
        assert self.TOOL not in comp._active_action_clients


class TestTheStopTool:
    """Every component running an action server inherits cancel_main_goal,
    described by sugarcoat, so the registry walk offers it as a tool"""

    def test_action_server_components_get_it_and_others_do_not(
        self, rclpy_init, mock_model_client
    ):
        comp = _make_cortex([], mock_model_client, "test_cortex_stop_tool")
        arm = BaseComponent(
            component_name="test_cortex_stop_arm", main_action_type=VisionLanguageAction
        )
        arm.run_type = ComponentRunType.ACTION_SERVER
        worker = BaseComponent(component_name="test_cortex_stop_worker")
        mock_component_internals(comp)
        comp._action_registry = SystemActionRegistry.from_components([arm, worker])
        comp.get_routines = MagicMock(return_value=[])

        comp._register_system_tools()

        assert f"{arm.node_name}.cancel_main_goal" in comp._execution_tools
        assert f"{worker.node_name}.cancel_main_goal" not in comp._execution_tools
        tool = next(
            t["function"]
            for t in comp._execution_tool_descriptions
            if t["function"]["name"] == f"{arm.node_name}.cancel_main_goal"
        )
        assert tool["description"].startswith("Stop the goal")
        assert tool["parameters"]["properties"] == {}


def _noop(**_):
    return True, ""


class TestRoutinesAsTools:
    """Routines the Monitor hosts are skills for the planner: a start tool
    each, in the recipe author's words, plus pause, resume and abort"""

    CURSOR = {
        "name": "pick_object",
        "description": "Pick the object in front.",
        "steps": ["detect", "grasp"],
        "status": RoutineStatus.RUNNING,
        "index": 1,
        "active_step": "grasp",
        "step_message": "found it",
        "abort_reason": "",
        "elapsed": 4.3,
    }

    def _cortex(self, mock_model_client, name, cursors):
        comp = _make_cortex([], mock_model_client, name)
        mock_component_internals(comp)
        comp._action_registry = _registry()
        comp.get_routines = MagicMock(return_value=cursors)
        return comp

    def test_a_routine_given_to_cortex_is_hosted(self, rclpy_init, mock_model_client):
        routine = Routine("tidy", steps=[Action(_noop)], description="Tidy")
        comp = _make_cortex(
            [], mock_model_client, "test_cortex_routine_hosted", routines=[routine]
        )
        comp.host_routines = MagicMock()

        comp._init_internal_monitor(components_names=[])

        comp.host_routines.assert_called_once_with([routine])

    def test_a_routine_without_a_description_is_refused(
        self, rclpy_init, mock_model_client
    ):
        routine = Routine("tidy", steps=[Action(_noop)])

        with pytest.raises(ValueError, match="description"):
            _make_cortex(
                [],
                mock_model_client,
                "test_cortex_routine_undescribed",
                routines=[routine],
            )

    def test_each_hosted_routine_is_a_start_tool(self, rclpy_init, mock_model_client):
        comp = self._cortex(
            mock_model_client,
            "test_cortex_routine_tool",
            [dict(self.CURSOR, status=RoutineStatus.IDLE)],
        )

        comp._register_system_tools()

        tool = next(
            t["function"]
            for t in comp._execution_tool_descriptions
            if t["function"]["name"] == "routine.pick_object"
        )
        assert "Pick the object in front. Steps: detect, grasp." in tool["description"]
        assert tool["parameters"]["properties"] == {}
        for control in ("pause_routine", "resume_routine", "abort_routine"):
            assert control in comp._execution_tools

    def test_no_routines_means_no_control_tools(self, rclpy_init, mock_model_client):
        comp = self._cortex(mock_model_client, "test_cortex_routine_none", [])

        comp._register_system_tools()

        assert not comp._routine_tools
        assert "pause_routine" not in comp._execution_tools

    def test_starting_a_routine_tracks_it(self, rclpy_init, mock_model_client):
        comp = self._cortex(
            mock_model_client, "test_cortex_routine_start", [self.CURSOR]
        )
        comp._register_system_tools()
        comp.start_routine = MagicMock(return_value=(True, "Routine started"))

        result = comp._execute_system_tool("routine.pick_object", {})

        comp.start_routine.assert_called_once_with("pick_object")
        assert "started" in result
        assert comp._active_routines == {"pick_object"}

    def test_a_routine_that_will_not_start_is_an_error(
        self, rclpy_init, mock_model_client
    ):
        comp = self._cortex(
            mock_model_client, "test_cortex_routine_busy", [self.CURSOR]
        )
        comp._register_system_tools()
        comp.start_routine = MagicMock(return_value=(False, "already running"))

        result = comp._execute_system_tool("routine.pick_object", {})

        assert result.startswith("Error:") and "already running" in result
        assert not comp._active_routines

    def test_control_tools_reach_the_monitor(self, rclpy_init, mock_model_client):
        comp = self._cortex(
            mock_model_client, "test_cortex_routine_pause", [self.CURSOR]
        )
        comp._register_system_tools()
        comp.pause_routine = MagicMock(return_value=(True, "paused at grasp"))

        result = comp._execute_system_tool(
            "pause_routine", {"routine_name": "pick_object"}
        )

        comp.pause_routine.assert_called_once_with("pick_object")
        assert result == "paused at grasp"

    def test_the_status_block_follows_a_routine_to_its_end(
        self, rclpy_init, mock_model_client
    ):
        comp = self._cortex(
            mock_model_client, "test_cortex_routine_status", [self.CURSOR]
        )
        comp._active_routines = {"pick_object"}

        status = comp._monitor_active_clients()

        assert "routine.pick_object: running at step 'grasp'" in status
        assert "found it" in status
        assert comp._active_routines == {"pick_object"}

        comp.get_routines.return_value = [
            dict(self.CURSOR, status=RoutineStatus.COMPLETED, step_message="lifted")
        ]
        status = comp._monitor_active_clients()

        assert "routine.pick_object: COMPLETED | lifted" in status
        assert not comp._active_routines
        assert comp._monitor_active_clients() is None

    def test_ending_the_task_aborts_its_routines(self, rclpy_init, mock_model_client):
        comp = self._cortex(
            mock_model_client, "test_cortex_routine_abort", [self.CURSOR]
        )
        comp._active_routines = {"pick_object"}
        comp.abort_routine = MagicMock(return_value=(True, "aborted"))

        comp._cancel_all_active_clients()

        assert comp.abort_routine.call_args.args == ("pick_object",)
        assert not comp._active_routines

    def test_a_real_routine_runs_and_is_reported(self, rclpy_init, mock_model_client):
        """In process, through the Monitor: hosted, offered, started by its
        tool and followed to completion"""
        comp = _make_cortex([], mock_model_client, "test_cortex_routine_real")
        mock_component_internals(comp)
        comp._init_internal_monitor(components_names=[])
        comp.create_publisher = MagicMock()
        grasped = threading.Event()

        def detect(**_):
            return True, "detected"

        def grasp(**_):
            grasped.set()
            return True, "grasped"

        routine = Routine(
            "pick", steps=[Action(detect), Action(grasp)], description="Pick it up"
        )
        assert comp.add_routine(routine)[0]
        comp._register_system_tools()

        assert "started" in comp._execute_system_tool("routine.pick", {})
        assert grasped.wait(2.0)
        deadline = time.monotonic() + 2.0
        status = comp._monitor_active_clients()
        while comp._active_routines and time.monotonic() < deadline:
            time.sleep(0.02)
            status = comp._monitor_active_clients()

        assert "routine.pick: COMPLETED | grasped" in status
        assert not comp._active_routines


def _step(name, **args):
    """A planned tool call"""
    return {"function": {"name": name, "arguments": args}}


class _Handle:
    """A task goal handle whose cancellation arrives after a few reads"""

    def __init__(self, cancel_after=None):
        self.reads = 0
        self.cancel_after = cancel_after
        self.publish_feedback = MagicMock()
        self.canceled = MagicMock()

    @property
    def is_cancel_requested(self):
        self.reads += 1
        return self.cancel_after is not None and self.reads > self.cancel_after


class TestCompilingPlans:
    """A run of plan steps whose arguments are all known becomes one routine
    hosted by the Monitor, with no confirmation call between its steps"""

    def _cortex(self, mock_model_client, name):
        comp = _make_cortex([], mock_model_client, name)
        mock_component_internals(comp)
        comp.config.monitoring_interval = 0.01
        # The Monitor side: the registry with the Monitor's own actions, and
        # the routine machinery
        comp._init_internal_monitor(components_names=[])
        comp._action_registry.add(*_method("vision", "take_picture"))
        comp._action_registry.add(*_server("vla", "vla/run"))
        comp._action_registry.add(
            *_server("planner", "save_plan", COMPONENT_SERVICE, SetBool)
        )
        comp.create_publisher = MagicMock()
        comp.destroy_publisher = MagicMock()
        # The Monitor's wait checks the node context, which an unstarted node
        # does not have
        comp._context = MagicMock()
        comp._register_system_tools()
        # The stepwise engine, so a test sees when it is used
        comp._wait_for_active_clients = MagicMock(return_value=("EXECUTE", None))
        comp._execute_action_step = MagicMock(return_value="ran stepwise")
        return comp

    def _picture_taken(self, comp, returns=(True, "saved to /tmp/x.png")):
        comp._execute_component_method_srv_client = {"vision": MagicMock()}
        comp.execute_component_method = MagicMock(return_value=returns)

    def test_which_steps_compile(self, rclpy_init, mock_model_client):
        comp = self._cortex(mock_model_client, "test_cortex_compile_which")
        ref = comp._compiled_ref

        assert (
            ref(_step("vision.take_picture", topic_name="/cam"))
            == "vision/take_picture"
        )
        assert ref(_step("send_goal_to_vla_run", task="go")) == "vla/run"
        assert (
            ref(_step("send_goal_to_vla_run", task="go", wait_to_finish=True))
            == "vla/run"
        )
        assert ref(_step("wait", duration=2)) == "monitor/wait"
        # A goal the planner does not wait for runs alongside what follows
        assert (
            ref(_step("send_goal_to_vla_run", task="go", wait_to_finish=False)) is None
        )
        assert (
            ref(_step("send_goal_to_vla_run", task="go", wait_to_finish="false"))
            is None
        )
        # Services, Cortex's other tools and unknown tools run step by step
        assert ref(_step("send_request_to_planner_save_plan", data=True)) is None
        assert ref(_step("update_parameter", component="vision")) is None
        assert ref(_step("tts.say", text="hi")) is None
        # So does anything waiting for an earlier result, however deep
        assert (
            ref(_step("vision.take_picture", topic_name="<output from step 1>")) is None
        )
        assert (
            ref(
                _step(
                    "vision.take_picture", save_path={"dir": ["<output from step 2>"]}
                )
            )
            is None
        )
        as_json = {
            "function": {
                "name": "vision.take_picture",
                "arguments": '{"topic_name": "<output from step 1>"}',
            }
        }
        assert ref(as_json) is None

    def test_a_run_breaks_at_a_placeholder_or_a_stepwise_tool(
        self, rclpy_init, mock_model_client
    ):
        comp = self._cortex(mock_model_client, "test_cortex_compile_runs")
        plan = [
            _step("vision.take_picture", topic_name="/cam"),
            _step("send_goal_to_vla_run", task="go"),
            _step("vision.take_picture", topic_name="<output from step 1>"),
            _step("vision.take_picture", topic_name="/cam2"),
            _step("send_request_to_planner_save_plan", data=True),
        ]

        assert comp._compilable_run(plan, 0) == {
            0: "vision/take_picture",
            1: "vla/run",
        }
        assert comp._compilable_run(plan, 2) == {}
        # A lone component action is not worth a routine, a lone goal is
        assert comp._compilable_run(plan, 3) == {}
        assert comp._compilable_run(plan, 4) == {}
        goal = _step("send_goal_to_vla_run", task="go")
        assert comp._compilable_run([goal], 0) == {0: "vla/run"}
        # Decided once for the whole plan, keyed by where each run starts
        assert comp._compilable_runs(plan) == {
            0: {0: "vision/take_picture", 1: "vla/run"},
        }
        assert comp._compilable_runs(plan + [goal]) == {
            0: {0: "vision/take_picture", 1: "vla/run"},
            5: {5: "vla/run"},
        }
        concurrent = [
            _step("send_goal_to_vla_run", task="go", wait_to_finish=False),
            _step("vision.take_picture", topic_name="/cam"),
            _step("vision.take_picture", topic_name="/cam2"),
        ]
        assert comp._compilable_run(concurrent, 0) == {}
        assert comp._compilable_run(concurrent, 1) == {
            1: "vision/take_picture",
            2: "vision/take_picture",
        }
        comp.config.compile_routines = False
        assert comp._compilable_run(plan, 0) == {}

    def test_the_spec_of_a_run(self, rclpy_init, mock_model_client):
        comp = self._cortex(mock_model_client, "test_cortex_compile_spec")
        plan = [
            _step("vision.take_picture", topic_name="/cam"),
            _step("send_goal_to_vla_run", task="go", wait_to_finish=True),
            _step("wait", duration=1),
        ]

        spec = comp._routine_spec(
            plan, {0: "vision/take_picture", 1: "vla/run", 2: "monitor/wait"}
        )

        assert spec["name"].startswith("cortex_plan_")
        assert "1 to 3" in spec["description"]
        assert spec["steps"][0] == {
            "ref": "vision/take_picture",
            "name": "1_vision.take_picture",
            "kwargs": {"topic_name": "/cam"},
            "timeout": 60.0,
            "on_timeout": "fail",
        }
        assert spec["steps"][1] == {
            "ref": "vla/run",
            "name": "2_send_goal_to_vla_run",
            "goal": {"task": "go"},
        }
        assert spec["steps"][2] == {
            "ref": "monitor/wait",
            "name": "3_wait",
            "kwargs": {"duration": 1},
        }

    def test_a_compiled_run_executes_and_reports_each_step(
        self, rclpy_init, mock_model_client
    ):
        """End to end, in process: built through the Monitor, run by it,
        each step's message the planner's result, the routine gone after"""
        comp = self._cortex(mock_model_client, "test_cortex_compile_run")
        self._picture_taken(comp)
        handle = _Handle()
        plan = [
            _step("vision.take_picture", topic_name="/cam"),
            _step("wait", duration=0.05),
        ]

        results, aborted = comp._execute_plan(plan, handle, MagicMock())

        assert not aborted
        assert [r["result"] for r in results] == ["saved to /tmp/x.png", "Waited 0.05s"]
        assert not any(r["failed"] for r in results)
        called = comp.execute_component_method.call_args.args
        assert called[:3] == ("vision", "take_picture", {"topic_name": "/cam"})
        comp._wait_for_active_clients.assert_not_called()
        assert comp.get_routines() == []
        assert handle.publish_feedback.called

    def test_a_failed_step_ends_the_run_and_returns_to_planning(
        self, rclpy_init, mock_model_client
    ):
        comp = self._cortex(mock_model_client, "test_cortex_compile_failed")
        self._picture_taken(comp, returns=(False, "no camera"))
        plan = [
            _step("vision.take_picture", topic_name="/cam"),
            _step("vision.take_picture", topic_name="/cam2"),
            _step("vision.take_picture", topic_name="<output from step 2>"),
        ]

        results, aborted = comp._execute_plan(plan, _Handle(), MagicMock())

        assert not aborted
        assert len(results) == len(plan)
        assert results[0]["failed"] and "no camera" in results[0]["result"]
        assert results[1]["result"] == "NOT RUN: an earlier step failed"
        assert results[2]["result"] == "NOT RUN: an earlier step failed"
        comp._execute_action_step.assert_not_called()

    def test_cancelling_the_task_aborts_the_routine(
        self, rclpy_init, mock_model_client
    ):
        comp = self._cortex(mock_model_client, "test_cortex_compile_cancel")
        self._picture_taken(comp)
        handle = _Handle(cancel_after=2)
        started = time.monotonic()
        plan = [
            _step("vision.take_picture", topic_name="/cam"),
            _step("wait", duration=30),
        ]

        results, aborted = comp._execute_plan(plan, handle, MagicMock())

        assert aborted
        assert time.monotonic() - started < 5
        handle.canceled.assert_called_once()
        assert results[1]["result"].startswith("NOT RUN: the routine was aborted")
        assert comp.get_routines() == []

    def test_a_run_that_cannot_be_built_runs_step_by_step(
        self, rclpy_init, mock_model_client
    ):
        """A goal with a field its message does not have is refused when the
        routine is built, so the step falls back to the stepwise engine"""
        comp = self._cortex(mock_model_client, "test_cortex_compile_fallback")

        plan = [
            _step("vision.take_picture", topic_name="/cam"),
            _step("send_goal_to_vla_run", nope=1),
        ]

        results, aborted = comp._execute_plan(plan, _Handle(), MagicMock())

        assert not aborted
        assert [r["result"] for r in results] == ["ran stepwise"] * 2
        # Once for the run, not again from each of its remaining steps
        comp.get_logger().warning.assert_called_once()
        assert comp.get_routines() == []

    def test_a_single_compilable_step_runs_directly(
        self, rclpy_init, mock_model_client
    ):
        """A routine of one step would only add overhead"""
        comp = self._cortex(mock_model_client, "test_cortex_compile_single")
        comp.add_routine = MagicMock()

        results, _ = comp._execute_plan(
            [_step("vision.take_picture", topic_name="/cam")], _Handle(), MagicMock()
        )

        assert results[0]["result"] == "ran stepwise"
        comp.add_routine.assert_not_called()

    def test_compilation_can_be_switched_off(self, rclpy_init, mock_model_client):
        comp = self._cortex(mock_model_client, "test_cortex_compile_off")
        comp.config.compile_routines = False
        comp.add_routine = MagicMock()

        results, _ = comp._execute_plan(
            [_step("vision.take_picture", topic_name="/cam")], _Handle(), MagicMock()
        )

        assert results[0]["result"] == "ran stepwise"
        comp.add_routine.assert_not_called()

    def test_a_lone_awaited_goal_is_compiled(self, rclpy_init, mock_model_client):
        comp = self._cortex(mock_model_client, "test_cortex_compile_lone_goal")
        comp._execute_compiled = MagicMock(return_value=(1, None))

        comp._execute_plan(
            [_step("send_goal_to_vla_run", task="go")], _Handle(), MagicMock()
        )

        comp._execute_compiled.assert_called_once()
        comp._execute_action_step.assert_not_called()

    def test_a_goal_not_waited_for_runs_as_before(self, rclpy_init, mock_model_client):
        comp = self._cortex(mock_model_client, "test_cortex_compile_async_goal")
        comp._execute_compiled = MagicMock()

        results, _ = comp._execute_plan(
            [_step("send_goal_to_vla_run", task="go", wait_to_finish=False)],
            _Handle(),
            MagicMock(),
        )

        comp._execute_compiled.assert_not_called()
        assert results[0]["result"] == "ran stepwise"

    def test_the_prompt_mandates_the_placeholder_spelling(self):
        assert '"<output from step N>"' in Cortex._PLANNING_PROMPT
        assert "wait_to_finish=false" in Cortex._PLANNING_PROMPT


class _Watched:
    """A managed component with topics, as inspection and events see it"""

    def __init__(self, name):
        self.node_name = name
        self.in_topics = []
        self.out_topics = [Topic(name="detections", msg_type="Detections")]

    def inspect_component(self):
        return f"Component {self.node_name}"


class TestEventsAsTools:
    """A standing instruction becomes a runtime event on the Monitor: a
    condition on a topic field the planner has seen in inspection, and
    actions by registry reference"""

    PERSON = {
        "topic": "detections",
        "field": "labels",
        "operator": "contains",
        "value": "person",
    }
    TAKE = {"tool": "vision.take_picture", "arguments": {"topic_name": "/cam"}}

    def _cortex(self, mock_model_client, name):
        comp = _make_cortex([], mock_model_client, name)
        mock_component_internals(comp)
        comp._init_internal_monitor(
            components_names=[], components=[_Watched("vision")]
        )
        comp._action_registry.add(*_method("vision", "take_picture"))
        comp._action_registry.add(*_server("vla", "vla/run"))
        comp.create_subscription = MagicMock()
        comp.create_publisher = MagicMock()
        # The event lists a Monitor creates when it activates, which needs a
        # live node. A runtime event is added to them
        comp._Monitor__events = []
        comp._Monitor__event_listeners = {}
        comp.config.enable_events = True
        comp._register_system_tools()
        comp._routine_tools["routine.patrol"] = "patrol"
        return comp

    def _add(self, comp, **overrides):
        call = {
            "event_id": "spot_person",
            "conditions": [self.PERSON],
            "actions": [self.TAKE],
        }
        call.update(overrides)
        return comp._execute_system_tool("add_event", call)

    def test_inspection_lists_the_fields_of_each_topic(
        self, rclpy_init, mock_model_client
    ):
        comp = self._cortex(mock_model_client, "test_cortex_events_inspect")

        text = comp._inspect_component("vision")

        assert "Topic fields, for event conditions:" in text
        assert "detections (Detections):" in text
        assert "labels: sequence<string>" in text
        assert "header.frame_id: string" in text

    def test_the_tools_are_registered(self, rclpy_init, mock_model_client):
        comp = self._cortex(mock_model_client, "test_cortex_events_tools")

        assert {"add_event", "remove_event", "list_events"} <= comp._execution_tools
        assert "list_events" in comp._planning_tools
        assert "add_event" not in comp._planning_tools

    def test_an_event_is_installed_listed_and_removed(
        self, rclpy_init, mock_model_client
    ):
        """Through the Monitor, in process"""
        comp = self._cortex(mock_model_client, "test_cortex_events_round_trip")

        result = self._add(comp)

        assert not result.startswith("Error"), result
        assert "spot_person" in json.loads(
            comp._execute_planning_tool("list_events", {})
        )
        removed = comp._execute_system_tool("remove_event", {"event_id": "spot_person"})
        assert not removed.startswith("Error"), removed
        assert json.loads(comp._execute_system_tool("list_events", {})) == {}

    def test_the_condition_is_written_as_sugarcoat_reads_it(
        self, rclpy_init, mock_model_client
    ):
        comp = self._cortex(mock_model_client, "test_cortex_events_condition")

        event, actions = comp._event_spec({
            "event_id": "e",
            "conditions": [self.PERSON],
            "actions": [self.TAKE],
        })

        condition = event["condition"]
        assert condition["type"] == "simple"
        assert condition["topic_name"] == "detections"
        assert condition["topic_msg_type"] == "Detections"
        assert condition["attribute_path"] == ["labels"]
        assert condition["operator"] == "contains"
        assert condition["ref_value"] == "person"
        assert "topic_qos_config" in condition
        assert event["name"] == "e"
        assert event["handle_once"] is True and event["on_change"] is False
        assert actions == [
            {"ref": "vision/take_picture", "kwargs": {"topic_name": "/cam"}}
        ]

    def test_several_conditions_are_grouped(self, rclpy_init, mock_model_client):
        comp = self._cortex(mock_model_client, "test_cortex_events_group")
        frame = {
            "topic": "detections",
            "field": "header.frame_id",
            "operator": "equals",
            "value": "cam",
        }
        call = {
            "event_id": "e",
            "conditions": [self.PERSON, frame],
            "actions": [self.TAKE],
        }

        event, _ = comp._event_spec({**call, "match": "any"})
        assert event["condition"]["type"] == "composite"
        assert event["condition"]["logic_operator"] == 2
        assert len(event["condition"]["sub_conditions"]) == 2

        event, _ = comp._event_spec(call)
        assert event["condition"]["logic_operator"] == 1

    def test_no_field_means_any_message(self, rclpy_init, mock_model_client):
        comp = self._cortex(mock_model_client, "test_cortex_events_any")

        event, _ = comp._event_spec({
            "event_id": "e",
            "conditions": [{"topic": "detections"}],
            "actions": [self.TAKE],
        })

        assert event["condition"]["attribute_path"] == []
        assert event["condition"]["operator"] == "none"
        assert event["condition"]["ref_value"] is None

    def test_a_kept_event_fires_on_the_transition(self, rclpy_init, mock_model_client):
        comp = self._cortex(mock_model_client, "test_cortex_events_kept")

        event, _ = comp._event_spec({
            "event_id": "e",
            "conditions": [self.PERSON],
            "actions": [self.TAKE],
            "once": False,
        })

        assert event["handle_once"] is False and event["on_change"] is True

    def test_what_an_event_may_run(self, rclpy_init, mock_model_client):
        comp = self._cortex(mock_model_client, "test_cortex_events_actions")

        _, specs = comp._event_spec({
            "event_id": "e",
            "conditions": [self.PERSON],
            "actions": [
                {
                    "tool": "send_goal_to_vla_run",
                    "arguments": {"task": "go", "wait_to_finish": True},
                },
                {"tool": "routine.patrol"},
                {"tool": "abort_routine", "arguments": {"routine_name": "patrol"}},
            ],
        })

        assert specs == [
            {"ref": "vla/run", "goal": {"task": "go"}},
            {"ref": "monitor/start_routine", "kwargs": {"routine_name": "patrol"}},
            {"ref": "monitor/abort_routine", "kwargs": {"routine_name": "patrol"}},
        ]

    def test_what_is_refused(self, rclpy_init, mock_model_client):
        comp = self._cortex(mock_model_client, "test_cortex_events_refused")
        bad_conditions = [
            ({"topic": "ghost"}, "Unknown topic"),
            (
                {
                    "topic": "detections",
                    "field": "colour",
                    "operator": "equals",
                    "value": 1,
                },
                "has no field",
            ),
            (
                {"topic": "detections", "field": "labels", "value": "x"},
                "needs an operator",
            ),
            (
                {
                    "topic": "detections",
                    "field": "labels",
                    "operator": "near",
                    "value": "x",
                },
                "needs an operator",
            ),
        ]
        for condition, why in bad_conditions:
            result = self._add(comp, conditions=[condition])
            assert result.startswith("Error:") and why in result, result
        for tool in ("wait", "inspect_component", "add_event", "unknown.tool"):
            result = self._add(comp, actions=[{"tool": tool}])
            assert result.startswith("Error:") and "cannot run" in result, result
        assert self._add(comp, event_id="").startswith("Error:")
        assert self._add(comp, conditions=[]).startswith("Error:")
        assert self._add(comp, actions=[]).startswith("Error:")
        assert json.loads(comp._execute_system_tool("list_events", {})) == {}

    def test_off_by_default(self, rclpy_init, mock_model_client):
        """Neither the tools nor the prompt guidance, unless the recipe asks"""
        comp = _make_cortex([], mock_model_client, "test_cortex_events_off")
        mock_component_internals(comp)
        comp._action_registry = _registry()
        comp.get_routines = MagicMock(return_value=[])

        comp._managed_components = {"vision": _Watched("vision")}
        comp._register_system_tools()
        comp._compose_planning_prompt()

        assert not {"add_event", "remove_event", "list_events"} & comp._execution_tools
        assert "add_event" not in comp._effective_planning_prompt
        assert "Topic fields" not in comp._inspect_component("vision")

    def test_the_prompt_points_at_events_when_enabled(
        self, rclpy_init, mock_model_client
    ):
        comp = self._cortex(mock_model_client, "test_cortex_events_prompt")

        comp._compose_planning_prompt()

        assert "=== Standing Instructions ===" in comp._effective_planning_prompt
        assert "add_event" in comp._effective_planning_prompt
        assert "add_event" not in Cortex._PLANNING_PROMPT


class TestTheWaitTool:
    """The planner has no other way to dwell. The wait runs in slices, so a
    cancelled task does not sit it out"""

    def _cortex(self, mock_model_client, name):
        comp = _make_cortex([], mock_model_client, name)
        mock_component_internals(comp)
        comp.config.monitoring_interval = 0.01
        comp.get_routines = MagicMock(return_value=[])
        return comp

    def test_it_is_an_execution_tool(self, rclpy_init, mock_model_client):
        comp = self._cortex(mock_model_client, "test_cortex_wait_tool")
        comp._action_registry = _registry()

        comp._register_system_tools()

        assert "wait" in comp._execution_tools

    def test_it_waits_the_duration(self, rclpy_init, mock_model_client):
        comp = self._cortex(mock_model_client, "test_cortex_wait_duration")
        started = time.monotonic()

        result = comp._execute_system_tool("wait", {"duration": 0.05})

        assert time.monotonic() - started >= 0.05
        assert result.startswith("Waited")

    def test_a_cancelled_task_cuts_it_short(self, rclpy_init, mock_model_client):
        comp = self._cortex(mock_model_client, "test_cortex_wait_cancelled")
        comp._main_goal_handle = MagicMock(is_cancel_requested=True)
        started = time.monotonic()

        result = comp._execute_system_tool("wait", {"duration": 30})

        assert time.monotonic() - started < 1
        assert "cancelled" in result

    def test_a_bad_duration_is_refused(self, rclpy_init, mock_model_client):
        comp = self._cortex(mock_model_client, "test_cortex_wait_bad")

        assert comp._wait("soon").startswith("Error:")
        assert comp._wait(-1).startswith("Error:")


class TestToolsFromTheRegistry:
    """Everything a component offers reaches Cortex through the action
    registry the Launcher built: described methods with their schema whole,
    action servers and services with their message fields"""

    def _register(self, comp, *entries):
        mock_component_internals(comp)
        comp._action_registry = _registry(*entries)
        comp.get_routines = MagicMock(return_value=[])
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
        assert tool["parameters"]["properties"]["wait_to_finish"]["type"] == "boolean"
        assert "wait_to_finish" not in tool["parameters"]["required"]
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

        result = comp._execute_system_tool(
            "send_goal_to_vla_run", {"task": "go", "wait_to_finish": False}
        )

        # The wait flag is Cortex's, not a field of the goal message
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

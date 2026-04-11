"""Tests for Cortex component — requires rclpy."""

import pytest
from unittest.mock import MagicMock

from agents.config import CortexConfig
from agents.ros import Topic, Action, ComponentRunType
from agents.components.cortex import Cortex
from tests.conftest import mock_component_internals


def _make_mock_action(name="test_action", description="A test action"):
    """Create a mock Action with the given name and description."""
    action = MagicMock(spec=Action)
    action.action_name = name
    action.description = description
    return action


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
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[action],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_tools",
        )

        assert len(comp._execution_tool_descriptions) == 1
        tool_desc = comp._execution_tool_descriptions[0]
        assert tool_desc["function"]["name"] == "navigate"
        assert tool_desc["function"]["description"] == "Go somewhere"

    def test_action_registers_in_execution_tools(self, rclpy_init, mock_model_client):
        action = _make_mock_action(name="grasp", description="Grasp object")
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[action],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_events",
        )

        assert "grasp" in comp._execution_tools

    def test_multiple_actions(self, rclpy_init, mock_model_client):
        actions = [
            _make_mock_action(name="navigate", description="Go to location"),
            _make_mock_action(name="grasp", description="Grasp object"),
            _make_mock_action(name="release", description="Release object"),
        ]
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=actions,
            model_client=mock_model_client,
            config=CortexConfig(),
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

        plan = comp._plan_task("fetch a cup")
        assert plan is not None
        assert len(plan) == 2
        assert plan[0]["function"]["name"] == "navigate"
        assert plan[1]["function"]["name"] == "grasp"

    def test_plan_task_no_tool_calls_returns_none(self, rclpy_init, mock_model_client):
        mock_model_client.inference.return_value = {
            "output": "I don't need to do anything.",
        }
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_mock_action()],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_plan_none",
        )
        mock_component_internals(comp)

        plan = comp._plan_task("just say hello")
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
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_mock_action()],
            model_client=mock_model_client,
            config=CortexConfig(max_execution_steps=5),
            component_name="test_cortex_truncate",
        )
        mock_component_internals(comp)

        plan = comp._plan_task("big task")
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
            outputs=[Topic(name="out", msg_type="String")],
            actions=[
                _make_mock_action(name="navigate", description="Go"),
            ],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_plan_loop",
        )
        mock_component_internals(comp)
        # Add inspect_component as a planning tool (normally done during activation)
        comp._planning_tools.add("inspect_component")
        comp._managed_components = {}

        plan = comp._plan_task("find an object")
        assert plan is not None
        assert len(plan) == 1
        assert plan[0]["function"]["name"] == "navigate"
        # LLM was called twice (one inspect, one action)
        assert mock_model_client.inference.call_count == 2

    def test_plan_exhausts_max_planning_steps(self, rclpy_init, mock_model_client):
        """Planning loop exits when max_planning_steps is reached."""
        # LLM always calls inspect_component, never produces action tools
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
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_mock_action()],
            model_client=mock_model_client,
            config=CortexConfig(max_planning_steps=3),
            component_name="test_cortex_plan_exhaust",
        )
        mock_component_internals(comp)
        comp._planning_tools.add("inspect_component")
        comp._managed_components = {}

        plan = comp._plan_task("complex task")
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
        decision = comp._confirm_step(plan, [], 0)
        assert decision == "EXECUTE"

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
        decision = comp._confirm_step(plan, [], 0)
        assert decision == "SKIP"

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
        decision = comp._confirm_step(plan, [], 0)
        assert decision == "ABORT"

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
        decision = comp._confirm_step(plan, [], 0)
        assert decision == "EXECUTE"


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

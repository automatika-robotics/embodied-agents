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

    def test_no_actions_raises(self, rclpy_init, mock_model_client):
        with pytest.raises(ValueError):
            Cortex(
                outputs=[Topic(name="out", msg_type="String")],
                actions=[],
                model_client=mock_model_client,
                config=CortexConfig(),
                component_name="test_cortex_no_actions",
            )

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

        assert len(comp.config._tool_descriptions) == 1
        tool_desc = comp.config._tool_descriptions[0]
        assert tool_desc["function"]["name"] == "navigate"
        assert tool_desc["function"]["description"] == "Go somewhere"

    def test_action_creates_internal_event_topic(self, rclpy_init, mock_model_client):
        action = _make_mock_action(name="grasp", description="Grasp object")
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[action],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_events",
        )

        assert "grasp" in comp._action_event_topics
        event_topic = comp._action_event_topics["grasp"]
        assert "internal_cortex_event" in event_topic.name
        assert "grasp" in event_topic.name

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

        assert len(comp.config._tool_descriptions) == 3
        assert len(comp._action_event_topics) == 3

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

        result = comp._dispatch_action("nonexistent")
        assert "does not exist" in result
        assert "real_action" in result


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

    def test_plan_truncated_to_max_iterations(self, rclpy_init, mock_model_client):
        mock_model_client.inference.return_value = {
            "output": "",
            "tool_calls": [
                {"function": {"name": f"step_{i}", "arguments": {}}}
                for i in range(20)
            ],
        }
        comp = Cortex(
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_mock_action()],
            model_client=mock_model_client,
            config=CortexConfig(max_iterations=5),
            component_name="test_cortex_truncate",
        )
        mock_component_internals(comp)

        plan = comp._plan_task("big task")
        assert len(plan) == 5


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
        mock_model_client.inference.return_value = {
            "output": "SKIP: already done"
        }
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
        mock_model_client.inference.return_value = {
            "output": "ABORT: unsafe condition"
        }
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
        mock_model_client.inference.return_value = {
            "output": "Sure, go ahead!"
        }
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

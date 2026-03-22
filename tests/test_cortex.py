"""Tests for Cortex component — requires rclpy."""

import pytest
from unittest.mock import MagicMock

from agents.config import CortexConfig
from agents.ros import Topic, Action, ComponentRunType
from agents.components.cortex import Cortex, CortexAction
from tests.conftest import mock_component_internals


def _make_mock_action(name="test_action"):
    """Create a mock Action with the given name."""
    action = MagicMock(spec=Action)
    action.action_name = name
    return action


def _make_cortex_action(name="test_action", description="A test action"):
    """Create a CortexAction with a mock Action."""
    return CortexAction(
        action=_make_mock_action(name),
        description=description,
    )


class TestCortexConstruction:
    def test_with_model_client(self, rclpy_init, mock_model_client):
        comp = Cortex(
            inputs=[],
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_cortex_action()],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex",
        )
        assert comp.model_client is mock_model_client

    def test_with_local_model(self, rclpy_init):
        comp = Cortex(
            inputs=[],
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_cortex_action()],
            config=CortexConfig(enable_local_model=True),
            component_name="test_cortex_local",
        )
        assert comp.config.enable_local_model is True

    def test_no_client_no_local_raises(self, rclpy_init):
        with pytest.raises(RuntimeError):
            Cortex(
                inputs=[],
                outputs=[Topic(name="out", msg_type="String")],
                actions=[_make_cortex_action()],
                config=CortexConfig(),
                component_name="test_cortex_fail",
            )

    def test_no_actions_raises(self, rclpy_init, mock_model_client):
        with pytest.raises(ValueError):
            Cortex(
                inputs=[],
                outputs=[Topic(name="out", msg_type="String")],
                actions=[],
                model_client=mock_model_client,
                config=CortexConfig(),
                component_name="test_cortex_no_actions",
            )

    def test_config_enforced(self, rclpy_init, mock_model_client):
        comp = Cortex(
            inputs=[],
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_cortex_action()],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_config",
        )
        assert comp.config.chat_history is True
        assert comp.config.stream is False

    def test_action_server_run_type(self, rclpy_init, mock_model_client):
        comp = Cortex(
            inputs=[],
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_cortex_action()],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_runtype",
        )
        assert comp.run_type == ComponentRunType.ACTION_SERVER


class TestCortexActions:
    def test_action_registers_tool_description(self, rclpy_init, mock_model_client):
        action = _make_cortex_action(name="navigate", description="Go somewhere")
        comp = Cortex(
            inputs=[],
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
        action = _make_cortex_action(name="grasp", description="Grasp object")
        comp = Cortex(
            inputs=[],
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
            _make_cortex_action(name="navigate", description="Go to location"),
            _make_cortex_action(name="grasp", description="Grasp object"),
            _make_cortex_action(name="release", description="Release object"),
        ]
        comp = Cortex(
            inputs=[],
            outputs=[Topic(name="out", msg_type="String")],
            actions=actions,
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_multi",
        )

        assert len(comp.config._tool_descriptions) == 3
        assert len(comp._action_event_topics) == 3

    def test_list_of_actions_per_cortex_action(self, rclpy_init, mock_model_client):
        """A single CortexAction can dispatch multiple Actions."""
        action1 = _make_mock_action("alert_sound")
        action2 = _make_mock_action("alert_light")
        cortex_action = CortexAction(
            action=[action1, action2],
            description="Sound alert and flash lights",
        )
        comp = Cortex(
            inputs=[],
            outputs=[Topic(name="out", msg_type="String")],
            actions=[cortex_action],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_multi_action",
        )

        # Named after the first action in the list
        assert "alert_sound" in comp._action_event_topics

    def test_tool_response_flags_set_to_true(self, rclpy_init, mock_model_client):
        """All tool responses must go back to the LLM for the ReAct loop."""
        action = _make_cortex_action(name="test", description="Test")
        comp = Cortex(
            inputs=[],
            outputs=[Topic(name="out", msg_type="String")],
            actions=[action],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_flags",
        )

        assert comp.config._tool_response_flags["test"] is True

    def test_dispatch_action_unknown(self, rclpy_init, mock_model_client):
        action = _make_cortex_action(name="real_action", description="Exists")
        comp = Cortex(
            inputs=[],
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


class TestNoLLMMethods:
    def test_no_llm_methods(self, rclpy_init, mock_model_client):
        """Cortex extends ModelComponent, not LLM — no LLM-specific methods."""
        comp = Cortex(
            inputs=[],
            outputs=[Topic(name="out", msg_type="String")],
            actions=[_make_cortex_action()],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_cortex_no_llm",
        )
        assert not hasattr(comp, "register_tool")
        assert not hasattr(comp, "set_component_prompt")

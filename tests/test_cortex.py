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
    """The plugin-action bridge: factories on a `RobotPlugin` registered as
    namespaced execution tools, dispatchable via the same internal-event
    path as constructor-supplied behavioral actions."""

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
        assert "lite3.sit_stand" in comp._additional_internal_actions
        assert "lite3.stop" in comp._additional_internal_actions
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

    def test_factory_failure_logged_and_skipped(
        self, rclpy_init, mock_model_client
    ):
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
        comp.add_plugin_actions(plugin)

        assert "lite3.ok" in comp._execution_tools
        assert "lite3.broken" not in comp._execution_tools


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

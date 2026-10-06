"""Tests for the DecisionComponent and the Decision message — requires rclpy."""

import json
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from agents.clients.model_base import ModelClient
from agents.components.component_base import Component
from agents.components.decision import DecisionComponent
from agents.config import DecisionConfig
from agents.ros import ActionReturnType, Decision, Topic, component_action
from tests.conftest import mock_component_internals
from tests.test_vision import _callback

NOUL = {"type": "noul", "noul": 0.97}
CHOICE = {
    "type": "choice",
    "choice": "kitchen",
    "probabilities": {"kitchen": 0.9, "bedroom": 0.1},
    "confidence": 0.8,
}
SCORE = {
    "type": "score",
    "score": 1.1,
    "legend": {"0": "empty", "1": "a few items", "2": "busy"},
    "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
    "confidence": 0.7,
}

QUESTIONS = {
    "stop": {
        "type": "noul",
        "instructions": "Is the person telling the robot to stop?",
    },
    "room": {
        "type": "choice",
        "instructions": "Which room is this?",
        "criteria": {"kitchen": None, "bedroom": None},
    },
}


@pytest.fixture
def decision_client():
    """A mock client serving a decision model"""
    client = MagicMock(spec=ModelClient)
    type(client).supports_decisions = PropertyMock(return_value=True)
    type(client).supports_tool_calls = PropertyMock(return_value=False)
    type(client).inference_timeout = PropertyMock(return_value=30)
    client.inference.return_value = {"output": {"stop": NOUL, "room": CHOICE}}
    return client


@pytest.fixture
def decider(rclpy_init, decision_client):
    """A component with two questions, a text input and an image input, whose
    question publishers are mocks"""
    speech = Topic(name="speech", msg_type="String")
    comp = DecisionComponent(
        inputs=[speech, Topic(name="camera", msg_type="Image")],
        model_client=decision_client,
        questions=QUESTIONS,
        trigger=speech,
        component_name="decider",
    )
    mock_component_internals(comp)
    comp.publishers_dict = {t.name: MagicMock() for t in comp.out_topics}
    comp.callbacks = {
        "speech": _callback(output="Stop right now!", msg_type="String", name="speech"),
        "camera": _callback(output=np.zeros((2, 2, 3), np.uint8), name="camera"),
    }
    comp.trig_callbacks = {}
    return comp


class TestDecisionMessage:
    def test_noul(self):
        msg = Decision.convert(NOUL, id="stop")
        assert (msg.id, msg.type) == ("stop", "noul")
        assert msg.noul == pytest.approx(0.97)
        assert msg.confidence == pytest.approx(0.94)
        assert list(msg.options) == ["false", "true"]
        assert list(msg.probabilities) == pytest.approx([0.03, 0.97])

    def test_choice(self):
        msg = Decision.convert(CHOICE, id="room")
        assert msg.choice == "kitchen"
        assert msg.confidence == pytest.approx(0.8)
        assert list(msg.options) == ["kitchen", "bedroom"]
        assert list(msg.probabilities) == pytest.approx([0.9, 0.1])

    def test_score_options_are_the_level_descriptions(self):
        msg = Decision.convert(SCORE, id="clutter")
        assert msg.score == pytest.approx(1.1)
        assert list(msg.options) == ["empty", "a few items", "busy"]
        assert list(msg.probabilities) == pytest.approx([0.1, 0.8, 0.1])

    def test_event_conditions_read_the_answer(self):
        topic = Topic(name="decider/stop", msg_type="Decision")
        condition = (topic.msg.noul > 0.8) & (topic.msg.confidence > 0.5)
        assert condition.evaluate({"decider/stop": Decision.convert(NOUL, id="stop")})
        assert not condition.evaluate({
            "decider/stop": Decision.convert({"type": "noul", "noul": 0.6}, id="stop")
        })

    @pytest.mark.parametrize(
        "answer, id, text",
        [
            (NOUL, "stop", "stop: yes (0.97)"),
            ({"type": "noul", "noul": 0.2}, "stop", "stop: no (0.80)"),
            (CHOICE, "room", "room: kitchen (0.90)"),
            (SCORE, "clutter", "clutter: a few items (0.80)"),
        ],
    )
    def test_callback_gives_the_answer_as_text(self, answer, id, text):
        topic = Topic(name="decider/q", msg_type="Decision")
        callback = topic.msg_type.callback(topic)
        assert callback.get_output() is None
        callback.msg = Decision.convert(answer, id=id)
        assert callback.get_output() == text


class TestConstruction:
    def test_a_question_gets_its_own_topic(self, decider):
        assert [t.name for t in decider.out_topics] == ["decider/stop", "decider/room"]
        assert all(t.msg_type is Decision for t in decider.out_topics)
        assert "decider/room" in decider.inspect_component()

    def test_questions_reach_the_component_through_its_config(
        self, rclpy_init, decision_client
    ):
        """As a multiprocess launch rebuilds the component from its config"""
        speech = Topic(name="speech", msg_type="String")
        given = DecisionComponent(
            inputs=[speech],
            model_client=decision_client,
            questions=QUESTIONS,
            component_name="decider",
        )
        config = DecisionConfig()
        config.from_json(given.config.to_json())
        rebuilt = DecisionComponent(
            inputs=[speech],
            model_client=decision_client,
            config=config,
            outputs=None,
            db_client=None,
            component_name="decider",
        )
        assert rebuilt.config._questions == QUESTIONS
        assert [t.name for t in rebuilt.out_topics] == ["decider/stop", "decider/room"]

    def test_rejects_a_client_without_decisions(self, rclpy_init, mock_model_client):
        type(mock_model_client).supports_decisions = PropertyMock(return_value=False)
        with pytest.raises(TypeError, match="decision model"):
            DecisionComponent(
                inputs=[Topic(name="speech", msg_type="String")],
                model_client=mock_model_client,
                component_name="decider",
            )

    @pytest.mark.parametrize(
        "id, question, error",
        [
            ("q", {"type": "pick", "instructions": "x"}, "must be one of"),
            ("q", {"type": "noul"}, "needs instructions"),
            ("q", {"type": "choice", "instructions": "x"}, "non-empty dictionary"),
            ("q", {"type": "score", "instructions": "x", "criteria": ["a"]}, "2 to 10"),
            ("q", {"type": "noul", "instructions": "x", "criteria": "yes"}, "optional"),
            ("a/b", {"type": "noul", "instructions": "x"}, "without '/'"),
        ],
    )
    def test_rejects_an_invalid_question(
        self, rclpy_init, decision_client, id, question, error
    ):
        with pytest.raises(ValueError, match=error):
            DecisionComponent(
                inputs=[Topic(name="speech", msg_type="String")],
                model_client=decision_client,
                questions={id: question},
                component_name="decider",
            )


class TestToolOnly:
    """A decider without inputs only answers `ask` with a given state"""

    @pytest.fixture
    def tool(self, rclpy_init, decision_client):
        comp = DecisionComponent(model_client=decision_client, component_name="tool")
        mock_component_internals(comp)
        comp.publishers_dict = {}
        comp.callbacks = {}
        return comp

    def test_standing_questions_need_inputs(self, rclpy_init, decision_client, tool):
        with pytest.raises(ValueError, match="none were given"):
            DecisionComponent(
                model_client=decision_client, questions=QUESTIONS, component_name="t"
            )
        assert not tool.config._questions and not tool.publishers_dict

    def test_asks_about_a_given_state_only(self, tool, decision_client):
        decision_client.inference.return_value = {"output": {"ask": NOUL}}
        ok, _ = ask(tool, instructions="Stop?", state="Please stop")
        assert ok
        ok, why = ask(tool, instructions="Stop?")
        assert not ok and "no state was given" in why

    def test_a_trigger_does_nothing(self, tool, decision_client):
        tool.log_once = MagicMock()
        tool._execution_step()
        decision_client.inference.assert_not_called()


class _Arm(Component):
    """A component with an action that reports what it is doing"""

    def _execution_step(self, **kwargs):
        pass

    @component_action
    def get_current_task(self) -> ActionReturnType:
        """The task being carried out"""
        return True, "pick up the orange"


class TestActionStates:
    """The state can hold what other components report when asked"""

    @pytest.fixture
    def checker(self, rclpy_init, decision_client):
        arm = _Arm(component_name="arm")
        comp = DecisionComponent(
            inputs=[Topic(name="camera", msg_type="Image")],
            model_client=decision_client,
            questions={"stop": QUESTIONS["stop"]},
            action_states={"task": arm.get_current_task},
            component_name="checker",
        )
        mock_component_internals(comp)
        comp.publishers_dict = {t.name: MagicMock() for t in comp.out_topics}
        comp.callbacks = {
            "camera": _callback(output=np.zeros((2, 2, 3), np.uint8), name="camera")
        }
        comp.trig_callbacks = {}
        comp._call_component_method = MagicMock(
            return_value=(True, "pick up the orange")
        )
        comp.log_once = MagicMock()
        return comp

    def test_they_travel_by_name_in_the_config(
        self, rclpy_init, decision_client, checker
    ):
        assert checker.config._action_states == {"task": "arm.get_current_task"}
        assert "task: arm.get_current_task" in checker.inspect_component()

        config = DecisionConfig()
        config.from_json(checker.config.to_json())
        rebuilt = DecisionComponent(
            model_client=decision_client, config=config, component_name="checker"
        )
        assert rebuilt.config._action_states == {"task": "arm.get_current_task"}

    def test_a_client_is_made_for_each_component(self, checker):
        with (
            patch.object(Component, "create_all_service_clients"),
            patch("agents.components.model_component.ServiceClientHandler") as handler,
        ):
            checker.create_all_service_clients()
        assert list(checker._component_clients) == ["arm"]
        assert handler.call_args.kwargs["srv_name"] == "arm/execute_method"

    def test_only_component_actions_are_taken(self, rclpy_init, decision_client):
        with pytest.raises(TypeError, match="action of a component"):
            DecisionComponent(
                model_client=decision_client,
                action_states={"task": lambda: "pick"},
                component_name="checker",
            )

    def test_questions_can_be_asked_about_action_states_alone(
        self, rclpy_init, decision_client
    ):
        arm = _Arm(component_name="arm_alone")
        comp = DecisionComponent(
            model_client=decision_client,
            questions={"stop": QUESTIONS["stop"]},
            action_states={"task": arm.get_current_task},
            component_name="checker_alone",
        )
        assert [t.name for t in comp.out_topics] == ["checker_alone/stop"]

    def test_the_result_joins_the_state(self, checker, decision_client):
        decision_client.inference.return_value = {"output": {"stop": NOUL}}

        checker._execution_step()

        checker._call_component_method.assert_called_once_with(
            "arm", "get_current_task"
        )
        sent = decision_client.inference.call_args.args[0]
        assert (
            sent["state"] == {"task": "pick up the orange"} and len(sent["images"]) == 1
        )
        checker.publishers_dict["checker/stop"].publish.assert_called_once()

    def test_a_failed_action_state_skips_the_cycle(self, checker, decision_client):
        checker._call_component_method.return_value = (False, "No goal is running")

        checker._execution_step()

        decision_client.inference.assert_not_called()
        checker.publishers_dict["checker/stop"].publish.assert_not_called()
        assert "No goal is running" in checker.log_once.call_args.args[1]

    def test_ask_reads_them_too_unless_given_a_state(self, checker, decision_client):
        decision_client.inference.return_value = {"output": {"ask": NOUL}}

        ok, _ = ask(checker, instructions="Is the task done?")
        assert ok
        assert decision_client.inference.call_args.args[0]["state"] == {
            "task": "pick up the orange"
        }

        checker._call_component_method.reset_mock()
        ask(
            checker,
            instructions="Is the task done?",
            state="The orange is on the plate",
        )
        checker._call_component_method.assert_not_called()

        checker._call_component_method.return_value = (False, "No goal is running")
        ok, why = ask(checker, instructions="Is the task done?")
        assert not ok and "No goal is running" in why


class TestAskingOnTrigger:
    def test_text_is_the_state_and_images_are_images(self, decider):
        inference_input, _ = decider._create_input()
        assert inference_input["state"] == {"speech": "Stop right now!"}
        assert len(inference_input["images"]) == 1

    def test_a_trigger_input_is_part_of_the_state(self, decider):
        decider.trig_callbacks = {"speech": decider.callbacks.pop("speech")}
        assert decider._create_input()[0]["state"] == {"speech": "Stop right now!"}

    def test_each_answer_is_published_on_its_question_topic(
        self, decider, decision_client
    ):
        decider._execution_step(topic=Topic(name="speech", msg_type="String"))

        sent = decision_client.inference.call_args.args[0]
        assert sent["questions"] == QUESTIONS and sent["state"] == {
            "speech": "Stop right now!"
        }
        decider.publishers_dict["decider/stop"].publish.assert_called_once_with(
            NOUL, id="stop"
        )
        decider.publishers_dict["decider/room"].publish.assert_called_once_with(
            CHOICE, id="room"
        )

    def test_nothing_is_asked_without_inputs(self, decider, decision_client):
        for callback in decider.callbacks.values():
            callback.get_output.return_value = None
        decider._execution_step()
        decision_client.inference.assert_not_called()

    def test_a_failed_inference_publishes_nothing(self, decider, decision_client):
        decision_client.inference.return_value = None
        decider._execution_step()
        decider.publishers_dict["decider/stop"].publish.assert_not_called()


def ask(decider, **kwargs):
    return DecisionComponent.ask.__wrapped__(decider, **kwargs)


class TestAsk:
    def test_returns_the_answer(self, decider, decision_client):
        decision_client.inference.return_value = {"output": {"ask": NOUL}}

        ok, answer = ask(decider, instructions="Stop?")

        assert ok and json.loads(answer) == NOUL
        sent = decision_client.inference.call_args.args[0]
        assert sent["questions"] == {
            "ask": {"type": "noul", "instructions": "Stop?", "criteria": None}
        }
        assert sent["state"] == {"speech": "Stop right now!"} and "images" in sent

    def test_a_given_state_replaces_the_inputs(self, decider, decision_client):
        decision_client.inference.return_value = {"output": {"ask": NOUL}}

        ask(decider, instructions="Stop?", state="Please stop")

        sent = decision_client.inference.call_args.args[0]
        assert sent["state"] == "Please stop" and "images" not in sent

    def test_without_inputs_or_an_answer(self, decider, decision_client):
        decision_client.inference.return_value = None
        ok, why = ask(decider, instructions="Stop?")
        assert not ok and "did not answer" in why

        for callback in decider.callbacks.values():
            callback.get_output.return_value = None
        ok, why = ask(decider, instructions="Stop?")
        assert not ok and "no state was given" in why

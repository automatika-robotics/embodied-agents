import json
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..clients.model_base import ModelClient
from ..config import DecisionConfig
from ..ros import (
    ActionPhase,
    ActionReturnType,
    BaseComponent,
    Decision,
    Detections,
    Detections3D,
    DetectionsMultiSource,
    Event,
    FixedInput,
    Image,
    RGBD,
    String,
    StreamingString,
    Topic,
    component_action,
)
from ..utils import validate_func_args
from .model_component import ModelComponent

# The kinds of question a decision model answers
QUESTION_TYPES = ("choice", "score", "noul")

# The fields of a question, as the `ask` tool takes them
_QUESTION_PROPERTIES = {
    "instructions": {
        "type": "string",
        "description": (
            "The question, asked about the given state, or about the component's "
            "current inputs when no state is given"
        ),
    },
    "type": {
        "type": "string",
        "enum": list(QUESTION_TYPES),
        "description": (
            "The kind of answer. 'noul': a yes/no question, answered with the "
            "probability of yes (default). 'choice': pick one of the options "
            "given in criteria, answered with the chosen option and the "
            "probability of each. 'score': rate on the ordered levels given in "
            "criteria, answered with the expected level."
        ),
    },
    "criteria": {
        "description": (
            "The possible answers. Required for 'choice': an object with one key "
            "per option, whose value is a short description of the option or "
            'null, e.g. {"kitchen": null, "bedroom": "where people sleep"}. '
            "Required for 'score': a list of 2 to 10 level descriptions from the "
            'lowest level to the highest, e.g. ["empty", "a few items", '
            "\"cluttered\"]. Leave out for 'noul'."
        ),
    },
}


class DecisionComponent(ModelComponent):
    """
    This component asks a decision model typed questions about its inputs and
    publishes the answers. A decision model (served through the TypeSafe-compatible
    /v1/systemone API) answers each question in one forward pass, without
    generating text. A yes/no probability, a choice among given options, or a
    score on ordered levels, with a probability for every option. This makes it
    a fast "System 1" as compared to autoregressive LLM/VLM models.

    The questions are given as a dictionary keyed by question id, in the format
    of the API: each question has a `type` (`noul` for yes/no, `choice` or
    `score`), `instructions`, and `criteria` (the options of a choice, as a
    dictionary of option to description, or the levels of a score, as a list).
    On each trigger, all questions are asked about the current inputs in one
    request: text inputs make up the state, as a JSON object keyed by topic
    name, and image inputs are given as images (with a model that reads images).
    The state can also hold what other components report when asked: each of
    the `action_states` is a component action that is called before the
    questions are asked, and its result joins the state under the given name.
    When one of them fails the questions are not asked on that trigger.

    **The component does not need to be given output topics.** Each question is
    published on its own topic, named `<component_name>/<question_id>`, with a
    [Decision](agents.ros.md#classes) message per answer. Use that topic in an
    event condition or give it to the UI (the topics are also listed in
    `out_topics`). Since each answer field is a scalar, thresholds and confidence
    cutoffs are set in the event condition.

    The component also offers `ask` as an action that an LLM component or Cortex
    can call as a tool: it answers one question now, about the current inputs or
    about a state the caller gives.

    :param inputs: The input topics the questions are asked about. Text
        (String, StreamingString), detections (Detections, Detections3D,
        DetectionsMultiSource) and images (Image, CompressedImage, RGBD).
        Without inputs, the component is only used as a tool: another model
        gives `ask` the state to decide about, and there are no standing
        questions.
    :type inputs: Optional[list[Union[Topic, FixedInput]]]
    :param model_client: A model client with a decision model, such as
        GenericHTTPClient with a GenericDecisionModel.
    :type model_client: ModelClient
    :param questions: The standing questions to ask on every trigger, keyed
        by id, in the API format. Each gets an output topic. Needs inputs or
        action states to be asked about.
    :type questions: Optional[dict[str, dict]]
    :param action_states: Component actions whose results are part of the
        state, keyed by the name each result has in the state, e.g.
        ``{"task": vla.get_current_task}``. They are called on every trigger,
        in whatever process their components run.
    :type action_states: Optional[dict[str, Callable]]
    :param config: The configuration for the component. Defaults to DecisionConfig().
    :type config: Optional[DecisionConfig]
    :param trigger: The trigger for asking the questions: input topic(s), a rate
        in Hz, or an event. Defaults to 1.0 Hz.
    :type trigger: Union[Topic, list[Topic], float, Event]
    :param component_name: The name of the component, which prefixes its
        question topics.
    :type component_name: str

    Example usage:
    ```python
    speech = Topic(name="speech", msg_type="String")
    decider = DecisionComponent(
        inputs=[speech],
        model_client=GenericHTTPClient(GenericDecisionModel(name="lev", checkpoint="lev")),
        questions={
            "stop": {"type": "noul", "instructions": "Is the person telling the robot to stop?"},
            "addressed": {"type": "noul", "instructions": "Is this speech directed at the robot?"},
        },
        trigger=speech,
        component_name="speech_decider",
    )

    # Answers are published on speech_decider/stop and speech_decider/addressed
    stop = Topic(name="speech_decider/stop", msg_type="Decision")
    stop_event = Event(stop.msg.noul > 0.8, on_change=True)
    launcher.add_pkg(components=[decider, vla], events_actions={stop_event: Action(vla.cancel_main_goal)})
    ```
    """

    @validate_func_args
    def __init__(
        self,
        *,
        inputs: Optional[List[Union[Topic, FixedInput]]] = None,
        model_client: ModelClient,
        questions: Optional[Dict[str, Dict]] = None,
        action_states: Optional[Dict[str, Callable]] = None,
        config: Optional[DecisionConfig] = None,
        trigger: Union[Topic, List[Topic], float, Event] = 1.0,
        component_name: str,
        **kwargs,
    ):
        self.allowed_inputs = {
            "Required": [],
            "Optional": [
                String,
                StreamingString,
                Detections,
                Detections3D,
                DetectionsMultiSource,
                Image,
                RGBD,
            ],
        }
        self.handled_outputs = [Decision]

        if not model_client.supports_decisions:
            raise TypeError(
                f"The provided model client ({model_client.__class__.__name__}) "
                "does not serve a decision model. Use a client with a "
                "GenericDecisionModel, such as GenericHTTPClient."
            )

        config = config or DecisionConfig()
        if questions is not None:
            config._questions = questions
        # verify action states are component methods
        for name, action in (action_states or {}).items():
            component = getattr(action, "__self__", None)
            if not isinstance(component, BaseComponent):
                raise TypeError(
                    f"Action state '{name}' must be an action of a component, "
                    f"such as vla.get_current_task; got {action!r}"
                )
            config._action_states[name] = f"{component.node_name}.{action.__name__}"
        # if questions are provided at init then some state should always be given
        if config._questions and not inputs and not config._action_states:
            raise ValueError(
                "Standing questions are asked about the component's inputs or "
                "action states, and none were given. Give the inputs or action "
                "states to ask about, or no questions and use the component "
                "action as a tool with `ask`."
            )
        for question_id, question in config._questions.items():
            self._validate_question(question_id, question)

        # The outputs are one topic per question, not given by the user
        for kwarg in ["outputs", "db_client"]:
            kwargs.pop(kwarg, None)
        outputs = [
            Topic(name=f"{component_name}/{question_id}", msg_type="Decision")
            for question_id in config._questions
        ]

        super().__init__(
            inputs=inputs,
            outputs=outputs or None,
            model_client=model_client,
            config=config,
            trigger=trigger,
            component_name=component_name,
            **kwargs,
        )
        self.config: DecisionConfig

        # For keeping a component action and a triggered run serial
        self._inference_lock = threading.Lock()

    # =========================================================================
    # Validation
    # =========================================================================

    @staticmethod
    def _validate_question(question_id: str, question: Any) -> None:
        """Validate questions so that an error is raised at construction rather
        than inference.

        :raises ValueError: Naming what is wrong with the question
        """
        if not question_id or "/" in question_id:
            raise ValueError(
                f"Invalid question id {question_id!r}: it names the question's "
                "topic, so it must be a non-empty string without '/'"
            )
        where = f"Question '{question_id}'"
        if not isinstance(question, dict):
            raise ValueError(f"{where} must be a dictionary, got {type(question)}")
        question_type = question.get("type")
        if question_type not in QUESTION_TYPES:
            raise ValueError(
                f"{where} has type {question_type!r}, which must be one of {QUESTION_TYPES}"
            )
        if not question.get("instructions"):
            raise ValueError(f"{where} needs instructions")
        criteria = question.get("criteria")
        if question_type == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError(
                    f"{where} is a choice, so its criteria must be a non-empty "
                    "dictionary of option to description (or None)"
                )
        elif question_type == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError(
                    f"{where} is a score, so its criteria must be a list of 2 to "
                    "10 level descriptions, lowest first"
                )
        elif criteria is not None and not isinstance(criteria, dict):
            raise ValueError(
                f"{where} is a noul, so its criteria are optional descriptions "
                "of 'true' and 'false' in a dictionary"
            )

    @property
    def _component_methods(self) -> List[str]:
        """The component actions that give state"""
        return list(self.config._action_states.values())

    def inspect_component(self) -> str:
        """Return component info including its questions and their topics"""
        result = super().inspect_component()
        if self.config._action_states:
            result += "\nAction states (asked for on every trigger):\n" + "\n".join(
                f"  - {name}: {action}"
                for name, action in self.config._action_states.items()
            )
        if not self.config._questions:
            return result + "\nQuestions: none"
        lines = ["Questions (each answered on its own topic):"]
        for question_id, question in self.config._questions.items():
            lines.append(
                f"  - {question_id} ({question['type']}): {question['instructions']} "
                f"-> {self.node_name}/{question_id}"
            )
        return result + "\n" + "\n".join(lines)

    # =========================================================================
    # Inference
    # =========================================================================

    def _create_input(self) -> Tuple[Optional[Dict[str, Any]], str]:  # type: ignore
        """Gather the inference input. Text inputs and the results of the
        action states as the state, keyed by name, and the images as images.

        :return: The inference input, or None with why there is nothing to
            decide about
        """
        state: Dict[str, Any] = {}
        images = []
        callbacks = list(self.callbacks.values()) + list(
            getattr(self, "trig_callbacks", {}).values()
        )
        # gather all topic states
        for callback in callbacks:
            if (item := callback.get_output()) is None:
                continue
            if issubclass(callback.input_topic.msg_type, (Image, RGBD)):
                images.append(item)
            else:
                state[callback.input_topic.name] = item
        # gather all action states
        for name, action in self.config._action_states.items():
            component_name, method_name = action.split(".")
            success, message = self._call_component_method(component_name, method_name)
            if not success:
                return None, f"action state '{name}' ({action}) gave nothing: {message}"
            state[name] = message
        if not state and not images:
            return None, "no input has been received"
        inference_input: Dict[str, Any] = {"state": state}
        if images:
            inference_input["images"] = images
        return inference_input, ""

    def _ask(
        self, inference_input: Dict[str, Any], questions: Dict[str, Dict]
    ) -> Optional[Dict]:
        """Ask the questions about an input and return the answers keyed by
        question id, or None if the inference failed"""
        with self._inference_lock:
            result = self._call_inference({**inference_input, "questions": questions})
        return result.get("output") if result else None

    def _execution_step(self, *args, **kwargs):
        """Ask all the questions about the current inputs and publish each
        answer on its question's topic"""
        if not self.config._questions:
            self.log_once("no_questions", "No standing questions to ask", level="debug")
            return
        inference_input, why = self._create_input()
        if inference_input is None:
            self.log_once(why, f"Not asking the standing questions: {why}")
            return
        answers = self._ask(inference_input, self.config._questions)
        if answers is None:
            return
        # each answer goes to its own output topic
        for question_id, answer in answers.items():
            publisher = self.publishers_dict.get(f"{self.node_name}/{question_id}")
            if publisher is not None:
                publisher.publish(answer, id=question_id)

    def _warmup(self):
        """Warm up and stat check"""
        import time

        question = {"warmup": {"type": "noul", "instructions": "Is this a test?"}}
        if not self.model_client:
            return
        self.model_client.inference({"state": "Warming up.", "questions": question})
        start_time = time.time()
        result = self.model_client.inference({
            "state": "Warming up.",
            "questions": question,
        })
        elapsed_time = time.time() - start_time
        if result:
            self.get_logger().warning(
                f"Approximate Inference time: {elapsed_time} seconds"
            )
        else:
            self.get_logger().error("Model inference failed during warmup.")

    def _handle_websocket_streaming(self):
        """Not used -- decision model outputs are not streamed."""
        pass

    # =========================================================================
    # Actions
    # =========================================================================

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "ask",
                "description": (
                    "Ask the decision model one question about the current inputs "
                    "(what the robot hears or sees), or about a given state. "
                    "Returns the answer with its probabilities."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        **_QUESTION_PROPERTIES,
                        "state": {
                            "description": (
                                "Text or an object to ask about instead of the "
                                "component's inputs"
                            ),
                        },
                    },
                    "required": ["instructions"],
                },
            },
        },
        active=True,
        phase=ActionPhase.BOTH,
    )
    def ask(
        self,
        instructions: str,
        type: str = "noul",
        criteria: Optional[Union[Dict, List]] = None,
        state: Optional[Any] = None,
    ) -> ActionReturnType:
        """Ask one question now, about the current inputs or a given state.

        :param instructions: The question
        :param type: noul (yes/no), choice or score
        :param criteria: The options of a choice or the levels of a score
        :param state: What to ask about instead of the component's inputs and
            action states
        :return: Whether the answer was obtained, with the answer as JSON or why not
        :rtype: ActionReturnType
        """
        question = {"type": type, "instructions": instructions, "criteria": criteria}
        try:
            self._validate_question("ask", question)
        except ValueError as e:
            return False, str(e)

        if state is None:
            inference_input, why = self._create_input()
            if inference_input is None:
                return False, (f"Nothing to decide about: no state was given and {why}")
        else:
            inference_input = {"state": state}

        answers = self._ask(inference_input, {"ask": question})
        if not answers:
            return False, "The decision model did not answer"
        return True, json.dumps(answers["ask"])

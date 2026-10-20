"""Chat requests through the Ollama client.

The messages a component sends are its chat history, so what the client adds
for one request (the images of the current turn) must not end up in them.
"""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from ollama import ResponseError

from agents.clients.ollama import OllamaClient
from agents.models import OllamaModel


@pytest.fixture(autouse=True)
def no_server(monkeypatch):
    """No test reaches a server: the client checks its connection with a
    request of its own when constructed, not through the fake"""
    monkeypatch.setattr(OllamaClient, "_check_connection", lambda self: None)


def make_client(capabilities=("completion", "tools")):
    """A client over a fake Ollama, which records the requests it gets. The
    fake reports the given capabilities for the model"""
    fake = MagicMock()
    fake.show.return_value.capabilities = list(capabilities)
    fake.chat.return_value = {"message": {"content": "a cup"}}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("ollama.Client", MagicMock(return_value=fake))
        client = OllamaClient(OllamaModel(name="m", checkpoint="m"))
    client.logger = MagicMock()
    return client


@pytest.fixture
def ollama():
    return make_client()


def test_images_go_to_the_server_and_not_into_the_history(ollama):
    history = [
        {"role": "system", "content": "You are a robot"},
        {"role": "user", "content": "what is this?"},
    ]

    result = ollama.inference({
        "query": history,
        "images": [np.zeros((4, 4, 3), dtype=np.uint8)],
        "temperature": 0.1,
        "max_new_tokens": 10,
        "stream": False,
    })

    assert result["output"] == "a cup"
    sent = ollama.client.chat.call_args.kwargs["messages"]
    assert sent[0] == history[0]
    assert sent[-1]["content"] == "what is this?"
    assert len(sent[-1]["images"]) == 1
    assert history == [
        {"role": "system", "content": "You are a robot"},
        {"role": "user", "content": "what is this?"},
    ]


def test_images_kept_on_an_earlier_message_are_encoded(ollama):
    """With history_images above one, the history leaves the arrays of an
    earlier turn on its message; the client encodes them and leaves the
    history its arrays"""
    earlier = np.zeros((4, 4, 3), dtype=np.uint8)
    history = [
        {"role": "user", "content": "what was that?", "images": [earlier]},
        {"role": "assistant", "content": "a cup"},
        {"role": "user", "content": "and this?"},
    ]

    ollama.inference({
        "query": history,
        "images": [np.zeros((4, 4, 3), dtype=np.uint8)],
        "temperature": 0.1,
        "max_new_tokens": 10,
        "stream": False,
    })

    sent = ollama.client.chat.call_args.kwargs["messages"]
    assert isinstance(sent[0]["images"][0], str)
    assert len(sent[-1]["images"]) == 1
    assert history[0]["images"][0] is earlier


def running(name, context_length):
    model = MagicMock()
    model.model = model.name = name
    model.context_length = context_length
    return model


class TestContextLength:
    def test_the_served_length_of_the_loaded_model(self, ollama):
        ollama.client.ps.return_value.models = [
            running("other", 1),
            running("m:latest", 4096),
        ]
        assert ollama.context_length == 4096

    def test_num_ctx_in_the_options_wins(self, ollama):
        ollama.model_init_params["options"] = {"num_ctx": 32768}
        ollama.client.ps.return_value.models = [running("m", 4096)]
        assert ollama.context_length == 32768

    def test_the_trained_length_when_the_model_is_not_loaded(self, ollama):
        ollama.client.ps.return_value.models = []
        ollama.client.show.return_value.modelinfo = {
            "general.architecture": "qwen2",
            "qwen2.context_length": 40960,
        }
        assert ollama.context_length == 40960

    def test_none_when_nothing_can_be_read(self, ollama):
        ollama.client.ps.side_effect = RuntimeError("down")
        ollama.client.show.side_effect = RuntimeError("down")
        assert ollama.context_length is None

    def test_a_model_served_below_what_it_supports_is_warned_about(self, ollama):
        ollama.client.ps.return_value.models = [running("m", 4096)]
        ollama.client.show.return_value.modelinfo = {"qwen2.context_length": 40960}

        ollama._warn_on_a_small_context()

        warning = ollama.logger.warning.call_args.args[0]
        assert "4096" in warning and "40960" in warning and "num_ctx" in warning

    def test_no_warning_at_full_context(self, ollama):
        ollama.client.ps.return_value.models = [running("m", 40960)]
        ollama.client.show.return_value.modelinfo = {"qwen2.context_length": 40960}

        ollama._warn_on_a_small_context()

        ollama.logger.warning.assert_not_called()


def test_tool_calls_reach_the_component_as_dicts(ollama):
    """The package returns the calls as its own model objects"""
    from ollama._types import Message

    call = Message.ToolCall(
        function=Message.ToolCall.Function(name="set_led", arguments={"color": "red"})
    )
    ollama.client.chat.return_value = {"message": {"content": "", "tool_calls": [call]}}

    result = ollama.inference({
        "query": [{"role": "user", "content": "red light"}],
        "temperature": 0.1,
        "max_new_tokens": 10,
        "stream": False,
    })

    assert result["tool_calls"] == [
        {"function": {"name": "set_led", "arguments": {"color": "red"}}}
    ]
    json.dumps(result["tool_calls"])


class TestDecisionModels:
    """A tag Ollama reports as a decision model is asked typed questions"""

    QUESTIONS = {
        "stop": {"type": "noul", "instructions": "Is the robot told to stop?"},
        "route": {
            "type": "choice",
            "instructions": "Where does this go?",
            "criteria": {"goto": "a place", "chat": "talk"},
        },
    }

    def answers(self):
        """What the library returns: typed answers with model_dump"""
        return SimpleNamespace(
            answers={
                "stop": SimpleNamespace(
                    model_dump=lambda: {"type": "noul", "noul": 0.9}
                ),
                "route": SimpleNamespace(
                    model_dump=lambda: {
                        "type": "choice",
                        "choice": "goto",
                        "probabilities": {"goto": 0.8, "chat": 0.2},
                        "confidence": 0.5,
                    }
                ),
            }
        )

    def test_capabilities_tell_the_model_kinds_apart(self):
        decider = make_client(capabilities=["decision"])
        assert decider.supports_decisions
        assert not decider.supports_tool_calls
        chat = make_client()
        assert not chat.supports_decisions
        assert chat.supports_tool_calls

    def test_questions_go_to_systemone_and_answers_come_back_as_dicts(self):
        decider = make_client(capabilities=["decision"])
        decider.client.systemone.return_value = self.answers()
        result = decider.inference({
            "state": {"speech": "Stop!"},
            "questions": self.QUESTIONS,
        })
        sent = decider.client.systemone.call_args.kwargs
        assert sent == {
            "model": "m",
            "state": {"speech": "Stop!"},
            "questions": self.QUESTIONS,
        }
        assert result["output"]["stop"] == {"type": "noul", "noul": 0.9}
        assert result["output"]["route"]["choice"] == "goto"
        decider.client.chat.assert_not_called()

    def test_images_are_sent_as_base64(self):
        decider = make_client(capabilities=["decision"])
        decider.client.systemone.return_value = self.answers()
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        decider.inference({
            "state": "x",
            "questions": self.QUESTIONS,
            "images": [image],
        })
        sent = decider.client.systemone.call_args.kwargs["images"]
        assert len(sent) == 1 and isinstance(sent[0], str)

    def test_a_library_without_image_support_fails_the_call(self):
        decider = make_client(capabilities=["decision"])
        # The released library has no images parameter
        decider.client.systemone = (
            lambda model, state, questions, keep_alive=None: self.answers()
        )
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        assert (
            decider.inference({
                "state": "x",
                "questions": self.QUESTIONS,
                "images": [image],
            })
            is None
        )
        assert "images" in decider.logger.error.call_args.args[0]

    def test_a_decision_model_is_loaded_and_unloaded_with_a_question(self):
        decider = make_client(capabilities=["decision"])
        decider.client.show.return_value.modelinfo = {}
        decider.initialize()
        decider.client.generate.assert_not_called()
        loaded = decider.client.systemone.call_args.kwargs
        assert loaded["model"] == "m" and loaded["keep_alive"] == 10
        assert "warmup" in loaded["questions"]
        decider.deinitialize()
        assert decider.client.systemone.call_args.kwargs["keep_alive"] == 0


class TestBackgroundPull:
    """A tag that is not installed is pulled without holding up the recipe"""

    def pulling_client(self, capabilities):
        """A fake Ollama without the model: show fails until the pull, which
        waits on an event, has run"""
        fake = MagicMock()
        released = threading.Event()
        state = {"pulled": False}

        def show(_):
            if not state["pulled"]:
                raise ResponseError("not found", 404)
            return SimpleNamespace(capabilities=capabilities, modelinfo={})

        def pull(_, stream=False):
            released.wait(timeout=5)
            state["pulled"] = True
            return iter([{"status": "pulling manifest"}, {"status": "success"}])

        fake.show.side_effect = show
        fake.pull.side_effect = pull
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("ollama.Client", MagicMock(return_value=fake))
            client = OllamaClient(OllamaModel(name="m", checkpoint="m"))
        client.logger = MagicMock()
        return client, released

    def test_construction_does_not_wait_for_the_pull(self):
        client, released = self.pulling_client(["decision"])
        assert client._pull.is_alive()
        # A chat model is assumed until the pull says otherwise
        assert client.supports_tool_calls
        released.set()
        client._pull.join(timeout=5)

    def test_supports_decisions_waits_for_the_pull(self):
        client, released = self.pulling_client(["decision"])
        answer = {}
        asker = threading.Thread(
            target=lambda: answer.update(d=client.supports_decisions)
        )
        asker.start()
        asker.join(timeout=0.2)
        assert asker.is_alive() and not answer  # still waiting on the pull
        released.set()
        asker.join(timeout=5)
        assert answer == {"d": True}
        assert not client.supports_tool_calls

    def test_progress_is_logged_once_per_tenth(self):
        fake = MagicMock()
        fake.show.side_effect = [
            ResponseError("not found", 404),
            SimpleNamespace(capabilities=["decision"], modelinfo={}),
        ]

        def layer(done):
            return {
                "status": "pulling 7daa",
                "digest": "7daa",
                "completed": done,
                "total": 5_000_000,
            }

        tiny = {
            "status": "pulling 0e1b",
            "digest": "0e1b",
            "completed": 500,
            "total": 500,
        }
        fake.pull.return_value = iter([
            {"status": "pulling manifest"},
            layer(0),
            layer(600_000),
            layer(950_000),
            layer(4_950_000),
            layer(5_000_000),
            tiny,
            {"status": "verifying sha256 digest"},
            {"status": "success"},
        ])
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("ollama.Client", MagicMock(return_value=fake))
            logger = MagicMock()
            patch.setattr(
                "agents.clients.model_base.logging.get_logger", lambda _: logger
            )
            client = OllamaClient(OllamaModel(name="m", checkpoint="m"))
        assert client.supports_decisions
        logged = [
            call.args[0]
            for call in logger.info.call_args_list
            if call.args[0].startswith("Pulling m:")
        ]
        # Stages without a size are logged; a layer is its progress, a tiny one nothing
        assert logged == [
            "Pulling m: pulling manifest",
            "Pulling m: 1 of 5 MB (10%)",
            "Pulling m: 5 of 5 MB (90%)",
            "Pulling m: 5 of 5 MB (100%)",
            "Pulling m: verifying sha256 digest",
            "Pulling m: success",
        ]

    def test_a_failed_pull_is_not_tried_again(self):
        fake = MagicMock()
        fake.show.side_effect = ResponseError("not found", 404)
        fake.pull.side_effect = ResponseError(
            "pull model manifest: file does not exist"
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("ollama.Client", MagicMock(return_value=fake))
            client = OllamaClient(OllamaModel(name="m", checkpoint="nosuchmodel"))
        client.logger = MagicMock()
        client._pull.join(timeout=5)
        assert not client.supports_decisions
        assert fake.pull.call_count == 1
        errors = [call.args[0] for call in client.logger.error.call_args_list]
        assert any("could not be pulled" in error for error in errors)

    def test_initialize_waits_for_the_pull(self):
        client, released = self.pulling_client(["completion", "tools"])
        released.set()
        client.initialize()
        client.client.generate.assert_called_once()
        assert client.client.pull.call_count == 1  # the background pull, no second one

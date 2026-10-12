"""Chat requests through the Ollama client.

The messages a component sends are its chat history, so what the client adds
for one request (the images of the current turn) must not end up in them.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from agents.clients.ollama import OllamaClient
from agents.models import OllamaModel


@pytest.fixture
def ollama():
    """A client over a fake Ollama, which records the chat requests it gets"""
    client = OllamaClient(OllamaModel(name="m", checkpoint="m"))
    client.client = MagicMock()
    client.client.chat.return_value = {"message": {"content": "a cup"}}
    client.logger = MagicMock()
    return client


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

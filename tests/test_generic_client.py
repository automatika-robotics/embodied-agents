"""Tool calls and decisions through the client for OpenAI-compatible servers.

Components get the arguments of a tool call as a dict from every client. Most
OpenAI-compatible servers (vLLM, SGLang, llama.cpp, lmdeploy, ...) send them as
a JSON string, while some (e.g. TGI before 3.2) send a JSON object; both are
read. They are sent back to the server as a JSON string, as the OpenAI format
expects.

A decision model is asked its questions at the TypeSafe-compatible
/v1/systemone endpoint, and the client gives back the answers keyed by question
id.
"""

import base64
import json
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

from agents.clients.generic import GenericHTTPClient
from agents.models import GenericDecisionModel, GenericLLM

MODEL = {
    "model_type": "GenericLLM",
    "model_name": "m",
    "init_timeout": 10,
    "model_init_params": {"checkpoint": "m"},
}


def tool_call(arguments, name="get_battery_level", call_id="call_abc"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


@pytest.fixture
def server():
    """A client talking to a fake server, which records the requests it gets
    and replies with the tool calls it is given"""
    requests = []
    replies = {"tool_calls": []}

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        message = {"role": "assistant", "content": ""}
        if replies["tool_calls"]:
            message["tool_calls"] = replies["tool_calls"]
        return httpx.Response(200, json={"choices": [{"message": message}]})

    client = GenericHTTPClient(MODEL)
    client.client = httpx.Client(
        transport=httpx.MockTransport(handle), base_url="http://server"
    )
    client.api_endpoint = "/v1/chat/completions"
    client.logger = MagicMock()
    return client, requests, replies


def chat(client, messages=None):
    return client._inference_chat({
        "query": messages or [{"role": "user", "content": "battery?"}],
        "max_new_tokens": 100,
    })


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ('{"unit": "percent"}', {"unit": "percent"}),
        ({"unit": "percent"}, {"unit": "percent"}),
        ("", {}),
        ("  ", {}),
        (None, {}),
    ],
    ids=["json_string", "object_from_older_servers", "empty", "blank", "none"],
)
def test_tool_arguments_reach_components_as_a_dict(server, arguments, expected):
    client, _, replies = server
    replies["tool_calls"] = [tool_call(arguments)]

    (call,) = chat(client)["tool_calls"]

    assert call["function"]["arguments"] == expected
    assert call["id"] == "call_abc"


@pytest.mark.parametrize("arguments", ['{"unit": ', "[1, 2]", '"percent"', 3], ids=str)
def test_a_call_with_arguments_that_are_not_an_object_is_dropped(server, arguments):
    client, _, replies = server
    replies["tool_calls"] = [tool_call(arguments)]

    assert chat(client)["tool_calls"] == []
    client.logger.error.assert_called_once()


def test_the_other_calls_are_kept(server):
    client, _, replies = server
    replies["tool_calls"] = [
        tool_call('{"unit": ', name="broken", call_id="call_1"),
        tool_call('{"unit": "percent"}', call_id="call_2"),
    ]

    calls = chat(client)["tool_calls"]

    assert [call["id"] for call in calls] == ["call_2"]
    client.logger.error.assert_called_once()


def test_tool_call_arguments_are_sent_as_a_json_string(server):
    client, requests, _ = server
    messages = [
        {"role": "user", "content": "battery?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call({"unit": "percent"})],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "87 percent"},
    ]

    chat(client, messages)

    (request,) = requests
    sent = request["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(sent) == {"unit": "percent"}
    assert request["messages"][2]["tool_call_id"] == "call_abc"
    # the component's own messages are left as they are
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"unit": "percent"}


# --- Decisions ---------------------------------------------------------------

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

ANSWERS = {
    "stop": {"type": "noul", "noul": 0.97},
    "room": {
        "type": "choice",
        "choice": "kitchen",
        "probabilities": {"kitchen": 0.9, "bedroom": 0.1},
        "confidence": 0.8,
    },
}


@pytest.fixture
def decision_server():
    """A client with a decision model, talking to a fake server, which records
    the decision requests it gets and replies with the answers or the error it
    is given"""
    requests = []
    replies = {"status": 200}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "lev"}]})
        requests.append((request.url.path, json.loads(request.content)))
        if replies["status"] != 200:
            return httpx.Response(
                replies["status"],
                json={"error": {"message": "This model is not a decision model"}},
            )
        return httpx.Response(200, json={"model": "lev", "answers": ANSWERS})

    client = GenericHTTPClient(GenericDecisionModel(name="lev", checkpoint="lev"))
    client.client = httpx.Client(
        transport=httpx.MockTransport(handle), base_url="http://server"
    )
    client.logger = MagicMock()
    client.initialize()
    return client, requests, replies


def test_a_decision_model_is_asked_at_systemone(decision_server):
    client, requests, _ = decision_server

    result = client.inference({"state": "Stop!", "questions": QUESTIONS})

    assert result == {"output": ANSWERS}
    ((path, body),) = requests
    assert path == "/v1/systemone"
    assert body == {"model": "lev", "state": "Stop!", "questions": QUESTIONS}


def test_images_are_sent_as_data_urls(decision_server):
    client, requests, _ = decision_server
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    client.inference({"state": {}, "questions": QUESTIONS, "images": [image]})

    ((_, body),) = requests
    (url,) = body["images"]
    prefix = "data:image/png;base64,"
    assert url.startswith(prefix)
    assert base64.b64decode(url[len(prefix) :]).startswith(b"\x89PNG")


def test_a_server_error_gives_no_answers(decision_server):
    client, _, replies = decision_server
    replies["status"] = 501

    assert client.inference({"state": "Stop!", "questions": QUESTIONS}) is None
    client.logger.error.assert_called_once()


def test_a_decision_model_takes_questions_not_tools():
    decision = GenericHTTPClient(GenericDecisionModel(name="lev", checkpoint="lev"))
    llm = GenericHTTPClient(GenericLLM(name="m", checkpoint="m"))

    assert decision.supports_decisions and not decision.supports_tool_calls
    assert llm.supports_tool_calls and not llm.supports_decisions


def test_a_serialized_decision_client_is_rebuilt():
    """As a component's client is rebuilt in its own process"""
    client = GenericHTTPClient(
        GenericDecisionModel(name="lev", checkpoint="lev"), port=8090
    )

    rebuilt = GenericHTTPClient(**client.serialize())

    assert rebuilt.supports_decisions
    assert rebuilt.model_init_params == {"checkpoint": "lev"}

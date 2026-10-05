"""Tool calls through the client for OpenAI-compatible servers.

Components get the arguments of a tool call as a dict from every client. Most
OpenAI-compatible servers (vLLM, SGLang, llama.cpp, lmdeploy, ...) send them as
a JSON string, while some (e.g. TGI before 3.2) send a JSON object; both are
read. They are sent back to the server as a JSON string, as the OpenAI format
expects.
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest

from agents.clients.generic import GenericHTTPClient

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

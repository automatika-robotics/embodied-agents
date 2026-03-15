"""Tests for LocalLLM wrapper — no ROS needed."""

import sys
import json
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_onnxruntime_genai():
    """Mock onnxruntime_genai before importing LocalLLM."""
    mock_og = MagicMock()
    mock_og.Model.return_value = MagicMock()
    mock_og.Tokenizer.return_value = MagicMock()
    with patch.dict(sys.modules, {"onnxruntime_genai": mock_og}):
        yield mock_og


@pytest.fixture
def local_llm(mock_onnxruntime_genai):
    from agents.utils.local_llm import LocalLLM

    llm = LocalLLM.__new__(LocalLLM)
    llm._og = mock_onnxruntime_genai
    llm.model = MagicMock()
    llm.tokenizer = MagicMock()
    llm.device = "cpu"
    llm.ncpu = 1
    return llm


def _mock_generator(og_mock, tokenizer, token_ids, decoded_texts):
    """Set up a mock Generator that yields the given tokens then reports done."""
    mock_gen = MagicMock()
    # is_done returns False for each token, then True
    mock_gen.is_done.side_effect = [False] * len(token_ids) + [True]
    mock_gen.get_next_tokens.side_effect = [[t] for t in token_ids]
    og_mock.Generator.return_value = mock_gen

    mock_stream = MagicMock()
    mock_stream.decode.side_effect = decoded_texts
    tokenizer.create_stream.return_value = mock_stream

    return mock_gen


class TestApplyChatTemplate:
    def test_basic_template(self, local_llm):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = local_llm._apply_chat_template(messages)
        assert "<|im_start|>system\nYou are helpful.<|im_end|>" in result
        assert "<|im_start|>user\nHello<|im_end|>" in result
        assert result.endswith("<|im_start|>assistant\n")


class TestParseToolCalls:
    def test_valid_tool_call(self, local_llm):
        text = '<tool_call>{"name": "my_func", "arguments": {"a": 1}}</tool_call>'
        calls = local_llm._parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "my_func"
        assert calls[0]["function"]["arguments"] == {"a": 1}

    def test_invalid_json_skipped(self, local_llm):
        text = "<tool_call>not valid json</tool_call>"
        calls = local_llm._parse_tool_calls(text)
        assert len(calls) == 0

    def test_multiple_tool_calls(self, local_llm):
        text = (
            '<tool_call>{"name": "f1", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "f2", "arguments": {"x": 2}}</tool_call>'
        )
        calls = local_llm._parse_tool_calls(text)
        assert len(calls) == 2


class TestCallNonStreaming:
    def test_returns_output(self, local_llm):
        local_llm.tokenizer.encode.return_value = [1, 2, 3]
        _mock_generator(
            local_llm._og,
            local_llm.tokenizer,
            token_ids=[4, 5],
            decoded_texts=["Hello", " there!"],
        )

        result = local_llm({"query": [{"role": "user", "content": "Hi"}]})
        assert result["output"] == "Hello there!"

    def test_stops_at_end_token(self, local_llm):
        local_llm.tokenizer.encode.return_value = [1]
        _mock_generator(
            local_llm._og,
            local_llm.tokenizer,
            token_ids=[2, 3],
            decoded_texts=["Response", "<|im_end|>"],
        )

        result = local_llm({"query": [{"role": "user", "content": "Test"}]})
        assert result["output"] == "Response"


class TestCallStreaming:
    def test_returns_generator(self, local_llm):
        local_llm.tokenizer.encode.return_value = [1]

        _mock_generator(
            local_llm._og,
            local_llm.tokenizer,
            token_ids=[1, 2],
            decoded_texts=["Hello", " world"],
        )

        result = local_llm(
            {"query": [{"role": "user", "content": "Hi"}]}, stream=True
        )
        assert "output" in result
        tokens = list(result["output"])
        assert tokens == ["Hello", " world"]


class TestCallWithTools:
    def test_tools_parsed_from_output(self, local_llm):
        tool_call_text = '<tool_call>{"name": "route_to_nav", "arguments": {}}</tool_call>'
        local_llm.tokenizer.encode.return_value = [1]
        _mock_generator(
            local_llm._og,
            local_llm.tokenizer,
            token_ids=[2],
            decoded_texts=[tool_call_text],
        )

        tools = [{"type": "function", "function": {"name": "route_to_nav"}}]
        result = local_llm(
            {"query": [{"role": "user", "content": "Go"}], "tools": tools}
        )
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "route_to_nav"

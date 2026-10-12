"""Tests for LLM component — requires rclpy."""

import json

import pytest
from unittest.mock import MagicMock, PropertyMock

from agents.config import LLMConfig
from agents.ros import Topic
from agents.components.llm import LLM
from agents.utils import get_prompt_template
from tests.conftest import mock_component_internals


@pytest.fixture
def llm(rclpy_init, mock_model_client):
    """Create an LLM with a mock model client."""
    comp = LLM(
        inputs=[Topic(name="in", msg_type="String")],
        outputs=[Topic(name="out", msg_type="String")],
        model_client=mock_model_client,
        config=LLMConfig(),
        component_name="test_llm",
    )
    mock_component_internals(comp)
    return comp


class TestLLMConstruction:
    def test_with_model_client(self, rclpy_init, mock_model_client):
        comp = LLM(
            inputs=[Topic(name="in", msg_type="String")],
            outputs=[Topic(name="out", msg_type="String")],
            model_client=mock_model_client,
            config=LLMConfig(),
            component_name="test_llm_client",
        )
        assert comp.model_client is mock_model_client

    def test_with_local_model(self, rclpy_init):
        comp = LLM(
            inputs=[Topic(name="in", msg_type="String")],
            outputs=[Topic(name="out", msg_type="String")],
            config=LLMConfig(enable_local_model=True),
            component_name="test_llm_local",
        )
        assert comp.config.enable_local_model is True

    def test_no_client_no_local_raises(self, rclpy_init):
        with pytest.raises(RuntimeError):
            LLM(
                inputs=[Topic(name="in", msg_type="String")],
                outputs=[Topic(name="out", msg_type="String")],
                config=LLMConfig(),
                component_name="test_llm_fail",
            )

    def test_system_prompt_in_messages(self, rclpy_init, mock_model_client):
        comp = LLM(
            inputs=[Topic(name="in", msg_type="String")],
            outputs=[Topic(name="out", msg_type="String")],
            model_client=mock_model_client,
            config=LLMConfig(_system_prompt="You are a robot"),
            component_name="test_llm_sp",
        )
        assert comp.history.request()[0] == {
            "role": "system",
            "content": "You are a robot",
        }

    def test_a_system_prompt_set_after_init_is_in_the_messages(self, llm):
        llm.set_system_prompt("You are a robot")
        assert llm.history.request() == [
            {"role": "system", "content": "You are a robot"}
        ]

    def test_no_system_prompt_empty_messages(self, rclpy_init, mock_model_client):
        comp = LLM(
            inputs=[Topic(name="in", msg_type="String")],
            outputs=[Topic(name="out", msg_type="String")],
            model_client=mock_model_client,
            config=LLMConfig(),
            component_name="test_llm_nosp",
        )
        assert comp.history.request() == []


class TestLLMCreateInput:
    def test_from_trigger_topic(self, llm):
        trigger = Topic(name="in", msg_type="String")
        mock_cb = MagicMock()
        mock_cb.get_output.return_value = "hello"
        llm.trig_callbacks = {"in": mock_cb}
        llm.callbacks = {}

        result = llm._create_input(topic=trigger)
        assert result is not None
        assert result["query"][-1]["content"] == "hello"

    def test_returns_none_no_query(self, llm):
        trigger = Topic(name="in", msg_type="String")
        mock_cb = MagicMock()
        mock_cb.get_output.return_value = None
        llm.trig_callbacks = {"in": mock_cb}
        llm.callbacks = {}

        result = llm._create_input(topic=trigger)
        assert result is None

    def test_chat_reset(self, llm):
        llm.config.chat_history = True
        llm.history.add_turn("old")
        llm.history.add_reply("fine")
        trigger = Topic(name="in", msg_type="String")
        mock_cb = MagicMock()
        mock_cb.get_output.return_value = "chat reset"
        llm.trig_callbacks = {"in": mock_cb}
        llm.callbacks = {}

        llm._execution_step(topic=trigger)
        assert llm.history.request() == []

    def test_chat_reset_keeps_the_system_prompt(self, llm):
        llm.config.chat_history = True
        llm.config._system_prompt = llm.history.system_prompt = "You are a robot"
        system = {"role": "system", "content": "You are a robot"}
        llm.history.add_turn("old")
        llm.history.add_reply("fine")
        trigger = Topic(name="in", msg_type="String")
        mock_cb = MagicMock()
        mock_cb.get_output.return_value = "chat reset"
        llm.trig_callbacks = {"in": mock_cb}
        llm.callbacks = {}

        llm._execution_step(topic=trigger)
        assert llm.history.request() == [system]

    @pytest.mark.parametrize("phrase", ["Chat reset.", "CHAT RESET!", " chat reset "])
    def test_chat_reset_phrase_ignores_case_and_punctuation(self, llm, phrase):
        llm.config.chat_history = True
        llm.history.add_turn("old")
        llm.history.add_reply("fine")
        trigger = Topic(name="in", msg_type="String")
        mock_cb = MagicMock()
        mock_cb.get_output.return_value = phrase
        llm.trig_callbacks = {"in": mock_cb}
        llm.callbacks = {}

        llm._execution_step(topic=trigger)
        assert llm.history.request() == []

    def test_a_reset_is_not_a_missing_input(self, llm, mock_model_client):
        llm.config.chat_history = True

        llm._execution_step(topic=_ask(llm, "chat reset"))

        mock_model_client.inference.assert_not_called()
        llm.get_logger().warning.assert_not_called()
        assert "reset" in llm.get_logger().debug.call_args.args[0]

    def test_chat_reset_needs_history_on(self, llm):
        llm.config.chat_history = False
        trigger = Topic(name="in", msg_type="String")
        mock_cb = MagicMock()
        mock_cb.get_output.return_value = "chat reset"
        llm.trig_callbacks = {"in": mock_cb}
        llm.callbacks = {}

        result = llm._create_input(topic=trigger)
        assert result["query"][-1]["content"] == "chat reset"

    def test_includes_tools(self, llm):
        llm.config._tool_descriptions = [{"function": {"name": "test_tool"}}]
        trigger = Topic(name="in", msg_type="String")
        mock_cb = MagicMock()
        mock_cb.get_output.return_value = "hello"
        llm.trig_callbacks = {"in": mock_cb}
        llm.callbacks = {}

        result = llm._create_input(topic=trigger)
        assert "tools" in result


def _ask(llm, text):
    """A trigger message on the input topic"""
    trigger = Topic(name="in", msg_type="String")
    mock_cb = MagicMock()
    mock_cb.get_output.return_value = text
    llm.trig_callbacks = {"in": mock_cb}
    llm.callbacks = {}
    return trigger


def _retrieving(llm, document):
    """RAG that finds one document"""
    llm.config.enable_rag = True
    llm.db_client = MagicMock()
    llm.db_client.query.return_value = {
        "output": {"metadatas": [{}], "documents": [document]}
    }


class TestLLMHistory:
    def test_without_history_the_request_is_the_system_prompt_and_the_turn(
        self, llm, mock_model_client
    ):
        llm.config.chat_history = False
        llm.config._system_prompt = llm.history.system_prompt = "sys"
        _retrieving(llm, "doc")

        llm._execution_step(topic=_ask(llm, "hello"))
        second = llm._create_input(topic=_ask(llm, "again"))

        assert mock_model_client.inference.call_args.args[0]["query"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "doc\nhello"},
        ]
        assert second["query"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "doc\nagain"},
        ]

    def test_without_history_a_template_renders_the_turn(self, llm):
        llm.config.chat_history = False
        llm.component_prompt = get_prompt_template("Answer briefly: {{ in }}")

        result = llm._create_input(topic=_ask(llm, "hello"))

        assert result["query"] == [{"role": "user", "content": "Answer briefly: hello"}]

    def test_earlier_turns_are_sent_as_said_without_their_documents(
        self, llm, mock_model_client
    ):
        llm.config.chat_history = True
        _retrieving(llm, "doc")

        llm._execution_step(topic=_ask(llm, "hello"))
        second = llm._create_input(topic=_ask(llm, "again"))

        assert second["query"] == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "test"},
            {"role": "user", "content": "doc\nagain"},
        ]

    def test_a_failed_call_leaves_no_turn(self, llm, mock_model_client):
        llm.config.chat_history = True
        mock_model_client.inference.return_value = None

        llm._execution_step(topic=_ask(llm, "hello"))
        second = llm._create_input(topic=_ask(llm, "again"))

        assert second["query"] == [{"role": "user", "content": "again"}]

    def test_the_history_is_capped_in_whole_turns(self, llm, mock_model_client):
        llm.config.chat_history = True
        llm.config.history_size = 3
        # a fresh result each call: publishing consumes the dict
        mock_model_client.inference.side_effect = lambda _: {"output": "test"}

        for text in ["one", "two", "three", "four"]:
            llm._execution_step(topic=_ask(llm, text))
        # four turns is within a third over the cap of three
        assert [m["content"] for m in llm.history.request() if m["role"] == "user"] == [
            "one",
            "two",
            "three",
            "four",
        ]

        llm._execution_step(topic=_ask(llm, "five"))
        # five is past it: trimmed back to three, the newest
        assert llm.history.request() == [
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "test"},
            {"role": "user", "content": "four"},
            {"role": "assistant", "content": "test"},
            {"role": "user", "content": "five"},
            {"role": "assistant", "content": "test"},
        ]

    def test_summarize_mode_folds_the_dropped_turns_by_one_call(
        self, llm, mock_model_client
    ):
        llm.config.chat_history = True
        llm.config.history_size = 2
        llm.config.history_compaction = "summarize"

        def model(request):
            if request["query"][0]["content"] == LLM._SUMMARY_PROMPT:
                assert request["stream"] is False
                assert "user: one" in request["query"][1]["content"]
                return {"output": "<think>hm</think>The user said one."}
            return {"output": "test"}

        mock_model_client.inference.side_effect = model

        for text in ["one", "two", "three", "four"]:
            llm._execution_step(topic=_ask(llm, text))

        sent = mock_model_client.inference.call_args.args[0]["query"]
        assert sent[0] == {
            "role": "user",
            "content": "Summary of the conversation so far:\nThe user said one.",
        }
        assert sent[1] == {"role": "assistant", "content": "Understood."}
        assert [m["content"] for m in sent[2:] if m["role"] == "user"] == [
            "three",
            "four",
        ]

    def test_a_failed_summary_lets_the_turns_go(self, llm, mock_model_client):
        llm.config.chat_history = True
        llm.config.history_size = 2
        llm.config.history_compaction = "summarize"
        mock_model_client.inference.side_effect = lambda request: (
            None
            if request["query"][0]["content"] == LLM._SUMMARY_PROMPT
            else {"output": "t"}
        )

        for text in ["one", "two", "three", "four"]:
            llm._execution_step(topic=_ask(llm, text))

        sent = mock_model_client.inference.call_args.args[0]["query"]
        assert sent[0] == {"role": "user", "content": "three"}
        assert "forgotten" in llm.get_logger().warning.call_args.args[0]

    def test_a_tool_calling_turn_is_kept_whole(self, llm, mock_model_client):
        llm.config.chat_history = True
        llm.config._tool_descriptions = [{"function": {"name": "test_tool"}}]
        llm._external_processors = {
            "test_tool": ([MagicMock(return_value="42")], "tool")
        }
        llm.config._tool_response_flags = {"test_tool": True}
        mock_model_client.inference.side_effect = [
            {
                "output": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "test_tool", "arguments": {}}}
                ],
            },
            {"output": "it is 42"},
        ]

        llm._execution_step(topic=_ask(llm, "how many?"))

        assert [m["role"] for m in llm.history.request()] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert llm.history.request()[1]["tool_calls"][0]["id"] == "c1"
        assert llm.history.request()[2] == {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "42",
        }
        # the reply to the results was asked with the whole turn so far
        second_request = mock_model_client.inference.call_args.args[0]["query"]
        assert [m["role"] for m in second_request] == ["user", "assistant", "tool"]


class TestLLMHistoryActions:
    def test_reset_history_keeps_the_system_prompt(self, llm):
        llm.history.system_prompt = "sys"
        llm.history.add_turn("hello")
        llm.history.add_reply("hi")

        ok, message = llm.reset_history.__wrapped__(llm)

        assert ok and message == "Chat history reset"
        assert llm.history.request() == [{"role": "system", "content": "sys"}]

    def test_get_history_is_the_conversation_as_text(self, llm):
        llm.history.add_turn("hello", "doc\nhello")
        llm.history.add_reply(
            "", tool_calls=[{"id": "c1", "function": {"name": "move"}}]
        )
        llm.history.add_tool_result("c1", "moved")
        llm.history.add_reply("hi")

        ok, text = llm.get_history.__wrapped__(llm)

        assert ok
        assert text == (
            "Conversation:\nuser: hello\nassistant: [called move]\n"
            "tool: moved\nassistant: hi"
        )

    def test_get_history_says_when_there_is_none(self, llm):
        assert llm.get_history.__wrapped__(llm) == (True, "The chat history is empty")

    def test_reading_is_a_planning_tool_and_resetting_an_execution_one(self, llm):
        assert json.loads(llm.get_history._action_description)["phase"] == "planning"
        assert json.loads(llm.reset_history._action_description)["phase"] == "execution"


class TestLLMHistoryBudget:
    def test_the_resolved_budget_trims_the_requests(self, llm, mock_model_client):
        llm.config.chat_history = True
        mock_model_client.inference.side_effect = lambda _: {"output": "test"}
        llm._history_budget = 190

        for text in ["one " * 40, "two " * 40, "three " * 40, "four " * 40]:
            llm._execution_step(topic=_ask(llm, text))

        assert len(llm.history.turns) < 4
        assert llm.history.turns[-1].utterance.startswith("four")


class TestLLMExecutionStep:
    def test_non_streaming(self, llm, mock_model_client):
        trigger = Topic(name="in", msg_type="String")
        mock_cb = MagicMock()
        mock_cb.get_output.return_value = "hello"
        llm.trig_callbacks = {"in": mock_cb}
        llm.callbacks = {}

        llm._execution_step(topic=trigger)
        mock_model_client.inference.assert_called_once()
        llm.publishers_dict["out"].publish.assert_called_once()

    def test_no_input_skips_inference(self, llm, mock_model_client):
        trigger = Topic(name="in", msg_type="String")
        mock_cb = MagicMock()
        mock_cb.get_output.return_value = None
        llm.trig_callbacks = {"in": mock_cb}
        llm.callbacks = {}

        llm._execution_step(topic=trigger)
        mock_model_client.inference.assert_not_called()

    def test_with_tool_calls(self, llm, mock_model_client):
        llm.config._tool_descriptions = [{"function": {"name": "test_tool"}}]
        mock_model_client.inference.return_value = {
            "output": "calling tool",
            "tool_calls": [
                {"function": {"name": "test_tool", "arguments": {"x": "1"}}}
            ],
        }

        # Register a mock tool
        mock_tool = MagicMock(return_value="tool_result")
        llm._external_processors = {
            "test_tool": ([mock_tool], "tool"),
        }
        llm.config._tool_response_flags = {"test_tool": False}

        trigger = Topic(name="in", msg_type="String")
        mock_cb = MagicMock()
        mock_cb.get_output.return_value = "do something"
        llm.trig_callbacks = {"in": mock_cb}
        llm.callbacks = {}

        llm._execution_step(topic=trigger)
        mock_tool.assert_called_once()

    def test_with_tool_calls_across_processes(
        self, llm, mock_model_client, launcher_processors
    ):
        """A tool registered in the recipe lives in the launcher process in a
        multiprocess launch, and is called from the component's own process"""
        calls = []

        def add_numbers(a: int, b: int) -> str:
            calls.append((a, b))
            return f"sum is {a + b}"

        llm.register_tool(
            tool=add_numbers,
            tool_description={"function": {"name": "add_numbers"}},
        )
        # as in the component's process, which gets the serialized processors
        llm._external_processors_json = launcher_processors(llm)

        mock_model_client.inference.return_value = {
            "output": "calling tool",
            "tool_calls": [
                {
                    "function": {
                        "name": "add_numbers",
                        "arguments": {"a": 2, "b": "3"},
                    }
                }
            ],
        }
        mock_cb = MagicMock()
        mock_cb.get_output.return_value = "add 2 and 3"
        llm.trig_callbacks = {"in": mock_cb}
        llm.callbacks = {}

        llm._execution_step(topic=Topic(name="in", msg_type="String"))

        assert calls == [(2, 3)]
        assert llm.history.request()[-1]["role"] == "tool"
        assert llm.history.request()[-1]["content"] == "sum is 5"


class TestLLMToolCallHistory:
    """The history records a tool call exchange as OpenAI-compatible servers
    expect it: the assistant message with its tool calls, then one tool message
    per call with the call's id"""

    def _call_tools(self, llm, mock_model_client, tool_calls, tool, follow_up=None):
        """Run a step in which the model calls the tools. With a follow_up
        reply, the results are sent back to the model, which replies with it"""
        llm.config._tool_descriptions = [{"function": {"name": "get_battery"}}]
        llm.config._tool_response_flags = {"get_battery": follow_up is not None}
        llm._external_processors = {"get_battery": ([tool], "Function")}
        replies = [{"output": "", "tool_calls": tool_calls}]
        if follow_up is not None:
            replies.append({"output": follow_up})
        mock_model_client.inference.side_effect = replies
        mock_cb = MagicMock()
        mock_cb.get_output.return_value = "battery?"
        llm.trig_callbacks = {"in": mock_cb}
        llm.callbacks = {}
        llm._execution_step(topic=Topic(name="in", msg_type="String"))

    def test_the_calls_and_their_results_are_paired_by_id(self, llm, mock_model_client):
        calls = [
            {
                "id": "call_abc",
                "function": {"name": "get_battery", "arguments": {"unit": "percent"}},
            }
        ]

        self._call_tools(llm, mock_model_client, calls, lambda unit: f"87 {unit}")

        assistant, tool = llm.history.request()[-2:]
        assert assistant["role"] == "assistant"
        assert assistant["tool_calls"] == [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_battery", "arguments": {"unit": "percent"}},
            }
        ]
        assert tool == {
            "role": "tool",
            "tool_call_id": "call_abc",
            "content": "87 percent",
        }

    def test_values_written_as_json_strings_reach_the_tool_decoded(
        self, llm, mock_model_client
    ):
        received = {}

        def get_battery(**kwargs):
            received.update(kwargs)
            return "87"

        arguments = {"count": "3", "where": ' {"room": "lab"} ', "name": " kitchen "}
        calls = [{"function": {"name": "get_battery", "arguments": arguments}}]

        self._call_tools(llm, mock_model_client, calls, get_battery)

        assert received == {"count": 3, "where": {"room": "lab"}, "name": "kitchen"}

    def test_calls_without_an_id_get_one(self, llm, mock_model_client):
        """Ollama and the built-in local model give no ids"""
        calls = [{"function": {"name": "get_battery", "arguments": {"unit": "%"}}}]

        self._call_tools(llm, mock_model_client, calls, lambda unit: "87")

        assistant, tool = llm.history.request()[-2:]
        assert assistant["tool_calls"][0]["id"] == tool["tool_call_id"] == "call_0"

    def test_a_failed_call_leaves_no_unanswered_calls(self, llm, mock_model_client):
        def failing(unit):
            raise RuntimeError("boom")

        calls = [{"function": {"name": "get_battery", "arguments": {"unit": "%"}}}]

        self._call_tools(llm, mock_model_client, calls, failing)

        assert llm.history.request()[-1]["role"] == "assistant"
        assert "tool_calls" not in llm.history.request()[-1]

    def test_the_reply_to_the_results_is_recorded_without_think_tokens(
        self, llm, mock_model_client
    ):
        calls = [
            {
                "id": "call_abc",
                "function": {"name": "get_battery", "arguments": {"unit": "%"}},
            }
        ]

        self._call_tools(
            llm,
            mock_model_client,
            calls,
            lambda unit: "87",
            follow_up="<think>\nthe tool said 87\n</think>\n\nBattery is at 87%.",
        )

        call, result, reply = llm.history.request()[-3:]
        assert call["tool_calls"][0]["id"] == "call_abc"
        assert result["tool_call_id"] == "call_abc"
        assert reply == {"role": "assistant", "content": "Battery is at 87%."}
        assert (
            llm.publishers_dict["out"].publish.call_args[0][0] == "Battery is at 87%."
        )


class TestLLMThinkTokens:
    THINKING = "<think>\nreasoning here\n</think>\n\nParis."

    def _run_step(self, llm, mock_model_client, text=None):
        mock_model_client.inference.return_value = {"output": text or self.THINKING}
        mock_cb = MagicMock()
        mock_cb.get_output.return_value = "capital of France?"
        llm.trig_callbacks = {"in": mock_cb}
        llm.callbacks = {}
        llm._execution_step(topic=Topic(name="in", msg_type="String"))
        return llm.publishers_dict["out"].publish.call_args[0][0]

    def test_stripped_by_default(self, llm, mock_model_client):
        assert self._run_step(llm, mock_model_client) == "Paris."

    def test_kept_when_disabled(self, llm, mock_model_client):
        llm.config.strip_think_tokens = False
        assert self._run_step(llm, mock_model_client) == self.THINKING

    def test_unterminated_think_block_stripped_with_warning(
        self, llm, mock_model_client
    ):
        """A block truncated by max_new_tokens has no closing tag but must
        still be stripped, and the component should warn about it."""
        truncated = "<think>\nreasoning cut off by the token limit"
        assert self._run_step(llm, mock_model_client, text=truncated) == ""
        llm.get_logger.return_value.warning.assert_called_once()

    def test_stream_tokens_filtered(self, llm):
        """Think tokens are dropped, including the whitespace models emit
        after the closing tag — the answer must not start with blank lines."""
        llm.config.break_character = ""
        llm._in_think_block = False
        llm._swallow_stream_ws = False
        llm.result_partial = []
        llm.result_complete = []

        for token in ["<think>", "\n", "reasoning", "</think>", "\n\n", "\nParis."]:
            llm._LLM__process_stream_token(token)

        publish = llm.publishers_dict["out"].publish
        publish.assert_called_once()
        assert publish.call_args[0][0] == "Paris."

    def test_stream_unterminated_think_warns(self, llm):
        """A stream that ends inside a think block must warn, same as the
        non-streaming path."""
        llm.config.break_character = ""
        llm._in_think_block = False
        llm._swallow_stream_ws = False
        llm.result_partial = []
        llm.result_complete = []

        for token in ["<think>", "reasoning cut off by the token limit"]:
            llm._LLM__process_stream_token(token)
        llm._LLM__finalize_stream()

        llm.get_logger.return_value.warning.assert_called_once()
        llm.publishers_dict["out"].publish.assert_not_called()


class TestLLMStreamCleanup:
    def test_generator_closed_when_publish_raises(self, llm):
        """An abandoned stream must be finalized immediately so it releases
        held resources (e.g. the local model lock) instead of waiting for
        garbage collection."""
        closed = []

        def stream():
            try:
                yield "one. "
                yield "two. "
            finally:
                closed.append(True)

        llm.config.break_character = ""
        llm._in_think_block = False
        llm._swallow_stream_ws = False
        llm.result_partial = []
        llm.result_complete = []
        llm.publishers_dict["out"].publish.side_effect = RuntimeError(
            "publisher destroyed"
        )

        # keep a reference so finalization cannot come from refcounting
        result = {"output": stream()}
        llm._LLM__handle_streaming_generator(result)

        assert closed == [True]


class TestLLMWarmup:
    def test_with_model_client(self, llm, mock_model_client):
        llm._warmup()
        assert mock_model_client.inference.call_count == 2

    def test_with_local_model(self, rclpy_init):
        comp = LLM(
            inputs=[Topic(name="in", msg_type="String")],
            outputs=[Topic(name="out", msg_type="String")],
            config=LLMConfig(enable_local_model=True),
            component_name="test_warmup_local",
        )
        mock_component_internals(comp)
        comp.model_client = None
        comp.local_model = MagicMock(return_value={"output": "ok"})
        comp._warmup()
        assert comp.local_model.call_count == 2


class TestDetections3DContext:
    """A Detections3D input enriches the prompt with metric positions."""

    def test_detections3d_string_reaches_the_prompt_template(self, llm):
        from agents.utils import get_prompt_template

        trigger = Topic(name="in", msg_type="String")
        mock_cb = MagicMock()
        mock_cb.get_output.return_value = "where is the orange?"
        llm.trig_callbacks = {"in": mock_cb}

        mock_d3 = MagicMock()
        mock_d3.get_output.return_value = "In odom: orange at (2.00, 3.00, 0.05)"
        mock_d3.input_topic = Topic(name="d3", msg_type="Detections3D")
        llm.callbacks = {"d3": mock_d3}
        llm.component_prompt = get_prompt_template("Visible objects: {{ d3 }}")

        result = llm._create_input(topic=trigger)

        assert "orange at (2.00, 3.00, 0.05)" in result["query"][-1]["content"]


class TestComponentActionToolResults:
    """The LLM component calls a component action over its ExecuteMethod
    service and turns the response into the tool result the model sees"""

    def _call(self, llm, response):
        client = MagicMock()
        client.send_request.return_value = response
        llm._component_clients = {"memory": client}
        return llm._execute_component_method("memory", "start_episode", name="tidy")

    def _response(self, success, message=""):
        response = MagicMock()
        response.success = success
        response.response_json = '"' + message + '"' if success and message else ""
        response.error_msg = "" if success else message
        return response

    def test_the_actions_message_is_the_tool_result(self, llm):
        result = self._call(llm, self._response(True, "Episode 'tidy' started"))

        assert result == "Episode 'tidy' started"

    def test_an_empty_message_is_a_confirmation(self, llm):
        assert "executed successfully" in self._call(llm, self._response(True))

    def test_a_failure_is_an_error_line(self, llm):
        result = self._call(llm, self._response(False, "no such layer"))

        assert result.startswith("Error:") and "no such layer" in result

    def test_no_response_is_an_error_line(self, llm):
        """The client returns None when the service is unavailable or the call
        times out. That used to raise inside the result decoding"""
        result = self._call(llm, None)

        assert result.startswith("Error:") and "no response" in result

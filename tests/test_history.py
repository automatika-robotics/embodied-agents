"""The chat history a model component keeps and the requests it builds.

A conversation is turns. The current turn is sent as it was dressed up for
its call, older turns as what the user said and what was answered, so the
documents and the context of an earlier call are not replayed. Turns are
dropped whole, in chunks that leave the request well under its limit, and a
turn that got no reply is not a turn.
"""

import numpy as np
import pytest

from types import SimpleNamespace
from unittest.mock import MagicMock

from agents.utils import (
    ChatHistory,
    estimate_tokens,
    matches_phrase,
    resolve_history_budget,
)


def image():
    return np.zeros((4, 4, 3), dtype=np.uint8)


def conversation(turns, system=None):
    """A history with that many answered turns"""
    history = ChatHistory(system)
    for index in range(turns):
        history.add_turn(f"question {index}", f"context\nquestion {index}")
        history.add_reply(f"answer {index}")
    return history


class TestRequest:
    def test_the_current_turn_goes_in_full_and_older_ones_as_said(self):
        history = conversation(1, system="You are a robot")
        history.add_turn("what next?", "doc\nwhat next?")

        assert history.request() == [
            {"role": "system", "content": "You are a robot"},
            {"role": "user", "content": "question 0"},
            {"role": "assistant", "content": "answer 0"},
            {"role": "user", "content": "doc\nwhat next?"},
        ]

    def test_without_a_system_prompt_the_first_message_is_the_user(self):
        history = ChatHistory()
        history.add_turn("hi")
        assert history.request() == [{"role": "user", "content": "hi"}]

    def test_an_unanswered_turn_is_dropped_by_the_next(self):
        history = ChatHistory()
        history.add_turn("the call failed")
        history.add_turn("again")

        assert history.request() == [{"role": "user", "content": "again"}]

    def test_a_turn_answered_with_tool_calls_alone_is_a_turn(self):
        history = ChatHistory()
        history.add_turn("go")
        history.add_reply("", tool_calls=[{"id": "c1", "function": {"name": "move"}}])
        history.add_tool_result("c1", "moved")
        history.add_turn("and now?")

        assert [m["role"] for m in history.request()] == [
            "user",
            "assistant",
            "tool",
            "user",
        ]

    def test_reset_keeps_the_system_prompt(self):
        history = conversation(2, system="sys")
        history.reset()
        assert history.request() == [{"role": "system", "content": "sys"}]

    def test_a_reply_with_no_turn_is_let_go(self):
        history = ChatHistory()
        history.add_reply("a stream that ended after a reset")
        assert history.request() == []


class TestTrimming:
    def test_the_turn_cap_lets_the_history_grow_a_third_before_trimming(self):
        history = conversation(5)
        # five turns, a cap of four: a third over the cap is allowed
        assert len(history.request(max_turns=4)) == 10

        history.add_turn("question 5")
        messages = history.request(max_turns=4)

        # six turns is past it: trimmed back to four, the newest
        assert [m["content"] for m in messages if m["role"] == "user"] == [
            "question 2",
            "question 3",
            "question 4",
            "question 5",
        ]
        assert len(history.turns) == 4

    def test_the_budget_drops_whole_turns_to_three_quarters(self):
        history = conversation(6)
        history.add_turn("question 6")
        full = history.tokens()
        budget = full - 1

        history.request(budget=budget)

        assert history.tokens() <= budget * ChatHistory.TRIM_TO
        # the oldest turns went, the newest stayed, each turn whole
        roles = [m["role"] for m in history.request()]
        assert (
            roles[-1] == "user" and roles.count("user") == roles.count("assistant") + 1
        )
        assert history.turns[-1].utterance == "question 6"

    def test_the_current_turn_is_never_dropped(self):
        history = ChatHistory()
        history.add_turn("a question longer than the budget allows")

        messages = history.request(budget=2)

        assert messages == [
            {"role": "user", "content": "a question longer than the budget allows"}
        ]

    def test_a_tool_calling_turn_is_dropped_whole(self):
        history = ChatHistory()
        history.add_turn("go")
        history.add_reply("", tool_calls=[{"id": "c1", "function": {"name": "move"}}])
        history.add_tool_result("c1", "moved")
        history.add_reply("I moved")
        history.add_turn("thanks")
        history.add_reply("sure")
        history.add_turn("again")
        history.add_reply("yes")
        history.add_turn("bye")

        history.request(max_turns=2)

        assert [m["role"] for m in history.request()] == [
            "user",
            "assistant",
            "user",
        ]

    def test_the_tool_descriptions_count_against_the_budget(self):
        history = conversation(3)
        history.add_turn("question 3")
        budget = history.tokens() + 10

        history.request(budget=budget)
        assert len(history.turns) == 4
        history.request(budget=budget, extra_tokens=100)
        assert len(history.turns) < 4


class TestSummaries:
    def test_dropped_turns_become_a_summary_after_the_system_prompt(self):
        history = conversation(4, system="sys")
        history.add_turn("question 4")
        seen = []

        def summarizer(text):
            seen.append(text)
            return "  They talked about questions 0 and 1.  "

        messages = history.request(max_turns=2, summarizer=summarizer)

        assert messages[:3] == [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": "Summary of the conversation so far:\nThey talked about "
                "questions 0 and 1.",
            },
            {"role": "assistant", "content": "Understood."},
        ]
        assert [m["content"] for m in messages[3:] if m["role"] == "user"] == [
            "question 3",
            "question 4",
        ]
        assert seen == [
            "Conversation:\nuser: question 0\nassistant: answer 0\n"
            "user: question 1\nassistant: answer 1\n"
            "user: question 2\nassistant: answer 2"
        ]

    def test_the_summary_so_far_is_folded_into_the_next(self):
        history = conversation(3)
        history.add_turn("question 3")
        history.request(max_turns=1, summarizer=lambda text: "first summary")
        for index in range(3, 6):
            history.add_reply(f"answer {index}")
            history.add_turn(f"question {index + 1}")
        seen = []

        history.request(max_turns=1, summarizer=lambda t: seen.append(t) or "second")

        assert seen[0].startswith("Summary so far:\nfirst summary\n\nConversation:")
        assert history.summary == "second"

    def test_tool_calls_are_named_in_the_text_to_summarize(self):
        history = ChatHistory()
        history.add_turn("go")
        history.add_reply("", tool_calls=[{"id": "c1", "function": {"name": "move"}}])
        history.add_tool_result("c1", "moved")
        history.add_reply("done")
        history.add_turn("next")
        seen = []

        history.request(max_turns=1, summarizer=lambda t: seen.append(t) or "s")

        assert "assistant: [called move]\ntool: moved\nassistant: done" in seen[0]

    def test_a_summarizer_that_gives_nothing_lets_the_turns_go(self):
        history = conversation(4)
        history.add_turn("question 4")

        messages = history.request(max_turns=2, summarizer=lambda text: None)

        assert history.summary is None
        assert messages[0] == {"role": "user", "content": "question 3"}

    def test_without_a_summarizer_nothing_is_summarized(self):
        history = conversation(4)
        history.add_turn("question 4")
        history.request(max_turns=2)
        assert history.summary is None

    def test_the_summary_counts_against_the_budget_and_reset_clears_it(self):
        history = conversation(4)
        history.add_turn("question 4")
        before = history.tokens()
        history.request(max_turns=2, summarizer=lambda text: "x" * 400)

        assert history.tokens() > before - 3 * estimate_tokens("question 0answer 0")
        history.reset()
        assert history.summary is None and history.request() == []


class TestImages:
    def test_only_the_recent_turns_keep_their_images(self):
        history = ChatHistory()
        history.add_turn("first", images=[image()], image_sources=["cam"])
        history.add_reply("a cup")
        history.add_turn("second", images=[image(), image()], image_sources=["cam"])
        history.add_reply("two cups")
        history.add_turn("third", images=[image()], image_sources=["cam"])

        messages = history.request(image_turns=2)

        # the current turn's images are sent by the component, not on the message
        assert "images" not in messages[-1]
        assert len(messages[2]["images"]) == 2
        assert messages[0] == {
            "role": "user",
            "content": "first\n[1 image from 'cam' omitted]",
        }

    def test_by_default_only_the_current_turn_has_images(self):
        history = ChatHistory()
        history.add_turn("first", images=[image(), image()], image_sources=["a", "b"])
        history.add_reply("ok")
        history.add_turn("second", images=[image()])

        messages = history.request()

        assert messages[0]["content"] == "first\n[2 images from 'a', 'b' omitted]"
        assert "images" not in messages[0]

    def test_the_arrays_of_a_turn_out_of_the_window_are_released(self):
        history = ChatHistory()
        history.add_turn("first", images=[image()], image_sources=["cam"])
        history.add_reply("ok")
        history.add_turn("second", images=[image()])
        history.add_reply("ok")
        history.add_turn("third", images=[image()])

        history.request(image_turns=2)

        assert history.turns[0].images == []
        assert history.turns[1].images != []
        assert history.turns[0].image_count == 1
        # the note still says what was there
        assert history.request(image_turns=2)[0]["content"] == (
            "first\n[1 image from 'cam' omitted]"
        )

    def test_images_count_against_the_budget(self):
        history = ChatHistory()
        history.add_turn("first", images=[image()])
        history.add_reply("ok")
        history.add_turn("second", images=[image()])

        text_only = history.tokens(image_turns=2, image_tokens=0)
        assert history.tokens(image_turns=2, image_tokens=100) == text_only + 200
        # an omitted image costs its note, not its tokens
        assert history.tokens(image_turns=1, image_tokens=100) < text_only + 200


class TestAsText:
    def test_the_whole_conversation_after_the_summary(self):
        history = conversation(2)
        history.summary = "earlier"
        assert history.as_text() == (
            "Summary so far:\nearlier\n\nConversation:\n"
            "user: question 0\nassistant: answer 0\n"
            "user: question 1\nassistant: answer 1"
        )


class TestEstimate:
    @pytest.mark.parametrize(
        "text, tokens",
        [("", 0), ("abcd", 2), ("a" * 400, 101)],
        ids=["empty", "short", "long"],
    )
    def test_text_is_a_token_per_four_characters(self, text, tokens):
        assert estimate_tokens(text) == tokens

    def test_messages_count_their_text_tool_calls_and_images(self):
        messages = [{"role": "user", "content": "a" * 40, "images": [image()]}]
        assert estimate_tokens(messages, image_tokens=50) == 4 + 11 + 50
        with_call = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "x": "y"}]}
        ]
        assert estimate_tokens(with_call) > estimate_tokens([
            {"role": "assistant", "content": ""}
        ])

    def test_openai_content_parts_are_counted(self):
        parts = [
            {"type": "text", "text": "a" * 40},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]
        assert estimate_tokens([{"role": "user", "content": parts}], 50) == 4 + 11 + 50


def config(**fields):
    defaults = {
        "chat_history": True,
        "history_budget": None,
        "history_size": 10,
        "max_new_tokens": 512,
    }
    return SimpleNamespace(**{**defaults, **fields})


def model(context_length):
    return SimpleNamespace(context_length=context_length)


class TestResolveHistoryBudget:
    def test_the_config_budget_wins(self):
        assert (
            resolve_history_budget(
                config(history_budget=3000), model(8192), None, [], MagicMock()
            )
            == 3000
        )

    def test_derived_from_the_model_context_minus_the_generation(self):
        logger = MagicMock()
        assert (
            resolve_history_budget(config(), model(8192), None, [], logger)
            == 8192 - 512
        )
        assert "8192" in logger.info.call_args.args[0]

    def test_no_budget_and_a_warning_when_the_model_does_not_report_one(self):
        logger = MagicMock()
        assert resolve_history_budget(config(), model(None), None, [], logger) is None
        warning = logger.warning.call_args.args[0]
        assert "history_size (10 turns)" in warning and "history_budget" in warning

    def test_no_model_is_no_budget(self):
        assert resolve_history_budget(config(), None, None, [], MagicMock()) is None

    def test_a_context_too_small_for_the_prompt_and_tools_is_warned_about(self):
        logger = MagicMock()
        budget = resolve_history_budget(config(), model(2048), "x" * 8000, [], logger)
        assert budget == 2048 - 512
        assert "will not fit" in logger.warning.call_args.args[0]

    def test_without_history_nothing_is_resolved_or_said(self):
        logger = MagicMock()
        assert (
            resolve_history_budget(
                config(chat_history=False), model(None), None, [], logger
            )
            is None
        )
        logger.warning.assert_not_called()


class TestMatchesPhrase:
    @pytest.mark.parametrize("text", ["chat reset", "Chat reset.", " CHAT RESET! "])
    def test_case_and_trailing_punctuation_do_not_matter(self, text):
        assert matches_phrase(text, "chat reset")

    @pytest.mark.parametrize("text", [None, "", "chat reset please", "reset"])
    def test_anything_else_is_not_the_phrase(self, text):
        assert not matches_phrase(text, "chat reset")

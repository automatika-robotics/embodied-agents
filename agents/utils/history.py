"""The chat history of a model component.

A conversation is kept as turns. The request for a call is built from the turns.
The current turn goes in full. Older turns go in as the utterance and the
replies, and their images only for the most recent few.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from attrs import define, field


def estimate_tokens(
    text_or_messages: Union[str, Sequence[Dict[str, Any]], None],
    image_tokens: int = 0,
) -> int:
    """Rough token count of a text or of a message list.

    Text counts a token per four characters, tool calls as their JSON, and images
    with a fixed cost.

    :param text_or_messages: A text or a list of chat messages
    :type text_or_messages: Union[str, Sequence[Dict[str, Any]], None]
    :param image_tokens: Tokens counted for each image of a message
    :type image_tokens: int
    :return: The estimate
    :rtype: int
    """
    if not text_or_messages:
        return 0
    if isinstance(text_or_messages, str):
        return len(text_or_messages) // 4 + 1
    total = 0
    for message in text_or_messages:
        total += 4  # role and separators
        content = message.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):  # OpenAI content parts
            for part in content:
                if part.get("type") == "text":
                    total += estimate_tokens(part.get("text", ""))
                else:
                    total += image_tokens
        if tool_calls := message.get("tool_calls"):
            total += estimate_tokens(json.dumps(tool_calls))
        total += len(message.get("images", ())) * image_tokens
    return total


@define
class Turn:
    """One exchange of a conversation.

    :param utterance: What the user said, kept for the turns that came before
        the current one
    :param content: What is sent for the turn's own call: the utterance after
        the prompt templates and the retrieved documents dressed it up
    :param images: The images of the call, released once they are no longer sent
    :param image_sources: Where each image came from, for the note left in their absence
    :param replies: The assistant and tool messages that answered the turn
    """

    utterance: str
    content: str
    images: List[np.ndarray] = field(factory=list)
    image_sources: List[str] = field(factory=list)
    replies: List[Dict[str, Any]] = field(factory=list)
    # how many images the call had, kept after the arrays are released
    image_count: int = field(init=False)

    def __attrs_post_init__(self):
        """Count the images while the arrays are still here"""
        self.image_count = len(self.images)

    @property
    def answered(self) -> bool:
        """Whether the assistant replied, with text or with tool calls"""
        return any(reply["role"] == "assistant" for reply in self.replies)

    def release_images(self) -> None:
        """Let the arrays go, keeping what the omitted images note needs"""
        self.images = []

    def omitted_images_note(self) -> str:
        """The note left on the turn once its images are no longer sent"""
        count = self.image_count
        plural = "s" if count != 1 else ""
        sources = sorted(set(self.image_sources))
        origin = f" from {', '.join(repr(s) for s in sources)}" if sources else ""
        return f"[{count} image{plural}{origin} omitted]"


class ChatHistory:
    """The conversation a model component keeps, and the requests built from it.

    Trimming is persistent and in whole turns, with hysteresis so that the
    request keeps the same prefix over several calls, which is what lets a
    server reuse its prompt cache.

    The budget is a hard limit and when the request passes it, the oldest turns
    are dropped until it is back at ``TRIM_TO`` of the budget.

    :param system_prompt: The system prompt, always the first message when set
    :type system_prompt: Optional[str]
    """

    # The share of the budget the history is trimmed back to, and the share
    # of the turn cap the history may grow past it by before it is trimmed
    TRIM_TO = 0.75
    # How the summary of dropped turns is given to the model, as a user and
    # assistant pair after the system prompt
    SUMMARY_LEAD = "Summary of the conversation so far:\n"
    SUMMARY_ACK = "Understood."

    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt
        self.turns: List[Turn] = []
        self.summary: Optional[str] = None

    def reset(self) -> None:
        """Forget the conversation, keeping the system prompt"""
        self.turns = []
        self.summary = None

    @property
    def current(self) -> Turn:
        """The turn being answered"""
        if not self.turns:
            raise RuntimeError("The chat history has no turn yet")
        return self.turns[-1]

    def add_turn(
        self,
        utterance: str,
        content: Optional[str] = None,
        images: Sequence[np.ndarray] = (),
        image_sources: Sequence[str] = (),
    ) -> Turn:
        """Start a turn. A previous turn that never got a reply is dropped
        first.

        :param utterance: What the user said
        :param content: What to send for this call in its place, defaults to
            the utterance
        :param images: The images of the call
        :param image_sources: Where the images came from, e.g. topic names
        :return: The new turn
        """
        if self.turns and not self.turns[-1].answered:
            self.turns.pop()
        turn = Turn(
            utterance=utterance,
            content=utterance if content is None else content,
            images=list(images),
            image_sources=list(image_sources),
        )
        self.turns.append(turn)
        return turn

    def add_reply(
        self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Record an assistant message answering the current turn. A reply
        with no turn to answer, as when a stream ends after a reset, is let go"""
        if not self.turns:
            return
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.current.replies.append(message)

    def add_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """Attach tool calls to the last assistant message of the current turn"""
        for reply in reversed(self.current.replies):
            if reply["role"] == "assistant":
                reply["tool_calls"] = tool_calls
                return
        raise RuntimeError("No assistant message to attach the tool calls to")

    def add_tool_result(self, call_id: str, content: Any) -> None:
        """Record the result of a tool call of the current turn"""
        self.current.replies.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        })

    def request(
        self,
        *,
        budget: Optional[int] = None,
        max_turns: Optional[int] = None,
        image_turns: int = 1,
        image_tokens: int = 0,
        extra_tokens: int = 0,
        summarizer: Optional[Callable[[str], Optional[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """The messages of the next call, after trimming the history to the
        cap and the budget. With a summarizer, the dropped turns are folded
        into a summary instead of being forgotten.

        The current turn's images are not put on its message as the component
        sends them with the request. Images of the older turns within ``image_turns``
        ride on their messages under ``images``, as arrays, for the clients to encode.

        :param budget: Tokens the whole request may take, None for no budget
        :param max_turns: Turns kept after trimming, None for no cap. The
            request holds up to a third more before the oldest are dropped
        :param image_turns: Turns, the current one included, whose images are
            sent
        :param image_tokens: Tokens counted for each image
        :param extra_tokens: Tokens of the request outside the messages, such
            as the tool descriptions
        :param summarizer: Given the dropped turns as text, with the summary so
            far, returns the new summary, or None to let the turns go
        :return: The messages
        """
        dropped: List[Turn] = []
        if max_turns is not None and len(self.turns) > max_turns / self.TRIM_TO:
            dropped, self.turns = self.turns[:-max_turns], self.turns[-max_turns:]
        if (
            budget is not None
            and self.tokens(image_turns, image_tokens, extra_tokens) > budget
        ):
            target = budget * self.TRIM_TO
            while len(self.turns) > 1 and (
                self.tokens(image_turns, image_tokens, extra_tokens) > target
            ):
                dropped.append(self.turns.pop(0))
        if dropped and summarizer is not None:
            if summary := summarizer(self.as_text(dropped)):
                self.summary = summary.strip()
        # drop image arrays of turns older than image turns
        for turn in self.turns[: max(0, len(self.turns) - image_turns)]:
            turn.release_images()
        return self._messages(image_turns, with_images=True)

    def tokens(
        self, image_turns: int = 1, image_tokens: int = 0, extra_tokens: int = 0
    ) -> int:
        """Estimated size of the request the history would build now, the
        current turn's images included"""
        messages = self._messages(image_turns, with_images=True)
        total = estimate_tokens(messages, image_tokens) + extra_tokens
        if self.turns:
            total += len(self.turns[-1].images) * image_tokens
        return total

    def as_text(self, turns: Optional[List[Turn]] = None) -> str:
        """The turns, all of them by default, as plain text after the summary
        so far. For the summarizer and for reading the conversation

        :param turns: The turns to render, the whole conversation when None
        :type turns: Optional[List[Turn]]
        :rtype: str
        """
        lines = []
        if self.summary:
            lines += ["Summary so far:", self.summary, ""]
        lines.append("Conversation:")
        for turn in self.turns if turns is None else turns:
            lines.append(f"user: {turn.utterance}")
            for reply in turn.replies:
                text = str(reply.get("content") or "")
                if calls := reply.get("tool_calls"):
                    names = ", ".join(c["function"]["name"] for c in calls)
                    text = f"{text} [called {names}]".strip()
                lines.append(f"{reply['role']}: {text}")
        return "\n".join(lines)

    def _messages(self, image_turns: int, with_images: bool) -> List[Dict[str, Any]]:
        """Flatten the turns into chat messages"""
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.summary:
            messages.append({
                "role": "user",
                "content": self.SUMMARY_LEAD + self.summary,
            })
            messages.append({"role": "assistant", "content": self.SUMMARY_ACK})
        last = len(self.turns) - 1
        for index, turn in enumerate(self.turns):
            if index == last:
                user: Dict[str, Any] = {"role": "user", "content": turn.content}
            else:
                user = {"role": "user", "content": turn.utterance}
                if turn.image_count:
                    if last - index < image_turns:
                        if with_images:
                            user["images"] = list(turn.images)
                    else:
                        user["content"] = (
                            f"{turn.utterance}\n{turn.omitted_images_note()}"
                        )
            messages.append(user)
            messages.extend(turn.replies)
        return messages


def matches_phrase(text: Optional[str], phrase: str) -> bool:
    """Whether a text is the phrase, case and trailing punctuation aside,
    which a transcription adds

    :param text: What was said
    :type text: Optional[str]
    :param phrase: The phrase, e.g. the one that resets the chat history
    :type phrase: str
    :rtype: bool
    """
    if not text:
        return False
    normalize = lambda t: t.strip().lower().rstrip(".!?,;:")  # noqa: E731
    return normalize(text) == normalize(phrase)


def resolve_history_budget(
    config, model, system_prompt: Optional[str], tools: Sequence[Dict], logger
) -> Optional[int]:
    """The token budget of a component's requests: the config's when set, else
    the model's context length minus the tokens it may generate, else None,
    the turn count alone then limiting the history. Says what it decided on
    the logger when chat history is on.

    :param config: The component config, with chat_history, history_budget,
        history_size and max_new_tokens
    :param model: The model client or the local model, with a context_length
    :param system_prompt: The system prompt the requests carry
    :param tools: The tool descriptions the requests carry
    :param logger: The component's logger
    :rtype: Optional[int]
    """
    if not config.chat_history or config.history_budget:
        return config.history_budget
    context = getattr(model, "context_length", None) if model else None
    if context is None:
        logger.warning(
            "The model does not report a context length and history_budget is "
            f"not set: the chat history is limited by history_size "
            f"({config.history_size} turns) alone. Set history_budget to protect "
            "the model's context window."
        )
        return None
    budget = context - config.max_new_tokens
    fixed = estimate_tokens(system_prompt or "") + (
        estimate_tokens(json.dumps(list(tools))) if tools else 0
    )
    if budget <= fixed:
        logger.warning(
            f"The model's context of {context} tokens leaves {budget} for the "
            f"request after max_new_tokens, and the system prompt and tool "
            f"descriptions take about {fixed}. The chat history will not fit."
        )
    else:
        logger.info(
            f"Chat history budget: {budget} tokens, the model's context of "
            f"{context} minus max_new_tokens."
        )
    return budget

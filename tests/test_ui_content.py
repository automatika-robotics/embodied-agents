"""Tests for the content the UI serves for agents types."""

import base64
import json

import cv2
import numpy as np
import pytest

from agents.ros import Topic


def test_video_ui_content_is_the_last_frame_as_jpeg():
    """A client shows a video like an image, so it gets the last frame"""
    from agents.callbacks import VideoCallback
    from agents.ros import Video

    red = np.zeros((8, 8, 3), dtype=np.uint8)
    red[..., 0] = 255
    blue = np.zeros((8, 8, 3), dtype=np.uint8)
    blue[..., 2] = 255
    callback = VideoCallback(Topic(name="video", msg_type="Video"))
    callback.msg = Video.convert([red, blue])

    content = callback._get_ui_content()

    json.dumps(content)  # served as JSON by the UI API
    jpeg = np.frombuffer(base64.b64decode(content), dtype=np.uint8)
    frame = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)  # BGR
    assert frame[4, 4, 0] > 200 and frame[4, 4, 2] < 50


def _chunk(text, done):
    from agents.ros import StreamingString

    return StreamingString.convert(text, stream=True, done=done)


def test_streaming_string_ui_content_says_when_the_stream_is_done():
    """A client can only tell a finished answer from a pause with the flag"""
    from agents.callbacks import StreamingStringCallback

    callback = StreamingStringCallback(Topic(name="answer", msg_type="StreamingString"))

    callback.callback(_chunk("Hello", done=False))
    assert callback._get_ui_content() == {"data": "Hello", "done": False}
    callback.callback(_chunk(" world", done=False))
    callback.callback(_chunk("", done=True))
    assert callback._get_ui_content() == {"data": "Hello world", "done": True}
    callback.callback(_chunk("Bye", done=False))
    assert callback._get_ui_content() == {"data": "Bye", "done": False}


def test_log_starts_a_new_entry_for_each_stream():
    """Streams that follow each other stay separate entries in the log"""
    pytest.importorskip("fasthtml")
    pytest.importorskip("monsterui")
    from fasthtml.common import to_xml
    from ros_sugar.ui_node.elements import initial_logging_card

    from agents.ui_elements import _log_streaming_string_element

    card = initial_logging_card()
    for content in (
        {"data": "Hel", "done": False},
        {"data": "Hello", "done": True},
        {"data": "", "done": True},  # an empty stream
        {"data": "Bye", "done": True},
    ):
        card = _log_streaming_string_element(card, content, data_src="robot")

    log = to_xml(card)
    assert log.count('id="inner-text"') == 2
    assert "</strong>Hello</span>" in log and "</strong>Bye</span>" in log

from typing import Dict

from .ros import (
    Detections,
    Detections3D,
    DetectionsMultiSource,
    PointsOfInterest,
    RGBD,
    StreamingString,
    Video,
)
from ros_sugar.ui_node.elements import (
    _out_image_element,
    _log_text_element,
    replace_text_in_logging_card,
)


def _log_streaming_string_element(logging_card, output: Dict, data_src: str):
    """Render StreamingString output in the logging card.

    ``output["data"]`` is the full text of the current stream, so the open
    entry's text is replaced on each update rather than appended. A done stream
    closes its entry, so the next stream starts a new one.
    """
    if getattr(logging_card.children[-1], "id", None) == "streaming-text":
        replace_text_in_logging_card(
            logging_card, output["data"], target_id="streaming-text"
        )
    elif output["data"]:
        _log_text_element(logging_card, output["data"], data_src, id="streaming-text")
    else:
        return logging_card  # an empty stream has nothing to show
    if output["done"]:
        logging_card.children[-1].id = "text"
    return logging_card


OUTPUT_ELEMENTS = {
    StreamingString: _log_streaming_string_element,
    Detections: _out_image_element,
    DetectionsMultiSource: _out_image_element,
    PointsOfInterest: _out_image_element,
    RGBD: _out_image_element,
    Video: _out_image_element,
    # 3D detections dont have an image; displayed as text
    Detections3D: _log_text_element,
}

INPUT_ELEMENTS = {}

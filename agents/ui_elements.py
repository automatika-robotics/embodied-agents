from .ros import (
    Detections,
    Detections3D,
    DetectionsMultiSource,
    PointsOfInterest,
    RGBD,
    StreamingString,
)
from ros_sugar.ui_node.elements import (
    _out_image_element,
    _log_text_element,
    replace_text_in_logging_card,
)


def _log_streaming_string_element(logging_card, output: str, data_src: str):
    """Render StreamingString output in the logging card.

    ``output`` is the full text of the current stream, so the entry text is
    replaced on each update rather than appended.
    """
    if getattr(logging_card.children[-1], "id", None) == "streaming-text":
        return replace_text_in_logging_card(
            logging_card, output, target_id="streaming-text"
        )
    else:
        return _log_text_element(logging_card, output, data_src, id="streaming-text")


OUTPUT_ELEMENTS = {
    StreamingString: _log_streaming_string_element,
    Detections: _out_image_element,
    DetectionsMultiSource: _out_image_element,
    PointsOfInterest: _out_image_element,
    RGBD: _out_image_element,
    # 3D detections dont have an image; displayed as text
    Detections3D: _log_text_element,
}

INPUT_ELEMENTS = {}

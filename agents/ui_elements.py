from .ros import Detections, DetectionsMultiSource, PointsOfInterest
from ros_sugar.ui_node.elements import _out_image_element

OUTPUT_ELEMENTS = {
    Detections: _out_image_element,
    DetectionsMultiSource: _out_image_element,
    PointsOfInterest: _out_image_element,
}

INPUT_ELEMENTS = {}

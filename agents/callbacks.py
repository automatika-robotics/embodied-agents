from typing import Optional
import os
import cv2
import numpy as np
from ros_sugar.io import (
    GenericCallback,
    TextCallback,
    get_logger,
)

from ros_sugar.io.utils import image_pre_processing, read_compressed_image

from .utils import create_detection_context

__all__ = ["GenericCallback", "TextCallback"]


class VideoCallback(GenericCallback):
    """
    Video Callback class. Its get method saves a video as an array of arrays
    """

    def __init__(self, input_topic, node_name: Optional[str] = None) -> None:
        """
        Constructs a new instance.
        :param      input_topic:  Subscription topic
        :type       input_topic:  Input
        """
        super().__init__(input_topic, node_name)
        # fixed video needs to be a path to cv2 readable video
        if hasattr(input_topic, "fixed"):
            if os.path.isfile(input_topic.fixed):
                try:
                    # read all video frames
                    video = []
                    cap = cv2.VideoCapture(input_topic.fixed)
                    if not cap.isOpened():
                        raise TypeError()
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            video.append(frame)
                        else:
                            break
                    # Convert frame list to ndarray
                    self.msg = np.array(video)
                except Exception:
                    get_logger(self.node_name).error(
                        f"Fixed path {self.msg} provided for Vidoe topic is not readable Video file"
                    )
            else:
                get_logger(self.node_name).error(
                    f"Fixed path {self.msg} provided for Video topic is not a valid file path"
                )

    def _get_output(self, **_) -> Optional[np.ndarray]:
        """
        Gets video as a numpy array.
        :returns:   Video as nd_array
        :rtype:     np.ndarray
        """
        if not self.msg:
            return None

        # return np.ndarray if fixed video has been read
        if isinstance(self.msg, np.ndarray):
            return self.msg
        else:
            # pre-process in case of weird encodings and reshape ROS topic
            video = []
            for img in self.msg.frames:
                video.append(image_pre_processing(img))
            for img in self.msg.compressed_frames:
                video.append(read_compressed_image(img))
            return np.array(video)


class RGBDCallback(GenericCallback):
    """
    RGBD Callback class. Its get method returns numpy array of the RGB part
    """

    def __init__(self, input_topic, node_name: Optional[str] = None) -> None:
        """
        Constructs a new instance.
        :param      input_topic:  Subscription topic
        :type       input_topic:  Input
        """
        super().__init__(input_topic, node_name)
        self.msg = None
        # fixed RGBD message cannot be read from a file
        if hasattr(input_topic, "fixed"):
            get_logger(self.node_name).error(
                "RGBD message cannot be read from a fixed file"
            )

    def _get_output(self, get_depth=False, **_) -> Optional[np.ndarray]:
        """
        Gets RGBD image as a numpy array.
        :returns:   Image and Depth as nd_array
        :rtype:     np.ndarray
        """
        if not self.msg:
            return None

        else:
            # pre-process and reshape the RGB image
            rgb = image_pre_processing(self.msg.rgb)
            if get_depth:
                depth = image_pre_processing(self.msg.depth)
                # Ensure depth has shape (H, W, 1)
                depth_expanded = np.expand_dims(depth, axis=-1)
                # Concatenate along the channel axis and return rgbd
                return np.concatenate((rgb, depth_expanded), axis=-1)
            else:
                return rgb


class ObjectDetectionCallback(GenericCallback):
    """
    Object detection Callback class.
    Its get method returns the bounding box data
    """

    def __init__(self, input_topic, node_name: Optional[str] = None) -> None:
        """
        Constructs a new instance.

        :param      input_topic:  Subscription topic
        :type       input_topic:  str
        """
        super().__init__(input_topic, node_name)
        self.msg = input_topic.fixed if hasattr(input_topic, "fixed") else None

    def _get_output(self, **_) -> Optional[str]:
        """
        Processes labels and returns a context string for
        prompt engineering

        :returns:   Comma separated classnames
        :rtype:     str
        """
        if not self.msg:
            return None
        # send fixed list of labels if it exists
        if isinstance(self.msg, list):
            return create_detection_context(self.msg)
        # send labels from ROS message
        else:
            label_list = [
                label for detection in self.msg.detections for label in detection.labels
            ]
            detections_string = create_detection_context(label_list)
            return detections_string

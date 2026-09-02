from typing import Any, Optional
import os
import cv2
import numpy as np
from ros_sugar.io import (
    GenericCallback,
    TextCallback,
    get_logger,
)

from ros_sugar.io.utils import (
    image_pre_processing,
    process_encoding,
    read_compressed_image,
    parse_format,
    convert_img_to_jpeg_str,
)

from .utils import (
    draw_detection_bounding_boxes,
    draw_points_2d,
)

from .utils.actions import JointsData

__all__ = ["GenericCallback", "TextCallback"]


class StreamingStringCallback(TextCallback):
    def __init__(self, input_topic, node_name: str = "") -> None:
        super().__init__(input_topic, node_name)
        # Full text of the current stream, accumulated across chunks
        self._stream_text: str = ""

    def callback(self, msg) -> None:
        # self.msg is still the previous message here unless the stream is new or
        # completed
        if getattr(self.msg, "done", True):
            self._stream_text = ""
        self._stream_text += msg.data
        super().callback(msg)

    def _get_ui_content(self, **_) -> str:
        """Full text of the current stream, so latest-value readers never miss
        coalesced chunks."""
        return self._stream_text

    def _get_output(self, **_) -> Optional[str]:
        """Gets text.
        :rtype: str | None
        """

        if self.msg is None:
            return None

        # return str if fixed str has been read
        if isinstance(self.msg, str):
            return self.msg
        # return ROS message data
        else:
            if self._template:
                get_logger(self.node_name).warning(
                    "StreamingString topics cannot render templated strings. Discarding template."
                )
            return self.msg.data


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
        if self._is_fixed:
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
        if self.msg is None:
            return None

        # return np.ndarray if fixed video has been read
        if isinstance(self.msg, np.ndarray):
            return self.msg

        # pre-process in case of weird encodings and reshape ROS topic
        video = []
        for img in self.msg.frames:
            if not getattr(self, "image_encoding", None):
                self.image_encoding = process_encoding(img.encoding)
            video.append(image_pre_processing(img, *self.image_encoding))
        for img in self.msg.compressed_frames:
            if not getattr(self, "compressed_encoding", None):
                self.compressed_encoding = parse_format(img.format)
            video.append(read_compressed_image(img, self.compressed_encoding))
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
        if self._is_fixed:
            get_logger(self.node_name).error(
                "RGBD message cannot be read from a fixed file"
            )

    def _get_output(
        self, get_depth=False, depth_only=False, **_
    ) -> Optional[np.ndarray]:
        """
        Gets RGBD image as a numpy array.
        Returns the RGB part by default. With `get_depth`, returns RGB and
        depth concatenated as (H, W, 4). With `depth_only`, returns just the
        depth part as (H, W, 1).
        :returns:   Image and/or Depth as nd_array
        :rtype:     np.ndarray
        """
        if self.msg is None or not self.msg.rgb:
            return None

        if depth_only:
            if not self.msg.depth:
                return None
            if not getattr(self, "depth_encoding", None):
                self.depth_encoding = process_encoding(self.msg.depth.encoding)
            depth = image_pre_processing(self.msg.depth, *self.depth_encoding)
            # Ensure depth has shape (H, W, 1)
            return np.expand_dims(depth, axis=-1)

        # pre-process and reshape the RGB image
        if not getattr(self, "rgb_encoding", None):
            self.rgb_encoding = process_encoding(self.msg.rgb.encoding)
        rgb = image_pre_processing(self.msg.rgb, *self.rgb_encoding)
        if get_depth:
            if not getattr(self, "depth_encoding", None):
                self.depth_encoding = process_encoding(self.msg.depth.encoding)
            depth = image_pre_processing(self.msg.depth, *self.depth_encoding)
            # Ensure depth has shape (H, W, 1)
            depth_expanded = np.expand_dims(depth, axis=-1)
            # Concatenate along the channel axis and return rgbd
            return np.concatenate((rgb, depth_expanded), axis=-1)
        else:
            return rgb

    def _get_ui_content(self, **_) -> str:
        """Get ui content for image"""
        output = self.get_output()
        return convert_img_to_jpeg_str(output, self.node_name)


class DetectionsMultiSourceCallback(GenericCallback):
    """
    Object detection Callback class for Detections2DMultiSource msg
    Its get method returns the bounding box data
    """

    def __init__(self, input_topic, node_name: Optional[str] = None) -> None:
        """
        Constructs a new instance.

        :param      input_topic:  Subscription topic
        :type       input_topic:  str
        """
        super().__init__(input_topic, node_name)
        self.msg = input_topic.fixed if self._is_fixed else None
        self.encoding = None

    def _get_output(self, **_) -> Optional[str]:
        """
        Processes labels and returns a context string for
        prompt engineering

        :returns:   Comma separated classnames
        :rtype:     str
        """
        if self.msg is None:
            return None
        labels = (
            self.msg
            if isinstance(self.msg, list)  # a fixed input is already a list of labels
            else [
                label
                for detection in self.msg.detections
                for label in detection.labels
            ]
        )
        return ", ".join(labels) if labels else None

    def _get_ui_content(self, **_) -> str:
        """Get UI content for the first Detections2D msg in Detections2DMultiSource: draw bounding boxes and labels on the image."""
        if self.msg is None:
            return ""

        # a fixed input is already a list of labels
        if isinstance(self.msg, list):
            return ", ".join(self.msg)

        detections = self.msg.detections
        if not detections:
            return ""

        img = None

        # Decode image or compressed image
        # NOTE: Only checks first detections source
        if self.msg.detections[0].compressed_image.data:
            compressed = self.msg.compressed_image
            if not getattr(self, "encoding", None):
                self.encoding = parse_format(compressed.format)
            img = read_compressed_image(compressed, self.encoding)

        elif self.msg.detections[0].image.data:
            image = self.msg.image
            if not getattr(self, "encoding", None):
                self.encoding = process_encoding(image.encoding)
            img = image_pre_processing(image, *self.encoding)

        # Ensure image exists
        if img is None:
            # Create blank white canvas if no image is available
            img = np.ones((480, 640, 3), dtype=np.uint8) * 255

        # Extract bounding boxes and labels
        bounding_boxes = getattr(detections[0], "boxes", [])
        labels = getattr(detections[0], "labels", [])

        img = draw_detection_bounding_boxes(img, bounding_boxes, labels)

        return convert_img_to_jpeg_str(img, getattr(self, "node_name", "ui"))


class DetectionsCallback(GenericCallback):
    """
    Object detection Callback class for Detections2D msg
    Its get method returns the bounding box data
    """

    def __init__(self, input_topic, node_name: Optional[str] = None) -> None:
        """
        Constructs a new instance.

        :param      input_topic:  Subscription topic
        :type       input_topic:  str
        """
        super().__init__(input_topic, node_name)
        self.msg = input_topic.fixed if self._is_fixed else None
        self.encoding = None

    def _get_output(self, **_) -> Optional[str]:
        """
        Processes labels and returns a context string for
        prompt engineering

        :returns:   Comma separated classnames
        :rtype:     str
        """
        if self.msg is None:
            return None
        # a fixed input is already a list of labels
        labels = self.msg if isinstance(self.msg, list) else list(self.msg.labels)
        return ", ".join(labels) if labels else None

    def _get_ui_content(self, **_) -> str:
        """Get UI content for Detections2D: draw bounding boxes and labels on the image."""
        if self.msg is None:
            return ""

        # a fixed input is already a list of labels
        if isinstance(self.msg, list):
            return ", ".join(self.msg)

        img = None

        # Decode image or compressed image
        if self.msg.compressed_image.data:
            compressed = self.msg.compressed_image
            if not getattr(self, "encoding", None):
                self.encoding = parse_format(compressed.format)
            img = read_compressed_image(compressed, self.encoding)

        elif self.msg.image.data:
            image = self.msg.image
            if not getattr(self, "encoding", None):
                self.encoding = process_encoding(image.encoding)
            img = image_pre_processing(image, *self.encoding)

        # Ensure image exists
        if img is None:
            # Create blank white canvas if no image is available
            img = np.ones((480, 640, 3), dtype=np.uint8) * 255

        # Extract bounding boxes and labels
        bounding_boxes = getattr(self.msg, "boxes", [])
        labels = getattr(self.msg, "labels", [])

        img = draw_detection_bounding_boxes(img, bounding_boxes, labels)

        return convert_img_to_jpeg_str(img, getattr(self, "node_name", "ui"))


class PointsOfInterestCallback(GenericCallback):
    """
    Callback class for PointsOfInterest msg
    Its get method returns the bounding box data
    """

    def __init__(self, input_topic, node_name: Optional[str] = None) -> None:
        """
        Constructs a new instance.

        :param      input_topic:  Subscription topic
        :type       input_topic:  str
        """
        super().__init__(input_topic, node_name)
        self.msg = input_topic.fixed if self._is_fixed else None
        self.encoding = None

    def _get_output(self, **_) -> Optional[np.ndarray]:
        """
        Processes labels and returns a context string for
        prompt engineering

        :returns:   Comma separated classnames
        :rtype:     str
        """
        if self.msg is None:
            return None

        # send fixed list of points if it exists
        if isinstance(self.msg, list):
            return np.array(self.msg)

        # send points from ROS message
        points = []
        for point in self.msg.points:
            points.append([point.x, point.y])
        return np.array(points)

    def _get_ui_content(self, **_) -> str:
        """Get UI content for PointsOfInterest: draw points on the image."""

        if self.msg is None:
            return ""

        points = self.get_output()

        img = None
        # Decode image or compressed image
        if self.msg.compressed_image.data:
            compressed = self.msg.compressed_image
            if not getattr(self, "encoding", None):
                self.encoding = parse_format(compressed.format)
            img = read_compressed_image(compressed, self.encoding)

        elif self.msg.image.data:
            image = self.msg.image
            if not getattr(self, "encoding", None):
                self.encoding = process_encoding(image.encoding)
            img = image_pre_processing(image, *self.encoding)

        # Ensure image exists
        if img is None:
            # Create blank white canvas if no image is available
            img = np.ones((480, 640, 3), dtype=np.uint8) * 255

        img = draw_points_2d(img, points)  # draw points as red circles

        return convert_img_to_jpeg_str(img, getattr(self, "node_name", "ui"))


class JointStateCallback(GenericCallback):
    """
    sensor_msgs/JointState Callback class.

    The state of each joint (revolute or prismatic) is defined by:
    * the position of the joint (rad or m),
    * the velocity of the joint (rad/s or m/s) and
    * the effort that is applied in the joint (Nm or N)
    """

    def _get_output(self, **_) -> Optional[JointsData]:
        """
        Gets joint states as a dictionary with 'joint_names' and 'positions' keys.
        :returns:   Joint states as dict
        :rtype:     dict
        """
        if self.msg is None:
            return None

        return JointsData(
            joints_names=self.msg.name,
            positions=np.array(self.msg.position),
            velocities=np.array(self.msg.velocity),
            efforts=np.array(self.msg.effort),
        )


class Detections3DCallback(GenericCallback):
    """
    Callback class for Detections3D msg

    Its get method returns a context string of the detected classes, like the
    2D detections callback, so the output can be used in prompts and stored as
    semantic memory. Consumers that need the geometry ask for the message itself
    with `get_msg=True`.
    """

    def __init__(self, input_topic, node_name: Optional[str] = None) -> None:
        """
        Constructs a new instance.

        :param      input_topic:  Subscription topic
        :type       input_topic:  str
        """
        super().__init__(input_topic, node_name)
        self.msg = input_topic.fixed if self._is_fixed else None

    def _get_output(self, get_msg: bool = False, **_) -> Optional[Any]:
        """
        Processes labels and returns a context string for prompt engineering,
        or the message itself when the metric boxes are needed.

        :param      get_msg:  Return the Detections3D message instead of a
                              context string
        :type       get_msg:  bool

        :returns:   Comma separated classnames, or the Detections3D message
        :rtype:     Optional[Union[str, Detections3D]]
        """
        if self.msg is None:
            return None

        if get_msg:
            return self.msg

        if not self.msg.labels:
            return None
        frame = self.msg.header.frame_id or "camera frame"
        objects = ", ".join(
            f"{label} at ({box.center.position.x:.2f}, {box.center.position.y:.2f}, "
            f"{box.center.position.z:.2f})"
            for label, box in zip(self.msg.labels, self.msg.boxes)
        )
        return f"In {frame}: {objects}"

    def _get_ui_content(self, **_) -> str:
        """Get UI content for Detections3D: what was found and how far away."""
        return self._get_output() or "No objects detected"

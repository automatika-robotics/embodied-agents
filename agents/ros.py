"""The following classes provide wrappers for data being transmitted via ROS topics. These classes form the inputs and outputs of [Components](agents.components.md)."""

from typing import Union, Any, Dict, List, Tuple
import numpy as np
from attrs import define, field, Factory
from importlib.util import find_spec

# FROM SUGARCOAT
from ros_sugar.supported_types import (
    add_additional_datatypes,
    SupportedType,
    Audio,
    Image,
    CompressedImage,
    OccupancyGrid,
    Odometry,
    String,
    ROSImage,
    ROSCompressedImage,
)
from ros_sugar.io import Topic as BaseTopic

from ros_sugar.config import (
    BaseComponentConfig,
    ComponentRunType,
    BaseAttrs,
    base_validators,
)
from ros_sugar.core import BaseComponent
from ros_sugar.core.component import MutuallyExclusiveCallbackGroup
from ros_sugar import Launcher
from ros_sugar.utils import component_action

# LEIBNIZ TYPES
from automatika_embodied_agents.msg import Point2D, Bbox2D, Detection2D, Detections2D
from automatika_embodied_agents.msg import (
    Video as ROSVideo,
    Tracking as ROSTracking,
    Trackings as ROSTrackings,
    PointsOfInterest as ROSPointsOfInterest,
)
from .callbacks import ObjectDetectionCallback, RGBDCallback, VideoCallback

__all__ = [
    "String",
    "Audio",
    "Image",
    "CompressedImage",
    "OccupancyGrid",
    "Odometry",
    "Topic",
    "FixedInput",
    "base_validators",
    "BaseAttrs",
    "BaseComponent",
    "BaseComponentConfig",
    "ComponentRunType",
    "Launcher",
    "component_action",
    "MapLayer",
    "Route",
    "MutuallyExclusiveCallbackGroup",
]


class Video(SupportedType):
    """Video."""

    _ros_type = ROSVideo
    callback = VideoCallback

    @classmethod
    def convert(
        cls,
        output: Union[List[ROSImage], List[ROSCompressedImage], List[np.ndarray]],
        **_,
    ) -> ROSVideo:
        """
        Takes an list of images and returns a video message (Image Array)
        :return: Video
        """
        msg = ROSVideo()
        frames = []
        compressed_frames = []
        for frame in output:
            if isinstance(frame, ROSCompressedImage):
                compressed_frames.append(CompressedImage.convert(frame))
            else:
                frames.append(Image.convert(frame))
        msg.frames = frames
        msg.compressed_frames = compressed_frames
        return msg


class Detection(SupportedType):
    """Detection."""

    _ros_type = Detection2D
    callback = None  # not defined

    @classmethod
    def convert(
        cls, output: Dict, img: Union[ROSImage, ROSCompressedImage, np.ndarray], **_
    ) -> Detection2D:
        """
        Takes object detection data and converts it into a ROS message
        of type Detection2D
        :return: Detection2D
        """
        msg = Detection2D()
        msg.scores = output["scores"]
        msg.labels = output["labels"]
        boxes = []
        for bbox in output["bboxes"]:
            box = Bbox2D()
            box.top_left_x = float(bbox[0])
            box.top_left_y = float(bbox[1])
            box.bottom_right_x = float(bbox[2])
            box.bottom_right_y = float(bbox[3])
            boxes.append(box)

        msg.boxes = boxes
        if isinstance(img, ROSCompressedImage):
            msg.compressed_image = CompressedImage.convert(img)
        # Handle RealSense RGBD msgs
        elif hasattr(img, "depth"):
            msg.image = Image.convert(img.rgb)
            msg.depth = Image.convert(img.depth)
        else:
            msg.image = Image.convert(img)
        return msg


class Detections(SupportedType):
    """Detections."""

    _ros_type = Detections2D
    callback = ObjectDetectionCallback

    @classmethod
    def convert(cls, output: List, images: List, **_) -> Detections2D:
        """
        Takes object detections data and converts it into a ROS message
        of type Detections2D
        :return: Detections2D
        """
        msg = Detections2D()
        detections = []
        for img, detection in zip(images, output):
            detections.append(Detection.convert(detection, img))
        msg.detections = detections
        return msg


class PointsOfInterest(SupportedType):
    """PointsOfInterest."""

    _ros_type = ROSPointsOfInterest
    callback = None  # not defined

    @classmethod
    def convert(
        cls,
        output: List[Tuple[int, int]],
        img: Union[ROSImage, ROSCompressedImage, np.ndarray],
        **_,
    ) -> ROSPointsOfInterest:
        """
        Takes points of interest on an image and converts it into a ROS message
        of type PointsOfInterest
        :return: PointsOfInterest
        """
        msg = ROSPointsOfInterest()
        points = []
        for p in output:
            point = Point2D()
            point.x = float(p[0])
            point.y = float(p[1])
            points.append(point)
        msg.points = points

        if isinstance(img, ROSCompressedImage):
            msg.compressed_image = CompressedImage.convert(img)
        # Handle RealSense RGBD msgs
        elif hasattr(img, "depth"):
            msg.image = Image.convert(img.rgb)
            msg.depth = Image.convert(img.depth)
        else:
            msg.image = Image.convert(img)
        return msg


class Tracking(SupportedType):
    """Tracking."""

    _ros_type = ROSTracking
    callback = None  # Not defined in EmbodiedAgents

    @classmethod
    def convert(
        cls, output: Dict, img: Union[ROSImage, ROSCompressedImage, np.ndarray], **_
    ) -> ROSTracking:
        """
        Takes tracking data and converts it into a ROS message
        of type Tracking
        :return: ROSTracking
        """
        msg = ROSTracking()
        msg.ids = output.get("ids") or []
        msg.labels = output.get("tracked_labels") or []

        estimated_velocities = []
        if o_estimated_velocities := output.get("estimated_velocities"):
            for obj_vels in o_estimated_velocities:
                for obj_instance_v in obj_vels:
                    estimated_velocity = Point2D()
                    estimated_velocity.x = obj_instance_v[0]
                    estimated_velocity.y = obj_instance_v[1]
                    estimated_velocities.append(estimated_velocity)

        tracked_boxes = []
        centroids = []
        if o_tracked_points := output.get("tracked_points"):
            for bbox in o_tracked_points:
                # Each 3 points represent one object (top-left, bottom-right, center)
                box = Bbox2D()
                box.top_left_x = bbox[0][0]
                box.top_left_y = bbox[0][1]
                box.bottom_right_x = bbox[1][0]
                box.bottom_right_y = bbox[1][1]
                tracked_boxes.append(box)
                centroid = Point2D()
                centroid.x = bbox[2][0]
                centroid.y = bbox[2][1]
                centroids.append(centroid)

        msg.boxes = tracked_boxes
        msg.centroids = centroids
        msg.estimated_velocities = estimated_velocities
        if isinstance(img, ROSCompressedImage):
            msg.compressed_image = CompressedImage.convert(img)
        # Handle RealSense RGBD msgs
        elif hasattr(img, "depth"):
            msg.image = Image.convert(img.rgb)
            msg.depth = Image.convert(img.depth)
        else:
            msg.image = Image.convert(img)
        return msg


class Trackings(SupportedType):
    """Trackings."""

    _ros_type = ROSTrackings
    callback = None  # Not defined

    @classmethod
    def convert(cls, output: List, images: List, **_) -> ROSTrackings:
        """
        Takes trackings data and converts it into a ROS message
        of type ROSTrackings
        :return: ROSTrackings
        """
        msg = ROSTrackings()
        trackings = []
        for img, tracking in zip(images, output):
            trackings.append(Tracking.convert(tracking, img))
        msg.trackings = trackings
        return msg


class RGBD(SupportedType):
    """Adds callback for automatika_embodied_agents/msg/Detections2D message"""

    callback = RGBDCallback

    @classmethod
    def get_ros_type(cls) -> type:
        if find_spec("realsense2_camera_msgs") is None:
            raise ModuleNotFoundError(
                "'realsense2_camera_msgs' module is required to use 'RGBD' msg type but it is not installed"
            )
        from realsense2_camera_msgs.msg import RGBD as RealSenseRGBD

        return RealSenseRGBD


agent_types = [
    Video,
    Detection,
    Detections,
    Tracking,
    Trackings,
    RGBD,
    PointsOfInterest,
]


add_additional_datatypes(agent_types)


@define(kw_only=True)
class Topic(BaseTopic):
    """
    A topic is an idomatic wrapper for a ROS2 topic, Topics can be given as inputs or outputs to components. When given as inputs, components automatically create listeners for the topics upon their activation. And when given as outputs, components create publishers for publishing to the topic.

    :param name: Name of the topic
    :type name: str
    :param msg_type: One of the SupportedTypes. This parameter can be set by passing the SupportedType data-type name as a string. See a list of supported types [here](https://automatika-robotics.github.io/sugarcoat/advanced/types.html)
    :type msg_type: Union[type[supported_types.SupportedType], str]
    :param qos_profile: QoS profile for the topic
    :type qos_profile: QoSConfig

    Example usage:
    ```python
    position = Topic(name="odom", msg_type="Odometry")
    map_meta_data = Topic(name="map_meta_data", msg_type="MapMetaData")
    ```
    """

    pass


@define(kw_only=True)
class FixedInput(Topic):
    """
    A FixedInput can be provided to components as input and is similar to a Topic except components do not create a subscriber to it and whenever they _read_ it, they always get the same data. The nature of the data depends on the _msg_type_ specified.

    :param name: Name of the topic
    :type name: str
    :param msg_type: One of the SupportedTypes. This parameter can be set by passing the SupportedType data-type name as a string
    :type msg_type: Union[type[supported_types.SupportedType], str]
    :param fixed: Fixed input string or path to a file. Various SupportedTypes implement FixedInput processing differently.
    :type fixed: str | Path

    Example usage:
    ```python
    text0 = FixedInput(
        name="text2",
        msg_type="String",
        fixed="What kind of a room is this? Is it an office, a bedroom or a kitchen? Give a one word answer, out of the given choices")
    ```
    """

    fixed: Any = field()


def _get_topic(topic: Union[Topic, Dict]) -> Topic:
    if isinstance(topic, Topic):
        return topic
    return Topic(**topic)


def _get_np_coordinates(
    pre_defined: List[Union[List, Tuple[np.ndarray, str]]],
) -> List[Union[List, Tuple[np.ndarray, str]]]:
    pre_defined_list = []
    for item in pre_defined:
        pre_defined_list.append((np.array(item[0]), item[1]))
    return pre_defined_list


@define(kw_only=True)
class MapLayer(BaseAttrs):
    """A MapLayer represents a single input for a MapEncoding component. It can subscribe to a specific text topic.

    :param subscribes_to: The topic that this map layer is subscribed to.
    :type subscribes_to: Topic
    :param temporal_change: Indicates whether the map should store changes over time for the same position. Defaults to False.
    :type temporal_change: bool
    :param resolution_multiple: A positive multiplication factor for the base resolution of the map grid, for fine or coarse graining the map. Defaults to 1.
    :type resolution_multiple: int
    :param pre_defined: An optional list of pre-defined data points in the layer. Each datapoint is a tuple of [position, text], where position is a numpy array of coordinates.
    :type pre_defined: list[tuple[np.ndarray, str]]

    Example of usage:
    ```python
    my_map_layer = MapLayer(subscribes_to='my_topic', temporal_change=True)
    ```
    """

    subscribes_to: Union[Topic, Dict] = field(converter=_get_topic)
    temporal_change: bool = field(default=False)
    resolution_multiple: int = field(
        default=1, validator=base_validators.in_range(min_value=0.1, max_value=10)
    )
    pre_defined: List[Union[List, Tuple[np.ndarray, str]]] = field(
        default=Factory(list), converter=_get_np_coordinates
    )


@define(kw_only=True)
class Route(BaseAttrs):
    """
    A Route defines a topic to be routed to by the SemanticRouter, along with samples of similar text that the input must match to for the route to be used.

    :param routes_to: The topic that the input to the SemanticRouter is routed to.
    :type routes_to: Topic
    :param samples: A list of sample text strings associated with this route.
    :type samples: list[str]

    Example of usage:
    ```python
    goto_route = Route(routes_to='goto', samples=['Go to the door', 'Go to the kitchen'])
    ```
    """

    routes_to: Union[Topic, Dict] = field(converter=_get_topic)
    samples: List[str] = field()

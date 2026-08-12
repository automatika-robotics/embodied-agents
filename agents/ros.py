"""The following classes provide wrappers for data being transmitted via ROS topics. These classes form the inputs and outputs of [Components](agents.components.md)."""

from enum import Enum
from typing import Callable, Union, Any, Dict, List, Optional, Tuple
import numpy as np
from attrs import define, field, Factory
from importlib.util import find_spec
from rclpy.logging import get_logger

from sensor_msgs.msg import JointState as JointStateROS

# FROM SUGARCOAT
from ros_sugar.supported_types import (
    SupportedType,
    Audio,
    Bool,
    CameraInfo,
    Image,
    CompressedImage,
    OccupancyGrid,
    Odometry,
    PointCloud2,
    Pose,
    PoseArray,
    PoseStamped,
    String,
    ROSImage,
    ROSCompressedImage,
    add_additional_datatypes,
    get_ros_msg_fields_dict,
    ros_msg_to_str,
)
from ros_sugar.io.topic import QoSConfig, Topic as BaseTopic

from ros_sugar.config import (
    BaseComponentConfig,
    ComponentRunType,
    BaseAttrs,
    base_validators,
)
from ros_sugar.core import BaseComponent, Monitor
from ros_sugar.core.component import MutuallyExclusiveCallbackGroup
from ros_sugar import UI_EXTENSIONS
from ros_sugar.utils import (
    component_action as _sugar_component_action,
    component_fallback,
    get_methods_with_decorator,
)
from ros_sugar.io.utils import run_external_processor
from ros_sugar.core import Event, Action
from ros_sugar import actions
from ros_sugar.base_clients import (
    ActionClientConfig,
    ActionClientHandler,
    ServiceClientConfig,
    ServiceClientHandler,
)

from .launcher import Launcher

# SUGATCOAT INTERFACES
from automatika_ros_sugar.srv import ExecuteMethod
from rcl_interfaces.srv import GetParameters

# AGENTS TYPES
from automatika_embodied_agents.msg import (
    Point2D,
    Bbox2D,
    Bbox3D as ROSBbox3D,
    Detections2D,
    Detections2DMultiSource,
    Detections3D as ROSDetections3D,
)
from automatika_embodied_agents.msg import (
    StreamingString as ROSStreamingString,
    Video as ROSVideo,
    Trackings as ROSTrackings,
    TrackingsMultiSource as ROSTrackingsMultiSource,
    PointsOfInterest as ROSPointsOfInterest,
)
from automatika_embodied_agents.action import MoveManipulator, VisionLanguageAction
from .callbacks import (
    DetectionsCallback,
    Detections3DCallback,
    DetectionsMultiSourceCallback,
    PointsOfInterestCallback,
    RGBDCallback,
    VideoCallback,
    StreamingStringCallback,
    JointStateCallback,
)

from .utils.actions import JointsData

__all__ = [
    "String",
    "StreamingString",
    "Video",
    "Audio",
    "Bool",
    "CameraInfo",
    "Image",
    "CompressedImage",
    "OccupancyGrid",
    "Odometry",
    "PointCloud2",
    "Pose",
    "PoseArray",
    "PoseStamped",
    "Detections",
    "Detections3D",
    "DetectionsMultiSource",
    "PointsOfInterest",
    "Trackings",
    "TrackingsMultiSource",
    "RGBD",
    "JointTrajectoryPoint",
    "JointTrajectory",
    "JointJog",
    "JointState",
    "Topic",
    "QoSConfig",
    "FixedInput",
    "base_validators",
    "BaseAttrs",
    "BaseComponent",
    "BaseComponentConfig",
    "ComponentRunType",
    "Launcher",
    "Monitor",
    "MemLayer",
    "MapLayer",
    "PriorMemory",
    "Route",
    "MutuallyExclusiveCallbackGroup",
    "Event",
    "actions",
    "Action",
    "component_fallback",
    "component_action",
    "VisionLanguageAction",
    "MoveManipulator",
    "GetParameters",
    "run_external_processor",
    "ServiceClientConfig",
    "ServiceClientHandler",
    "ActionClientConfig",
    "ActionClientHandler",
    "get_ros_msg_fields_dict",
    "ros_msg_to_str",
    "get_methods_with_decorator",
    "ActionPhase",
]


# =========================================================================
# Sugarcoat Overrides
# =========================================================================


class ActionPhase(str, Enum):
    """Cognitive phase in which a component action should be exposed.

    Read by Cortex when discovering tools on managed components.

    - ``PLANNING``: tool is only registered with the planner. Use for
      research / introspection tools that the planner calls while
      building a plan but that the executor has no reason to invoke.
    - ``EXECUTION``: tool is only registered with the executor. This is
      the default and matches the historical behavior of bare
      ``@component_action``. Use for state-changing actions (``say``,
      ``store_specific_memory``, ``start_episode``, ...).
    - ``BOTH``: tool is registered with both. Use for retrieval tools
      that the planner benefits from calling before emitting a plan
      (e.g. ``describe``, ``locate``) and that the executor may
      also need at run time.
    """

    PLANNING = "planning"
    EXECUTION = "execution"
    BOTH = "both"


def component_action(
    function: Optional[Callable] = None,
    *,
    description: Optional[Dict] = None,
    active: bool = False,
    phase: Union[ActionPhase, str] = ActionPhase.EXECUTION,
):
    """Wrapper around sugarcoat's ``component_action`` decorator.

    Delegates to ``ros_sugar.utils.component_action`` for the core
    behavior and additionally tags the method with ``_action_phase``
    — a hint to Cortex about whether the tool is a planning tool,
    an execution tool, or both.

    Can be used the same way as sugarcoat's decorator:

        @component_action(description={...}, phase=ActionPhase.BOTH)
        def c(self) -> str: ...

    If ``phase`` is not specified the action defaults to
    ``ActionPhase.EXECUTION``, preserving the previous behavior.

    :param function: The method being decorated (set when the decorator
        is used without parentheses).
    :param description: OpenAI-format tool description dict, forwarded
        verbatim to sugarcoat.
    :param active: If True, sugarcoat will reject calls while the
        component is not in the active lifecycle state.
    :param phase: Cognitive phase. Defaults to
        :attr:`ActionPhase.EXECUTION`.
    """
    phase_value = phase.value if isinstance(phase, ActionPhase) else str(phase)

    def _wrap(func: Callable) -> Callable:
        wrapped = _sugar_component_action(
            function=func, description=description, active=active
        )
        wrapped._action_phase = phase_value
        return wrapped

    if function is not None:
        return _wrap(function)
    return _wrap


# =========================================================================
# Additional Datatypes (Augment Sugarcoat datatypes)
# =========================================================================


class StreamingString(SupportedType):
    """
    Wraps the `automatika_embodied_agents.msg.StreamingString` message type.

    This type represents a string that is being streamed (e.g., token by token from an LLM).
    It contains fields to indicate if the stream is active and if the transmission is complete.

    **ROS2 Message Type**: `automatika_embodied_agents/msg/StreamingString`
    """

    callback = StreamingStringCallback
    _ros_type = ROSStreamingString

    @classmethod
    def convert(
        cls,
        output: str,
        stream: bool = False,
        done: bool = True,
        **_,
    ) -> ROSStreamingString:
        """
        Takes a string and streaming info to return a streaming string custom msg
        :return: ROSStreamingString
        """
        msg = ROSStreamingString()
        msg.stream = stream
        msg.done = done
        msg.data = output
        return msg


class Video(SupportedType):
    """
    Wraps the `automatika_embodied_agents.msg.Video` message type.

    This type represents a sequence of images (frames). It can handle both raw images and compressed images, bundling them into a single video message structure.

    **ROS2 Message Type**: `automatika_embodied_agents/msg/Video`
    """

    _ros_type = ROSVideo
    callback = VideoCallback
    _ui_rate_sampled = True  # lists of continuous frames

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


def _attach_source_image(msg: Any, image: Any) -> None:
    """Attach a source image (and its depth, if any) to a perception message.
    Also copies the image's header onto the message, so the detections carry
    the frame they were made in.

    :param msg: Perception message with image/compressed_image/depth fields
    :param image: Source ROS image message (Image, CompressedImage or RGBD)
    """
    if image is None:
        return
    if isinstance(image, ROSCompressedImage):
        msg.compressed_image = CompressedImage.convert(image)
        source = image
    # Handle RealSense RGBD msgs
    elif hasattr(image, "depth"):
        msg.image = Image.convert(image.rgb)
        msg.depth = Image.convert(image.depth)
        source = image.rgb
    else:
        msg.image = Image.convert(image)
        source = image

    if hasattr(source, "header") and hasattr(msg, "header"):
        msg.header = source.header


class Detections(SupportedType):
    """
    Wraps the `automatika_embodied_agents.msg.Detections2D` message type.

    This type represents 2D object detections, including bounding boxes, labels, and confidence scores.
    It can optionally bundle the source image (RGB or RGBD) associated with the detections.

    **ROS2 Message Type**: `automatika_embodied_agents/msg/Detections2D`
    """

    _ros_type = Detections2D
    callback = DetectionsCallback
    _ui_rate_sampled = True  # camera-rate frames + bbox drawing/JPEG encode

    @classmethod
    def convert(
        cls,
        output: Union[Dict, List[Dict]],
        images: Union[
            ROSImage,
            ROSCompressedImage,
            np.ndarray,
            List[ROSImage],
            List[ROSCompressedImage],
            List[np.ndarray],
        ],
        **_,
    ) -> Detections2D:
        """
        Takes object detection data and converts it into a ROS message
        of type Detection2D
        :return: Detection2D
        """
        if isinstance(output, List):
            output = output[0]
            images = images[0] if images else []
        msg = Detections2D()
        msg.scores = output.get("scores") or []
        msg.labels = output.get("labels") or []
        boxes = []
        for bbox in output.get("bboxes") or []:
            box = Bbox2D()
            box.top_left_x = float(bbox[0])
            box.top_left_y = float(bbox[1])
            box.bottom_right_x = float(bbox[2])
            box.bottom_right_y = float(bbox[3])
            boxes.append(box)

        msg.boxes = boxes
        if images:
            _attach_source_image(msg, images)
        return msg


class DetectionsMultiSource(SupportedType):
    """
    Wraps the `automatika_embodied_agents.msg.Detections2DMultiSource` message type.

    This type handles a list of `Detections2D` messages, typically used when receiving
    detection data from multiple cameras or sources simultaneously.

    **ROS2 Message Type**: `automatika_embodied_agents/msg/Detections2DMultiSource`
    """

    _ros_type = Detections2DMultiSource
    callback = DetectionsMultiSourceCallback
    _ui_rate_sampled = True  # camera-rate frames + bbox drawing/JPEG encode

    @classmethod
    def convert(cls, output: List, images: List, **_) -> Detections2DMultiSource:
        """
        Takes object detections data and converts it into a ROS message
        of type Detections2D
        :return: Detections2D
        """
        msg = Detections2DMultiSource()
        detections = []
        for img, detection in zip(images, output):
            detections.append(Detections.convert(detection, img))
        msg.detections = detections
        return msg


class Detections3D(SupportedType):
    """
    Wraps the `automatika_embodied_agents.msg.Detections3D` message type.

    This type represents detected objects in metric space, each with a
    labelled 3D bounding box in a named frame rather than in image space.

    **ROS2 Message Type**: `automatika_embodied_agents/msg/Detections3D`
    """

    _ros_type = ROSDetections3D
    callback = Detections3DCallback

    @classmethod
    def convert(
        cls,
        output: List,
        labels: Optional[List[str]] = None,
        scores: Optional[List[float]] = None,
        depth_validity: Optional[List[float]] = None,
        boxes_2d: Optional[List] = None,
        source_frame: str = "",
        **_,
    ) -> ROSDetections3D:
        """
        Takes 3D object detections and converts them into a ROS message
        of type Detections3D

        :param output: Boxes as (center, size) pairs, each in meters
        :return: Detections3D
        """
        msg = ROSDetections3D()
        boxes = []
        for center, size in output:
            box = ROSBbox3D()
            box.center.position.x = float(center[0])
            box.center.position.y = float(center[1])
            box.center.position.z = float(center[2])
            # Boxes are axis aligned in the frame they are given in
            box.center.orientation.w = 1.0
            box.size.x = float(size[0])
            box.size.y = float(size[1])
            box.size.z = float(size[2])
            boxes.append(box)

        msg.boxes = boxes
        msg.labels = labels or []
        msg.scores = [float(score) for score in scores or []]
        msg.depth_validity = [float(value) for value in depth_validity or []]
        msg.boxes_2d = boxes_2d or []
        msg.source_frame = source_frame
        return msg


class PointsOfInterest(SupportedType):
    """
    Wraps the `automatika_embodied_agents.msg.PointsOfInterest` message type.

    This type represents a set of 2D coordinates (x, y) on an image that are of interest,
    bundled with the source image or depth map.

    **ROS2 Message Type**: `automatika_embodied_agents/msg/PointsOfInterest`
    """

    _ros_type = ROSPointsOfInterest
    callback = PointsOfInterestCallback  # not defined
    _ui_rate_sampled = True  # camera-rate frames + point drawing/JPEG encode

    @classmethod
    def convert(
        cls,
        output: List[Tuple[int, int]],
        image: Union[ROSImage, ROSCompressedImage, np.ndarray],
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

        _attach_source_image(msg, image)
        return msg


class Trackings(SupportedType):
    """
    Wraps the `automatika_embodied_agents.msg.Trackings` message type.

    This type represents tracked objects over time. It includes object IDs, tracked labels,
    bounding boxes, centroids, and estimated velocities, along with the source image.

    **ROS2 Message Type**: `automatika_embodied_agents/msg/Trackings`
    """

    _ros_type = ROSTrackings
    callback = None  # Not defined in EmbodiedAgents

    @classmethod
    def convert(
        cls,
        output: Union[Dict, List[Dict]],
        images: Union[
            ROSImage,
            ROSCompressedImage,
            np.ndarray,
            List[ROSImage],
            List[ROSCompressedImage],
            List[np.ndarray],
        ],
        **_,
    ) -> ROSTrackings:
        """
        Takes tracking data and converts it into a ROS message
        of type Tracking
        :return: ROSTracking
        """
        # Only consider the first datapoint if a list is sent
        if isinstance(output, List):
            output = output[0]
            images = images[0]
        msg = ROSTrackings()
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
        # tracked_points: list of [x, y] center points from the tracker
        if o_tracked_points := output.get("tracked_points"):
            for point in o_tracked_points:
                centroid = Point2D()
                centroid.x = float(point[0])
                centroid.y = float(point[1])
                centroids.append(centroid)
        # tracked_bboxes: list of [x1, y1, x2, y2] bounding boxes
        if o_tracked_bboxes := output.get("tracked_bboxes"):
            for bbox in o_tracked_bboxes:
                box = Bbox2D()
                box.top_left_x = float(bbox[0])
                box.top_left_y = float(bbox[1])
                box.bottom_right_x = float(bbox[2])
                box.bottom_right_y = float(bbox[3])
                tracked_boxes.append(box)

        msg.boxes = tracked_boxes
        msg.centroids = centroids
        msg.estimated_velocities = estimated_velocities
        _attach_source_image(msg, images)
        return msg


class TrackingsMultiSource(SupportedType):
    """
    Wraps the `automatika_embodied_agents.msg.TrackingsMultiSource` message type.

    This type handles a list of `Trackings` messages, typically used for multi-camera
    tracking scenarios.

    **ROS2 Message Type**: `automatika_embodied_agents/msg/TrackingsMultiSource`
    """

    _ros_type = ROSTrackingsMultiSource
    callback = None  # Not defined

    @classmethod
    def convert(cls, output: List, images: List, **_) -> ROSTrackingsMultiSource:
        """
        Takes trackings data and converts it into a ROS message
        of type ROSTrackings
        :return: ROSTrackings
        """
        msg = ROSTrackingsMultiSource()
        trackings = []
        for img, tracking in zip(images, output):
            trackings.append(Trackings.convert(tracking, img))
        msg.trackings = trackings
        return msg


class RGBD(SupportedType):
    """
    Wraps the `realsense2_camera_msgs.msg.RGBD` message type.

    This type represents aligned RGB and Depth images typically produced by RealSense cameras.
    It requires the `realsense2_camera_msgs` package to be installed.

    **ROS2 Message Type**: `realsense2_camera_msgs/msg/RGBD`
    """

    callback = RGBDCallback
    _ui_rate_sampled = True  # camera-rate RGB-D frames

    @classmethod
    def get_ros_type(cls) -> type:
        if find_spec("realsense2_camera_msgs") is None:
            raise ModuleNotFoundError(
                "'realsense2_camera_msgs' module is required to use 'RGBD' msg type but it is not installed"
            )
        from realsense2_camera_msgs.msg import RGBD as RealSenseRGBD

        return RealSenseRGBD


class JointTrajectoryPoint(SupportedType):
    """
    Wraps the `trajectory_msgs.msg.JointTrajectoryPoint` message type.

    This type represents a single point in a joint trajectory, including positions,
    velocities, accelerations, and effort for a specific point in time.

    **ROS2 Message Type**: `trajectory_msgs/msg/JointTrajectoryPoint`
    """

    @classmethod
    def get_ros_type(cls) -> type:
        if find_spec("trajectory_msgs") is None:
            raise ModuleNotFoundError(
                "'trajectory_msgs' module is required to use 'JointTrajectory' msg type but it is not installed. Please install the 'ros-<distro>-trajectory-msgs' package."
            )
        from trajectory_msgs.msg import JointTrajectoryPoint as JointTrajectoryPointROS

        return JointTrajectoryPointROS

    @classmethod
    def _to_ros_duration(cls, seconds: float) -> Any:
        """Convert seconds as float to a builtin_interfaces Duration message"""
        from builtin_interfaces.msg import Duration

        sec = int(seconds)
        return Duration(sec=sec, nanosec=int((seconds - sec) * 1e9))

    @classmethod
    def convert(cls, output: JointsData, index: Optional[int] = None, **_) -> Any:
        """
        Takes joint state data and converts it into a ROS message
        of type JointTrajectoryPoint

        :return: JointTrajectory
        """
        msg = cls.get_ros_type()()
        # Point idx is reached after the initial delay plus the per point
        # duration of all points up to and including this one
        point_number = 1 if index is None else index + 1
        msg.time_from_start = cls._to_ros_duration(
            output.delay + output.duration * point_number
        )

        if index is None:
            msg.positions = output.positions.tolist()
            msg.velocities = output.velocities.tolist()
            msg.accelerations = output.accelerations.tolist()
            msg.effort = output.efforts.tolist()
            return msg

        if index < output.positions.shape[0]:
            msg.positions = output.positions[index].tolist()
        if index < output.velocities.shape[0]:
            msg.velocities = output.velocities[index].tolist()
        if index < output.accelerations.shape[0]:
            msg.accelerations = output.accelerations[index].tolist()
        if index < output.efforts.shape[0]:
            msg.effort = output.efforts[index].tolist()
        return msg


class JointTrajectory(SupportedType):
    """
    Wraps the `trajectory_msgs.msg.JointTrajectory` message type.

    This type represents a full joint trajectory, containing a list of `JointTrajectoryPoint`s
    and the names of the joints being controlled.

    **ROS2 Message Type**: `trajectory_msgs/msg/JointTrajectory`
    """

    callback = None

    @classmethod
    def get_ros_type(cls) -> type:
        if find_spec("trajectory_msgs") is None:
            raise ModuleNotFoundError(
                "'trajectory_msgs' module is required to use 'JointTrajectory' msg type but it is not installed. Please install the 'ros-<distro>-trajectory-msgs' package."
            )
        from trajectory_msgs.msg import JointTrajectory as JointTrajectoryROS

        return JointTrajectoryROS

    @classmethod
    def convert(cls, output: JointsData, **_) -> Any:
        """
        Takes joint state data and converts it into a ROS message
        of type JointTrajectory

        :return: JointTrajectory
        """
        msg = cls.get_ros_type()()
        msg.joint_names = output.joints_names
        msg.points = []

        if output.positions.ndim == 1:
            # a single point
            point_msg = JointTrajectoryPoint.convert(output)
            msg.points.append(point_msg)
            return msg

        if output.positions.ndim != 2:
            get_logger("joint_trajectory_publisher").error(
                f"Trying to publish invalid joint trajectory data. Expecting joint positions array dimension 2, got: `{output.positions.ndim}`"
            )
            return None

        # Get points data
        for idx in range(output.positions.shape[0]):
            point_msg = JointTrajectoryPoint.convert(output, index=idx)
            msg.points.append(point_msg)
        return msg


class JointJog(SupportedType):
    """
    Wraps the `control_msgs.msg.JointJog` message type.

    This type represents a command to jog joints, specifying displacements, velocities,
    or duration for immediate execution.

    **ROS2 Message Type**: `control_msgs/msg/JointJog`
    """

    callback = None

    @classmethod
    def get_ros_type(cls) -> type:
        if find_spec("control_msgs") is None:
            raise ModuleNotFoundError(
                "'control_msgs' module is required to use 'JointJog' msg type but it is not installed. Please install the 'ros-<distro>-control-msgs' package."
            )
        from control_msgs.msg import JointJog as JointJogROS

        return JointJogROS

    @classmethod
    def convert(cls, output: JointsData, **_) -> Any:
        """
        Takes joint state data and converts it into a ROS message
        of type JointJog

        :return: JointJog
        """
        msg = cls.get_ros_type()()
        msg.joint_names = output.joints_names

        msg.displacements = output.positions.tolist()
        msg.velocities = output.velocities.tolist()
        msg.duration = float(output.duration)

        return msg


class JointState(SupportedType):
    """
    Wraps the `sensor_msgs.msg.JointState` message type.

    This type represents the current state of a set of joints, including their names,
    positions, velocities, and efforts.

    **ROS2 Message Type**: `sensor_msgs/msg/JointState`
    """

    _ros_type = JointStateROS
    callback = JointStateCallback

    @classmethod
    def convert(cls, output: JointsData, **_) -> JointStateROS:
        """
        Takes joint state data and converts it into a ROS message
        of type JointState

        :return: JointState
        """
        msg = JointStateROS()
        msg.name = output.joints_names

        msg.position = output.positions.tolist()
        msg.velocity = output.velocities.tolist()
        msg.effort = output.efforts.tolist()

        return msg


agent_types = [
    StreamingString,
    Video,
    Detections,
    Detections3D,
    DetectionsMultiSource,
    Trackings,
    TrackingsMultiSource,
    PointsOfInterest,
    RGBD,
    JointState,
    JointJog,
    JointTrajectory,
    JointTrajectoryPoint,
]


add_additional_datatypes(agent_types)

# =========================================================================
# Additional UI Elements (Augment Sugarcoat UI elements)
# =========================================================================


def augment_ui():
    from .ui_elements import INPUT_ELEMENTS, OUTPUT_ELEMENTS

    return INPUT_ELEMENTS, OUTPUT_ELEMENTS


UI_EXTENSIONS["agents"] = augment_ui

# =========================================================================
# Additional Primitives (Augment Sugarcoat primitives)
# =========================================================================


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
    """Converter to get back a topic or an empty dictionary"""
    if isinstance(topic, Topic):
        return topic
    return Topic(**topic)


def _get_topic_or_action(
    entity: Union[Topic, Action, Dict, List[Union[Topic, Action, Dict]]],
) -> Union[Topic, Action, List[Topic], List[Action]]:
    """Converter to get back a topic, action, or a list of them."""

    # Handle List Input
    if isinstance(entity, list):
        converted_list = []
        for item in entity:
            if isinstance(item, (Topic, Action)):
                converted_list.append(item)
            elif isinstance(item, dict):
                # Convert Dict -> Topic
                converted_list.append(Topic(**item))
            else:
                raise TypeError(f"Invalid route item: {item}")
        return converted_list

    # Handle Single Item Input
    if isinstance(entity, (Topic, Action)):
        return entity

    # Handle Dict Input -> Topic
    return Topic(**entity)


def _to_position(
    value: Optional[Union[List[float], Tuple[float, ...]]],
) -> Optional[Tuple[float, float, float]]:
    """Normalize a position into an ``(x, y, z)`` float tuple (or ``None``)."""
    if value is None:
        return None
    coords = tuple(float(v) for v in value)
    if len(coords) != 3:
        raise ValueError(
            f"PriorMemory position must have 3 coordinates (x, y, z), got {len(coords)}"
        )
    return coords


@define(kw_only=True)
class PriorMemory(BaseAttrs):
    """A piece of prior knowledge used to seed a ``Memory`` layer at startup,
    before any live data arrives on the layer's topic.

    :param text: The memory text to store.
    :type text: str
    :param position: Optional world-frame ``(x, y, z)`` coordinates in
        meters. If omitted, the memory is stored at the origin
        ``(0.0, 0.0, 0.0)``.
    :type position: Optional[tuple[float, float, float]]
    :param timestamp: Optional POSIX timestamp in seconds to tag the
        memory with. If omitted, the component's current time is used
        when the memory is seeded.
    :type timestamp: Optional[float]

    Example of usage:
    ```python
    PriorMemory(text="charging dock", position=(1.5, 0.0, 0.0))
    ```
    """

    text: str = field()
    position: Optional[Tuple[float, float, float]] = field(
        default=None, converter=_to_position
    )
    timestamp: Optional[float] = field(default=None)


def _get_prior_memories(
    value: List[Union["PriorMemory", Dict]],
) -> List["PriorMemory"]:
    """Converter to rebuild ``PriorMemory`` items from dicts."""
    memories = []
    for item in value:
        if isinstance(item, PriorMemory):
            memories.append(item)
        elif isinstance(item, dict):
            memories.append(PriorMemory(**item))
        else:
            raise TypeError(
                "prior_memories items must be a PriorMemory or a dict, got "
                f"{type(item).__name__}"
            )
    return memories


@define(kw_only=True)
class MemLayer(BaseAttrs):
    """A MemLayer represents a single input layer for a ``Memory`` component.
    It subscribes to a topic whose callback produces a string representation
    that is stored as an observation.

    :param subscribes_to: The topic that this layer is subscribed to.
    :type subscribes_to: Topic
    :param prior_memories: An optional list of PriorMemory entries
        used to seed the layer with prior knowledge at startup, before any
        live data arrives on the topic. Each entry carries the memory text,
        an optional world-frame ``(x, y, z)`` position, and an optional
        timestamp.
    :type prior_memories: list[PriorMemory]
    :param is_internal_state: If True, observations from this layer are
        treated as internal state (interoception) by ``Memory``: they are
        written via ``add_body_state`` and are retrieved through the
        ``body_status`` tool rather than the perception tools. Use this
        for robot-internal signals like battery, temperature, or joint
        health. Defaults to False.
    :type is_internal_state: bool

    Example of usage:
    ```python
    my_layer = MemLayer(
        subscribes_to='my_topic',
        prior_memories=[PriorMemory(text="entrance", position=(0.0, 0.0, 0.0))],
    )
    battery_layer = MemLayer(subscribes_to='battery_state', is_internal_state=True)
    ```
    """

    subscribes_to: Topic = field(converter=_get_topic)
    prior_memories: List[PriorMemory] = field(
        default=Factory(list), converter=_get_prior_memories
    )
    is_internal_state: bool = field(default=False)


# Backwards-compatible alias
MapLayer = MemLayer


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

    routes_to: Union[Topic, Action, List[Topic], List[Action]] = field(
        converter=_get_topic_or_action
    )  # Only topics would get deserialized here
    samples: List[str] = field()

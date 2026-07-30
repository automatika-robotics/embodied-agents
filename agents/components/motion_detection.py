from typing import List, Optional, Union

import cv2
import numpy as np
from ros_sugar.tf import TFListener, TFListenerConfig

from ..config import MotionDetectorConfig
from ..ros import (
    Bool,
    Image,
    Odometry,
    PointCloud2,
    PoseArray,
    ROSCompressedImage,
    ROSImage,
    Topic,
    Video,
)
from ..utils import validate_func_args
from ..utils import motion
from .component_base import Component


class MotionDetector(Component):
    """
    This component detects motion (i.e. scene dynamics) from a stream of images or a stream of point clouds.

    A component instance takes either Image/CompressedImage input topics or PointCloud2 input topics (never both at once), and its available outputs depend on the input modality:

    - **Bool** output (both modalities): published motion state, True while motion is being detected. This output plugs directly into the EMOS event system, e.g. to trigger an MLLM component when something moves.
    - **Video** output (image inputs only): consecutive frames with perceivable motion are collected and published as a single video message once the motion stops (with a configurable debounce), i.e. the component makes intentionality decisions about what sequence of consecutive images should be treated as one coherent temporal sequence.
    - **PoseArray** output (point cloud inputs only): centers of the regions in which motion is being detected, published while motion is active. Multiple simultaneously moving regions produce multiple centers.

    Point cloud motion is detected by sparse voxel-occupancy differencing between consecutive clouds: appearing/disappearing voxels are thresholded for the motion state and clustered into motion centers.

    An optional ``position`` (Odometry) topic enables use on a moving robot:

    - With point cloud inputs, each cloud is transformed into the odometry frame before differencing (planar ego-motion subtraction), so static world geometry cancels out and motion centers are published in the odometry frame. If the transform betweeen cloud and robot base frame is not available, the cloud is assumed to be given in the robot base frame. Without a position topic, the sensor is assumed static and centers are published in the cloud frame.
    - With image inputs, motion detection is suppressed while the robot is moving faster than a configurable speed threshold.

    :param inputs: The input topics for motion detection.
        This should be a list of Topic objects of Image, CompressedImage or PointCloud2 type (one modality per component).
    :type inputs: list[Topic]
    :param outputs: The output topics for the detected motion.
        This should be a list of Topic objects of Bool, Video (image inputs) or PoseArray (point cloud inputs) type.
    :type outputs: list[Topic]
    :param config: The configuration for the motion detection.
        This should be an instance of MotionDetectorConfig. If not provided, defaults are used.
    :type config: MotionDetectorConfig
    :param trigger: The trigger topic(s) for the component. Input messages on these topics drive the processing.
        This can be a single Topic object or a list of Topic objects.
    :type trigger: Union[Topic, list[Topic]]
    :param position: Optional robot odometry topic used for ego-motion handling (see above).
        An Odometry topic given directly in ``inputs`` is treated the same way.
    :type position: Optional[Topic]
    :param component_name: The name of the motion detection component.
        This should be a string.
    :type component_name: str

    Example usage:
    ```python
    image_topic = Topic(name="image", msg_type="Image")
    video_topic = Topic(name="video", msg_type="Video")
    motion_topic = Topic(name="motion", msg_type="Bool")
    config = MotionDetectorConfig(motion_estimation_func="frame_difference")
    motion_detector = MotionDetector(
        inputs=[image_topic],
        outputs=[video_topic, motion_topic],
        config=config,
        trigger=image_topic,
        component_name="motion_detector",
    )
    ```
    """

    @validate_func_args
    def __init__(
        self,
        *,
        inputs: List[Topic],
        outputs: Optional[List[Topic]] = None,
        config: Optional[MotionDetectorConfig] = None,
        trigger: Union[Topic, List[Topic]],
        position: Optional[Topic] = None,
        component_name: str,
        **kwargs,
    ):
        if isinstance(trigger, float):
            raise TypeError(
                "MotionDetector component needs to be given a valid trigger topic. It cannot be started as a timed component."
            )

        self.config: MotionDetectorConfig = config or MotionDetectorConfig()

        # An Odometry input can only be the position topic, so it can be
        # given either through the position param or directly in inputs
        position = position or next(
            (t for t in inputs if issubclass(t.msg_type, Odometry)), None
        )
        self.position = position

        # Determine the input modality
        image_inputs = [t for t in inputs if issubclass(t.msg_type, Image)]
        cloud_inputs = [t for t in inputs if issubclass(t.msg_type, PointCloud2)]
        if image_inputs and cloud_inputs:
            raise TypeError(
                "MotionDetector can take either image inputs or point cloud inputs, not both at once. Create two component instances to process both modalities."
            )
        self._cloud_modality: bool = bool(cloud_inputs)

        self.allowed_inputs = {
            "Required": [[Image, PointCloud2]],
            "Optional": [Odometry],
        }
        # Allowed outputs depend on the input modality
        self.allowed_outputs = {
            "Required": [],
            "Optional": [Bool, PoseArray] if self._cloud_modality else [Bool, Video],
        }

        # Add position to inputs
        if position:
            if not issubclass(position.msg_type, Odometry):
                raise TypeError(
                    f"MotionDetector position topic must be of type Odometry, got {position.msg_type.__name__}"
                )
            if all(t.name != position.name for t in inputs):
                inputs = [*inputs, position]

        super().__init__(
            inputs,
            outputs,
            self.config,
            trigger,
            component_name,
            **kwargs,
        )

        # Check that position topic was not provided as a trigger
        if position and position.name in getattr(self, "trig_callbacks", {}):
            raise TypeError(
                "MotionDetector position topic cannot be used as the component trigger."
            )

        # Image modality state
        self._frames: Union[List[ROSImage], List[ROSCompressedImage]] = []
        self._last_frame: Optional[np.ndarray] = None
        self._roi_mask: Optional[np.ndarray] = None

        # Point cloud modality state
        self._previous_voxels: Optional[np.ndarray] = None
        self._sensor_tf_listener: Optional[TFListener] = None
        self._sensor_tf_warned: bool = False

        # Motion state machine
        self._motion_active: bool = False
        self._still_count: int = 0
        self._last_process_time: float = 0.0

    def custom_on_configure(self):
        """Configure."""
        super().custom_on_configure()

        # Sort publishers by output type
        publishers = list(self.publishers_dict.values())
        self._bool_publishers = [
            p for p in publishers if issubclass(p.output_topic.msg_type, Bool)
        ]
        self._video_publishers = [
            p for p in publishers if issubclass(p.output_topic.msg_type, Video)
        ]
        self._centers_publishers = [
            p for p in publishers if issubclass(p.output_topic.msg_type, PoseArray)
        ]
        if not publishers:
            self.get_logger().warning(
                "MotionDetector has no output topics. Motion will be detected but not published anywhere."
            )

        if self._cloud_modality and not self.position:
            self.get_logger().warning(
                "MotionDetector is processing point clouds WITHOUT a position (Odometry) topic: "
                "the sensor is assumed to be STATIC and any motion centers are published in the "
                "cloud frame. On a moving robot, provide the 'position' topic to enable ego-motion "
                "subtraction and world-frame motion centers."
            )

        # Fail fast on a missing torch installation before processing starts
        if self._cloud_modality and self.config.device == "cuda":
            motion.ensure_torch()

    def _execution_step(self, *_, **kwargs) -> None:
        """Processes an incoming image or point cloud and publishes the
        detected motion state, video segments (images) or motion centers
        (point clouds).
        """
        msg = kwargs.get("msg")
        topic = kwargs.get("topic")
        if msg is None or topic is None:
            return

        if not self._within_process_rate():
            return

        output = self.trig_callbacks[topic.name].get_output()
        if output is None:
            return

        if self._cloud_modality:
            self._process_cloud(output)
        else:
            self._process_image(msg, output)

    # ------------------------------------------------------------------
    # Common state machine
    # ------------------------------------------------------------------

    def _within_process_rate(self) -> bool:
        """Checks the optional processing rate limit."""
        if not self.config.process_rate:
            return True
        ros_time = self.get_ros_time()
        now = ros_time.sec + ros_time.nanosec * 1e-9
        if now - self._last_process_time < 1.0 / self.config.process_rate:
            return False
        self._last_process_time = now
        return True

    def _step_motion_state(self, detected: bool) -> bool:
        """Updates the motion state machine with a new detection result and
        publishes the Bool motion state.

        Motion is declared over only after ``motion_stop_delay`` consecutive
        still inputs.

        :return: True if a motion episode ended on this step
        :rtype: bool
        """
        episode_ended = False
        was_active = self._motion_active
        if detected:
            self._still_count = 0
            self._motion_active = True
        elif self._motion_active:
            self._still_count += 1
            if self._still_count >= self.config.motion_stop_delay:
                self._motion_active = False
                self._still_count = 0
                episode_ended = True

        if not self.config.publish_bool_on_change_only or (
            self._motion_active is not was_active
        ):
            for publisher in self._bool_publishers:
                publisher.publish(output=self._motion_active)

        return episode_ended

    # ------------------------------------------------------------------
    # Image modality
    # ------------------------------------------------------------------

    def _estimate_image_motion(self, current_gray: np.ndarray) -> bool:
        """Runs the configured motion estimation between the last two frames."""
        if self.config.motion_estimation_func == "frame_difference":
            return motion.frame_difference(
                self._last_frame, current_gray, self.config.threshold
            )
        elif self.config.motion_estimation_func == "optical_flow":
            return motion.optical_flow(
                self._last_frame,
                current_gray,
                self.config.threshold,
                self.config.flow_kwargs,
            )
        return True

    def _ego_motion_paused(self) -> bool:
        """True when image processing should pause because the robot itself is moving."""
        if not (self.position and self.config.pause_on_ego_motion):
            return False
        odom = self.callbacks[self.position.name].get_output()
        return odom is not None and abs(odom[4]) > self.config.ego_speed_threshold

    def _process_image(self, msg, output: np.ndarray) -> None:
        """Collects incoming image messages while motion is detected and
        publishes them as a video when the motion ends.
        """
        gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
        if self.config.roi_ignore_polygon:
            if self._roi_mask is None:
                self._roi_mask = motion.roi_mask(
                    gray.shape, self.config.roi_ignore_polygon
                )
            gray = gray * self._roi_mask

        detected = False
        if self._last_frame is not None and not self._ego_motion_paused():
            detected = self._estimate_image_motion(gray)
        self._last_frame = gray

        buffer_full = False
        if detected:
            self._frames.append(msg)
            # A full buffer ends the episode early so long motion produces
            # consecutive videos
            buffer_full = len(self._frames) >= self.config.max_video_frames

        episode_ended = self._step_motion_state(detected)

        if episode_ended or buffer_full:
            self._publish_video()

    def _publish_video(self) -> None:
        """Publishes the collected frames as a video message if long enough."""
        if len(self._frames) >= self.config.min_video_frames:
            self.get_logger().debug(f"Sending out video of {len(self._frames)} frames")
            for publisher in self._video_publishers:
                publisher.publish(output=self._frames)
        self._frames = []

    # ------------------------------------------------------------------
    # Point cloud modality
    # ------------------------------------------------------------------

    def _apply_sensor_extrinsic(
        self, points: np.ndarray, cloud_frame: str
    ) -> np.ndarray:
        """Transforms cloud points from the sensor frame to the robot base
        frame using the static sensor mount transform looked up from TF.

        Falls back to the points unchanged (cloud assumed in base frame)
        when the transform is not available.
        """
        if not cloud_frame or cloud_frame == self.config.base_frame:
            return points

        if self._sensor_tf_listener is None:
            self._sensor_tf_listener = self.create_tf_listener(
                TFListenerConfig(
                    source_frame=cloud_frame,
                    goal_frame=self.config.base_frame,
                    static_tf=True,
                )
            )

        if not self._sensor_tf_listener.got_transform:
            if not self._sensor_tf_warned:
                self.get_logger().warning(
                    f"Transform from cloud frame '{cloud_frame}' to base frame "
                    f"'{self.config.base_frame}' is not available (yet). Assuming the "
                    "cloud is given in the robot base frame until the transform is found."
                )
                self._sensor_tf_warned = True
            return points

        return motion.apply_transform(
            points,
            self._sensor_tf_listener.translation,
            self._sensor_tf_listener.rotation,
        )

    def _process_cloud(self, cloud) -> None:
        """Differences the incoming cloud against the previous one and
        publishes the motion state and the centers of changed regions.
        """
        points = cloud.xyz
        if points is None:
            return

        points = motion.crop_cloud(
            points,
            min_range=self.config.min_range,
            max_range=self.config.max_range,
            z_min=self.config.z_min,
            z_max=self.config.z_max,
        )

        frame_id = cloud.frame_id
        if self.position:
            odom = self.callbacks[self.position.name].get_output()
            if odom is not None:
                points = self._apply_sensor_extrinsic(points, cloud.frame_id)
                points = motion.transform_cloud(points, odom)
                frame_id = self.callbacks[self.position.name].frame_id or frame_id

        voxels = motion.unique_voxels(
            points, self.config.voxel_size, device=self.config.device
        )
        previous_voxels, self._previous_voxels = self._previous_voxels, voxels
        if previous_voxels is None:
            return

        changed = motion.voxel_set_difference(voxels, previous_voxels)
        detected = changed.size > self.config.changed_voxel_threshold

        self._step_motion_state(detected)

        if detected and self._centers_publishers:
            centers = motion.cluster_centers(
                changed,
                voxel_size=self.config.voxel_size,
                min_cluster_size=self.config.min_cluster_size,
                max_clusters=self.config.max_clusters,
            )
            if centers:
                self.get_logger().debug(
                    f"Detected {len(centers)} motion centers in frame '{frame_id}'"
                )
                for publisher in self._centers_publishers:
                    publisher.publish(output=centers, frame_id=frame_id)

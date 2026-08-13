from typing import Any, Union, Optional, List, Dict
import time
import queue
import threading
import os
import numpy as np
import cv2

from ..clients.model_base import ModelClient
from ..config import VisionConfig
from ..ros import (
    CameraInfo,
    CompressedImage,
    DetectionsMultiSource,
    Detections,
    Detections3D,
    Trackings,
    FixedInput,
    Image,
    RGBD,
    Topic,
    TrackingsMultiSource,
    ROSImage,
    ROSCompressedImage,
    component_action,
)
from ..utils import (
    validate_func_args,
    load_model,
    draw_points_2d,
    draw_detection_bounding_boxes,
    get_frame_id,
    get_stamp_secs,
)
from .model_component import ModelComponent
from .component_base import ComponentRunType


class Vision(ModelComponent):
    """
    This component performs object detection and tracking on input images and outputs a list of detected objects, along with their bounding boxes and confidence scores.

    :param inputs: The input topics for the object detection.
        This should be a list of Topic objects or FixedInput objects, limited to Image (or RGBD) type.
    :type inputs: list[Union[Topic, FixedInput]]
    :param outputs: The output topics for the object detection.
        This should be a list of Topic objects, Detection and Tracking types are handled automatically.
    :type outputs: list[Topic]
    :param model_client: Optional model client for the vision component to access remote vision models. If not provided, enable_local_classifier should be set to True in VisionConfig
        This should be an instance of ModelClient. Defaults to None.
    :type model_client: Optional[ModelClient]
    :param config: The configuration for the vision component.
        This should be an instance of VisionConfig. If not provided, defaults to VisionConfig().
    :type config: VisionConfig
    :param trigger: The trigger value or topic for the vision component.
        This can be a single Topic object, a list of Topic objects, or a float value for timed components.
    :type trigger: Union[Topic, list[Topic], float]
    :param depth: Optional depth image topic, for a camera that publishes depth separately
        rather than bundled in an RGBD message. The depth must be registered to the pictures
        being detected on, which is what a stereo camera publishes as its aligned or
        `depth_registered` stream. Only used when a Detections3D output is given, and it
        must be accompanied by `camera_info`.
    :type depth: Optional[Topic]
    :param camera_info: Optional CameraInfo topic describing the depth stream. Required
        alongside `depth`, and usable on its own with an RGBD input to override the
        calibration that message carries. Like `depth`, it is passed here rather than
        listed in `inputs`, which is reserved for pictures to detect on.
    :type camera_info: Optional[Topic]
    :param component_name: The name of the vision component.
        This should be a string and defaults to "vision_component".
    :type component_name: str

    Example usage:
    ```python
    image_topic = Topic(name="image", msg_type="Image")
    detections_topic = Topic(name="detections", msg_type="Detections")
    config = VisionConfig()
    model_client = ModelClient(model=DetectionModel(name='yolov5'))
    vision_component = Vision(
        inputs=[image_topic],
        outputs=[detections_topic],
        model_client=model_client
        config=config,
        component_name = "vision_component"
    )
    ```
    """

    @validate_func_args
    def __init__(
        self,
        *,
        inputs: List[Union[Topic, FixedInput]],
        outputs: List[Topic],
        model_client: Optional[ModelClient] = None,
        config: Optional[VisionConfig] = None,
        trigger: Union[Topic, List[Topic], float] = 1.0,
        depth: Optional[Topic] = None,
        camera_info: Optional[Topic] = None,
        component_name: str,
        **kwargs,
    ):
        self.config: VisionConfig = config or VisionConfig()
        self.allowed_inputs = {"Required": [[Image, RGBD]], "Optional": [CameraInfo]}
        self.handled_outputs = [
            Detections,
            Trackings,
            DetectionsMultiSource,
            TrackingsMultiSource,
            Detections3D,
        ]

        # Raw image captures
        self._images: List[Union[np.ndarray, ROSImage, ROSCompressedImage]] = []

        self.depth_topic, self.camera_info_topic = depth, camera_info

        # Reject camera info if present in inputs
        stray_info = [
            t.name
            for t in inputs
            if issubclass(t.msg_type, CameraInfo)
            and (camera_info is None or t.name != camera_info.name)
        ]
        if stray_info:
            raise TypeError(
                f"Vision was given CameraInfo topic(s) {stray_info} among its "
                "inputs. Intrinsics describe a camera rather than deliver "
                "pictures to detect on, so they are passed as "
                "`camera_info=Topic(...)`, as depth is passed as `depth=Topic(...)`."
            )
        self._aux_inputs = {t.name for t in (depth, camera_info) if t}

        # Asking for a Detections3D output turns 3D lifting on
        self._lift_to_3d = any(issubclass(t.msg_type, Detections3D) for t in outputs)
        self._lift_camera = self._check_3d_contract(inputs, depth, camera_info)

        # Add intrinsics and depth to inputs. They only get subscribed to with a
        # Detections3D output
        if self._lift_to_3d:
            for topic in (depth, camera_info):
                if topic and all(t.name != topic.name for t in inputs):
                    inputs = [*inputs, topic]

        # Sort which image inputs are actually run through the model, and which are
        # only there to be captured by component actions
        self._inference_set = self._resolve_detection_set(
            inputs, outputs, trigger, self._aux_inputs, self._lift_camera
        )
        self._spectators = [
            topic.name
            for topic in inputs
            if issubclass(topic.msg_type, (Image, RGBD))
            and topic.name not in self._inference_set
            and topic.name not in self._aux_inputs
        ]

        super().__init__(
            inputs,
            outputs,
            model_client,
            self.config,
            trigger,
            component_name,
            **kwargs,
        )

        if model_client:
            # check for correct model and setup number of trackers to be initialized if any
            if model_client.model_type != "VisionModel":
                raise TypeError(
                    "A vision component can only be started with a Vision Model"
                )
            if (
                hasattr(model_client, "_model")
                and self.model_client._model.setup_trackers  # type: ignore
            ):
                # one tracker per camera inferenced on, not per input
                model_client._model._num_trackers = len(self._inference_set)
        else:
            if not self.config.enable_local_classifier:
                raise TypeError(
                    "Vision component either requires a model client or enable_local_classifier needs to be set True in the VisionConfig."
                )

        # the detector for the 3D lifted camera and its parsed calibration,
        # each rebuilt only when the camera reports new values, and one-time warnings
        self._detector = None
        self._detector_key = None
        self._intrinsics = None
        self._intrinsics_key = None
        self._depth_encoding = None
        self._depth_encoding_key = None
        self._lift_index: Optional[int] = None
        self._lift_msg = None
        self._lift_depth = None
        self._warned: set = set()

        triggers = getattr(self, "trig_callbacks", {})
        if any(name in triggers for name in self._aux_inputs):
            raise TypeError(
                "Vision depth and camera_info topics describe a camera rather "
                "than deliver pictures to run inference on, so they cannot be used"
                " as the component trigger."
            )

    def _check_3d_contract(
        self,
        inputs: List[Union[Topic, FixedInput]],
        depth: Optional[Topic],
        camera_info: Optional[Topic],
    ) -> Optional[str]:
        """Check the component can produce the 3D detections it was asked for."""
        if not self._lift_to_3d:
            return None

        # Check if detection frame has been set. 3D Boxes are axis aligned in it.
        if not self.config.detections_frame:
            raise TypeError(
                "Vision was given a Detections3D output, which needs a frame to "
                "report boxes in. Set `detections_frame` on the VisionConfig to "
                "the frame the consumer works in, such as a robotic arm's planning "
                "frame."
            )

        pictures = [
            t
            for t in inputs
            if issubclass(t.msg_type, (Image, RGBD))
            and (not depth or t.name != depth.name)
        ]
        if camera_info and not issubclass(camera_info.msg_type, CameraInfo):
            raise TypeError(
                "Vision camera_info topic must be of type CameraInfo, got "
                f"{camera_info.msg_type.__name__}."
            )

        # Prefer RGBD over plain image. Return first RGBD if multiple.
        # Otherwise the first Image topic is taken at the end
        # and the rest are named in a warning.
        rgbd = [t for t in pictures if issubclass(t.msg_type, RGBD)]
        if rgbd:
            return rgbd[0].name

        if not depth:
            raise TypeError(
                "Vision was given a Detections3D output, which requires depth "
                "to place detections in space. Either give it an RGBD input, "
                "which carries depth registered to its picture, or pass the "
                "camera's registered depth topic as `depth=Topic(...)` along "
                f"with its `camera_info=Topic(...)`. Inputs given: "
                f"{[t.name for t in inputs]}"
            )
        if issubclass(depth.msg_type, CompressedImage) or not issubclass(
            depth.msg_type, Image
        ):
            raise TypeError(
                "Vision depth topic must be an uncompressed Image, got "
                f"{depth.msg_type.__name__}."
            )
        if not camera_info:
            raise TypeError(
                "Vision was given a depth topic but no camera_info topic. Depth "
                "pixels cannot be turned into distances without the calibration "
                "of the stream they were measured on: pass it as "
                "`camera_info=Topic(...)`."
            )
        # Take first image topic by default
        return pictures[0].name

    @staticmethod
    def _resolve_detection_set(
        inputs: List[Union[Topic, FixedInput]],
        outputs: List[Topic],
        trigger: Union[Topic, List[Topic], float],
        aux_inputs: Optional[set] = None,
        lift_camera: Optional[str] = None,
    ) -> List[str]:
        """Decide which image inputs are run through the model each tick.

        A component receives one picture per tick when it is triggered by a
        topic (EVENT mode), and all of its images at once when it is timed (TIMED mode).

        Single source outputs describe one camera, so they can only be used when
        a tick produces one picture. Image inputs left out are still subscribed
        and can be captured with component actions but they are not used for inference.

        :returns: Names of the image topics to run detection on
        """
        aux_inputs = aux_inputs or set()
        pictures = [
            t
            for t in inputs
            if issubclass(t.msg_type, (Image, RGBD)) and t.name not in aux_inputs
        ]
        multi_source = any(
            issubclass(t.msg_type, (DetectionsMultiSource, TrackingsMultiSource))
            for t in outputs
        )
        single_source = any(
            issubclass(t.msg_type, (Detections, Trackings)) for t in outputs
        )

        if isinstance(trigger, (int, float)):
            # Timed: the whole detection set reaches the model together, so
            # requires a multi message
            single = [t for t in pictures if t.name == lift_camera] or pictures[:1]
            inference_on = pictures if multi_source else single
            per_tick = len(inference_on)
        else:
            # Triggered: only the topic that fired is read, so a tick carries
            # one picture however many topics can trigger one
            triggers = trigger if isinstance(trigger, List) else [trigger]
            inference_on = [t for t in pictures if t.name in {t.name for t in triggers}]
            per_tick = 1

        if single_source and per_tick > 1:
            raise TypeError(
                f"{[t.name for t in inference_on]} are all used for inference in the "
                "same pass, so their inference results cannot be published on a "
                "Detections or Trackings topic, which describes one camera. "
                "Use a DetectionsMultiSource or TrackingsMultiSource output, "
                "or trigger the component on the cameras to take them one at "
                "a time."
            )
        return [t.name for t in inference_on]

    def custom_on_configure(self):
        # Warn which image inputs are never used for inference
        if self._spectators:
            self.get_logger().warning(
                f"Not running inference on {self._spectators}: this component "
                f"runs inference on {self._inference_set}. Those topics can still be "
                "captured with a component action (like take_picture and record_video). "
                "To run inference on all of them, give the component a "
                "DetectionsMultiSource or TrackingsMultiSource output topic, "
                "or make them all component triggers."
            )

        # Depth and intrinsics are only read to place detections in space
        if not self._lift_to_3d and (unused := sorted(self._aux_inputs)):
            self.get_logger().warning(
                f"Ignoring {unused}: depth and intrinsics are only used to place "
                "detections in metric space, and this component has no "
                "Detections3D output topic to publish those on. The topics are "
                "not being subscribed to."
            )

        # Only one camera's depth and calibration are given, an RGBD topic or
        # the first Image topic will be used in case of more than one inputs
        if self._lift_camera and len(self._inference_set) > 1:
            self.get_logger().warning(
                f"Lifting detections into 3D only from '{self._lift_camera}': "
                f"the depth and calibration given describe that camera. "
                f"Detections from {[n for n in self._inference_set if n != self._lift_camera]} "
                "are only published in 2D."
            )

        # deploy local model if enabled
        if not self.model_client and self.config.enable_local_classifier:
            self._deploy_local_model()

        # configure parent component
        super().custom_on_configure()

        # create visualization thread if enabled
        if self.config.enable_visualization:
            self.queue = queue.Queue()
            self.stop_event = threading.Event()
            self.visualization_thread = threading.Thread(target=self._visualize)
            self.visualization_thread.start()

    def _deploy_local_model(self):
        """Deploy local vision model on demand."""
        if self.local_model is not None:
            return  # already deployed
        from ..utils.local_vision import LocalVisionModel, _MS_COCO_LABELS

        if not self.config.dataset_labels:
            self.get_logger().warning(
                "No dataset labels provided for the local model, using default MS_COCO labels"
            )
            self.config.dataset_labels = _MS_COCO_LABELS

        # Auto-enable config flag
        self.config.enable_local_classifier = True

        self.local_model = LocalVisionModel(
            model_path=load_model(
                "local_classifier", self.config.local_classifier_model_path
            ),
            ncpu=self.config.ncpu_local_classifier,
            device=self.config.device_local_classifier,
            input_height=self.config.input_height,
            input_width=self.config.input_width,
            dataset_labels=self.config.dataset_labels,
        )

    def custom_on_deactivate(self):
        # if visualization is enabled, shutdown the thread
        if self.config.enable_visualization:
            if self.visualization_thread:
                self.stop_event.set()
                self.visualization_thread.join()
        # deactivate component
        super().custom_on_deactivate()

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "take_picture",
                "description": "Capture a photo from a camera topic and save it to disk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic_name": {
                            "type": "string",
                            "description": "Name of the input topic to capture from. Should be name of one of the topics which are inputs of this component.",
                        },
                        "save_path": {
                            "type": "string",
                            "description": "Directory path where the image will be saved. The default path is ~/emos/pictures, unless specifically asked, don't use another path.",
                        },
                    },
                    "required": ["topic_name"],
                },
            },
        }
    )
    def take_picture(
        self,
        topic_name: str,
        save_path: str = "~/emos/pictures",
        timeout: float = 0.5,
    ) -> str:
        """
        Take a picture from a specific input topic and save it to the specified location.

        This method acts as an Action to capture a specific frame from a specific camera/topic.
        It prioritizes triggers over standard inputs if a name conflict exists (though unique names are expected).

        :param topic_name: The name of the topic to capture the image from.
                           Must be one of the component's registered input topics.
        :type topic_name: str
        :param save_path: The directory path where images will be saved.
                          Defaults to "~/emos/pictures".
        :type save_path: str
        :param timeout: Timeout if an image is not available on the topic.
                          Defaults to 0.5 seconds.
        :type timeout: float
        :return: The full path to the saved image file.
        :rtype: str
        :raises ValueError: If the topic is not one of the component inputs.
        :raises TimeoutError: If no image was received within the timeout.
        """
        # Preflight check for timed components
        if (
            self.run_type == ComponentRunType.TIMED
            and (loop_time := 1 / self.config.loop_rate) > timeout
        ):
            self.get_logger().warning(
                f"Warning: take_picture timeout ({timeout}s) is strictly shorter than the component's trigger period ({loop_time}s) for this timed component. "
                f"The action is highly likely to timeout before the image callback executes. Consider running the component faster or increasing the timeout for this action."
            )
        # Expand user path
        save_path = os.path.expanduser(save_path)
        os.makedirs(save_path, exist_ok=True)

        # Identify callback type
        trig_dict = getattr(self, "trig_callbacks", {})
        target_callback = trig_dict.get(topic_name) or self.callbacks.get(topic_name)
        if not target_callback:
            raise ValueError(
                f"Topic '{topic_name}' is not one of the component inputs. "
                "You can only take pictures on topics that are provided as "
                "inputs to this component."
            )

        # if target is a trigger, issue a warning
        is_trigger = topic_name in trig_dict
        if is_trigger:
            self.get_logger().warning(
                f"Capturing image from trigger '{topic_name}'. Inference paused momentarily."
            )
        # save callback state
        original_callback = target_callback._extra_callback
        original_get_processed = target_callback._get_processed

        # Define a single frame interceptor function
        frames = []

        # extra callback for capturing image
        def single_frame_interceptor(msg, topic, output=None):
            if output is not None and not frames:
                frames.append(output.copy())

        # Swap extracallback, wait and restore
        try:
            target_callback.on_callback_execute(
                single_frame_interceptor, get_processed=True
            )

            start_time = time.time()
            while (time.time() - start_time) < timeout and not frames:
                time.sleep(0.01)  # Check frequently

        finally:
            # Always restore the original callback state
            if original_callback:
                target_callback.on_callback_execute(
                    original_callback, get_processed=original_get_processed
                )
            else:
                target_callback._extra_callback = None

        if not frames:
            raise TimeoutError(
                f"Timeout: No image received on '{topic_name}' within {timeout}s."
            )

        # Save Image
        timestamp = int(time.time() * 1000)
        filename = f"capture_{topic_name}_{timestamp}.jpg"
        full_path = os.path.join(save_path, filename)

        # Ensure BGR for OpenCV saving
        save_img = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
        cv2.imwrite(full_path, save_img)
        self.get_logger().info(f"Saved picture to {full_path}")

        return f"Picture from '{topic_name}' saved to {full_path}"

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "record_video",
                "description": "Record a video from a camera topic for a set duration and save it to disk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic_name": {
                            "type": "string",
                            "description": "Name of the input topic to record from. Should be name of one of the topics which are inputs of this component.",
                        },
                        "duration": {
                            "type": "number",
                            "description": "Duration of the recording in seconds. Default is 5 seconds",
                        },
                        "save_path": {
                            "type": "string",
                            "description": "Directory path where the video will be saved. The default path is ~/emos/videos, unless specifically asked, don't use another path.",
                        },
                        "fps": {
                            "type": "integer",
                            "description": "Frames per second for the recording. Default is 30",
                        },
                    },
                    "required": ["topic_name"],
                },
            },
        }
    )
    def record_video(
        self,
        topic_name: str,
        duration: float = 5.0,
        save_path: str = "~/emos/videos",
        fps: int = 30,
    ) -> str:
        """
        Record a video from a specific input topic for a set duration.

        This action spawns a background thread to capture frames and save them to a video file.
        It does not block the main execution loop.

        :param topic_name: The name of the topic to record from.
        :type topic_name: str
        :param duration: The duration of the recording in seconds. Defaults to 5.0.
        :type duration: float
        :param save_path: The directory path where the video will be saved.
                          Defaults to "~/emos/videos".
        :type save_path: str
        :param fps: The frames per second for the recording. Defaults to 20.
        :type fps: int
        :return: A confirmation message describing the started recording.
        :rtype: str
        :raises ValueError: If the topic is not one of the component inputs.
        """
        # Preflight checks for timed components
        if self.run_type == ComponentRunType.TIMED:
            if self.config.loop_rate < fps:
                self.get_logger().warning(
                    f"Warning: Requested {fps} FPS, but the component's trigger period is {1 / self.config.loop_rate}s "
                    f"(~{self.config.loop_rate:.2f} FPS max). The recorded video will heavily repeat frames or play too fast. Consider running the component faster or reduce the fps"
                )

            if duration < 1 / self.config.loop_rate:
                self.get_logger().warning(
                    f"Warning: Recording duration ({duration}s) is shorter than the component's loop period "
                    f"({1 / self.config.loop_rate}s). You are likely to capture 0 frames. Consider running the component faster or increase duration."
                )
        # Expand user path
        save_path = os.path.expanduser(save_path)
        os.makedirs(save_path, exist_ok=True)

        trig_dict = getattr(self, "trig_callbacks", {})
        target_callback = trig_dict.get(topic_name) or self.callbacks.get(topic_name)
        # Identify callback type
        if not target_callback:
            raise ValueError(
                f"Topic '{topic_name}' is not one of the component inputs. "
                "You can only record videos on topics that are provided as "
                "inputs to this component."
            )

        # if target is a trigger, issue a warning
        is_trigger = topic_name in trig_dict
        if is_trigger:
            self.get_logger().warning(
                f"Recording video on trigger topic '{topic_name}'. "
                f"Detection or tracking will be PAUSED for {duration} seconds!"
            )

        # Spawn the background thread
        recording_thread = threading.Thread(
            target=self._record_video_thread,
            kwargs={
                "target_callback": target_callback,
                "topic_name": topic_name,
                "duration": duration,
                "save_path": save_path,
                "fps": fps,
                "is_trigger": is_trigger,
            },
            daemon=True,
        )
        recording_thread.start()
        self.get_logger().info(
            f"Started recording video on topic '{topic_name}' for {duration} seconds."
        )
        # NOTE: We do not wait to join the video recording thread, any failures
        # will be silent. This can be changed when action execution
        # infrastructure is in place to allow for monitoring and returning
        # results from async actions.

        return (
            f"Started recording {duration}s video from '{topic_name}' at {fps} FPS. "
            f"Video will be saved to {save_path}."
        )

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "track",
                "description": (
                    "Start tracking objects with the given label in the camera feed. "
                    "This tool is a pre-requisite for starting vision based following "
                    "controllers. "
                    "Requires a remote RoboML model client (not a local model) and "
                    "at least one Tracking output topic on this component. "
                    "Once started, tracking results are published continuously."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Object label to track (e.g. 'person', 'cup').",
                        },
                    },
                    "required": ["label"],
                },
            },
        }
    )
    def track(self, label: str) -> str:
        """Start tracking objects matching the given label.

        Configures the remote model server to enable ByteTrack trackers
        (reinitializing if needed) and sets the label filter so that
        tracking results are published on the component's Tracking output
        topics.

        :param label: Object label to track (e.g. 'person', 'cup').
        :type label: str
        :return: A confirmation message describing the started tracking.
        :rtype: str
        :raises RuntimeError: If the component does not have a remote
            RoboML model client or a Tracking output topic.
        """
        from ..clients.roboml import RoboMLHTTPClient, RoboMLRESPClient

        # must have a remote RoboML client
        if not self.model_client or not isinstance(
            self.model_client, (RoboMLHTTPClient, RoboMLRESPClient)
        ):
            raise RuntimeError(
                "Tracking requires a RoboML model client. "
                "Local models do not support tracking."
            )

        # must have a Tracking output topic
        has_tracking_output = any(
            t.msg_type in (Trackings, TrackingsMultiSource) for t in self.out_topics
        )
        if not has_tracking_output:
            raise RuntimeError(
                "Tracking requires at least one output topic of type "
                "Trackings or TrackingsMultiSource."
            )

        with self.safe_restart():
            init_params = self.model_client.model_init_params
            # Enable trackers on the model if not already set up
            if not init_params.get("setup_trackers"):
                init_params["setup_trackers"] = True
            init_params["num_trackers"] = len(self.in_topics)

        self.get_logger().info("Trackers initialized on remote model server.")

        # Set the label to track
        self.config.labels_to_track = [label]
        self.inference_params = self.config._get_inference_params()
        self.get_logger().info(f"Now tracking: '{label}'")
        return (
            f"Tracking started for label '{label}'. Tracking results are now "
            "being published on the component's Tracking output topics."
        )

    def _record_video_thread(
        self,
        target_callback,
        topic_name: str,
        duration: float,
        save_path: str,
        fps: int,
        is_trigger: bool,
    ):
        """
        Internal worker thread to buffer frames and write video to disk.
        """
        frames = []

        # Save current callback state
        original_callback = target_callback._extra_callback
        original_get_processed = target_callback._get_processed

        # extra callback for capturing images
        def frame_interceptor(msg, topic, output=None):
            if output is not None:
                frames.append(output.copy())

        try:
            target_callback.on_callback_execute(frame_interceptor, get_processed=True)
            time.sleep(duration)
        finally:
            # Safely restore execution step or original state
            if original_callback:
                target_callback.on_callback_execute(
                    original_callback, get_processed=original_get_processed
                )
                if is_trigger:
                    self.get_logger().info(
                        f"Video recording finished. Vision inference RESUMED on '{topic_name}'."
                    )
            else:
                target_callback._extra_callback = None

        if not frames:
            self.get_logger().warning(
                f"No frames captured for video on topic '{topic_name}'."
            )
            return

        # Encode video
        timestamp = int(time.time() * 1000)
        filename = f"recording_{topic_name}_{timestamp}.mp4"
        full_path = os.path.join(save_path, filename)

        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(full_path, fourcc, fps, (width, height))

        # Subsample frames to match requested FPS
        actual_frames = len(frames)
        target_frames = int(duration * fps)

        if actual_frames > 0:
            step = max(1, actual_frames / target_frames)
            for i in range(target_frames):
                idx = int(i * step)
                if idx < actual_frames:
                    bgr_frame = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR)
                    out.write(bgr_frame)

        out.release()
        self.get_logger().info(f"Video saved successfully: {full_path}")

    def _visualize(self):
        """CV2 based visualization of inference results"""
        cv2.namedWindow(self.node_name)

        while not self.stop_event.is_set():
            try:
                # Add timeout to periodically check for stop event
                data = self.queue.get(timeout=1)
            except queue.Empty:
                self.get_logger().warning(
                    "Visualization queue is empty, waiting for new data..."
                )
                continue

            # Only handle the first image and its output
            image = cv2.cvtColor(
                data["images"][0], cv2.COLOR_RGB2BGR
            )  # as cv2 expects a BGR

            bounding_boxes = data["output"][0].get("bboxes", [])
            labels = data["output"][0].get("labels", [])
            tracked_objects = data["output"][0].get("tracked_points", [])

            image = draw_detection_bounding_boxes(
                image, bounding_boxes, labels, handle_bbox2d_msg=False
            )

            for point_list in tracked_objects:
                # Each point_list is a list of points on one tracked object
                image = draw_points_2d(image, point_list)

            cv2.imshow(self.node_name, image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.get_logger().warning("User pressed 'q', stopping visualization.")
                break

        cv2.destroyAllWindows()

    def _depth_snapshot(self) -> Optional[Any]:
        """Depth as it stands the instant its picture is taken.

        Read alongside the picture rather than at publish time.
        Not needed for RGBD msgs
        """
        if self.depth_topic and (callback := self.callbacks.get(self.depth_topic.name)):
            return callback.msg
        return None

    def _depth_for(self) -> Optional[Any]:
        """The depth image registered to the lifted picture, or None"""
        # For RGBD
        if (depth := getattr(self._lift_msg, "depth", None)) is not None and depth.data:
            return depth

        if (depth := self._lift_depth) is None:
            self._warn_once(
                "no_depth",
                f"Nothing has been received on depth topic "
                f"'{self.depth_topic.name if self.depth_topic else '<unknown>'}', so no detection "
                "can be placed in space.",
                error=True,
            )
            return None

        # For Image. Captured at the same instant, check for age nonetheless
        age = abs(get_stamp_secs(self._lift_msg) - get_stamp_secs(depth))
        if age > self.config.max_depth_age:
            self._warn_once(
                "depth_age",
                f"Depth on '{self.depth_topic.name}' is {age:.2f}s away from the "
                f"picture it would be paired with, more than max_depth_age "
                f"({self.config.max_depth_age}s), so these detections are not "
                "being published.",
            )
            return None
        return depth

    def _camera_intrinsics(self) -> Optional[Any]:
        """Calibration of the stream the depth was measured on.
        A camera_info topic wins when one was given.
        """

        from ..ros import read_camera_info

        if self.camera_info_topic and (
            info := self.callbacks.get(self.camera_info_topic.name)
        ):
            if (parsed := info.get_output()) is not None:
                return parsed

        info = getattr(self._lift_msg, "depth_camera_info", None)
        if info is None:
            return None
        key = (
            info.header.frame_id,
            info.width,
            info.height,
            tuple(info.k),
            tuple(info.p),
        )
        # Update key if it changed
        if key != self._intrinsics_key:
            self._intrinsics = read_camera_info(info)
            self._intrinsics_key = key
        return self._intrinsics

    def _depth_detector(self):
        """The 3D detector for the lifted camera, and the depth to run it on.

        :returns: (detector, depth in millimeters), both None when this tick
            cannot be lifted
        """
        from ..callbacks import image_pre_processing, process_encoding

        from ..utils.perception3d import make_detector, prepare_depth

        depth_msg = self._depth_for()
        if depth_msg is None:
            return None, None

        intrinsics = self._camera_intrinsics()
        if intrinsics is None:
            self._warn_once(
                "no_intrinsics",
                "No camera calibration has been received, so depth cannot be "
                "turned into distances. Pass the camera's `camera_info` topic, "
                "or use an RGBD input, which carries its own.",
                error=True,
            )
            return None, None

        # The intrinsics need to describe the depth image
        if (intrinsics.width, intrinsics.height) != (depth_msg.width, depth_msg.height):
            self._warn_once(
                "intrinsics_resolution",
                f"The camera reports intrinsics for {intrinsics.width}x"
                f"{intrinsics.height} images but publishes depth at "
                f"{depth_msg.width}x{depth_msg.height}. Detections cannot be "
                "placed in metric space until the two agree.",
                error=True,
            )
            return None, None

        # Check if the streams encoding changed (unlikely)
        if depth_msg.encoding != self._depth_encoding_key:
            self._depth_encoding = process_encoding(depth_msg.encoding)
            self._depth_encoding_key = depth_msg.encoding

        depth_mm = prepare_depth(
            image_pre_processing(depth_msg, *self._depth_encoding),
            encoding=depth_msg.encoding,
            scale=self.config.depth_scale,
        )

        # Registered depth shares the color image's pixel grid preferrably
        color_frame = get_frame_id(getattr(self._lift_msg, "rgb", self._lift_msg))
        depth_frame = get_frame_id(depth_msg)
        if color_frame and depth_frame and color_frame != depth_frame:
            self._warn_once(
                "frame_mismatch",
                f"The depth stream reports frame '{depth_frame}' while the "
                f"pictures are in '{color_frame}'. Depth registered to the "
                "color image lives in the color camera's frame, so that is "
                "the one used. If this depth stream is not the registered "
                "one, every 3D box will be misplaced.",
            )
        camera_frame = color_frame or depth_frame
        target = self.config.detections_frame
        translation = rotation = None
        if target != camera_frame:
            listener = self.get_transform_listener(
                camera_frame, target, self.config.static_camera_tf
            )
            if not listener.got_transform:
                self._warn_once(
                    "camera_transform",
                    f"The transform from camera frame '{camera_frame}' to "
                    f"'{target}' has not been resolved yet, so detections are "
                    "not being published in 3D.",
                )
                return None, None
            translation, rotation = listener.translation, listener.rotation

        # Rebuilding is cheap, so the detector is kept only until something it
        # was built from moves
        key = (
            camera_frame,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
            None if translation is None else tuple(translation),
            None if rotation is None else tuple(rotation),
        )
        if key != self._detector_key:
            self._detector = make_detector(
                intrinsics,
                translation=translation,
                rotation=rotation,
                depth_range=(self.config.min_depth, self.config.max_depth),
            )
            self._detector_key = key
        return self._detector, depth_mm

    # TODO: Upstream to sugarcoat component
    def _warn_once(self, key: str, message: str, error: bool = False) -> None:
        """Report a lasting misconfiguration the first time it is noticed"""
        if key in self._warned:
            return
        self._warned.add(key)
        if error:
            self.get_logger().error(message)
        else:
            self.get_logger().warning(message)

    def _create_input(self, *_, **kwargs) -> Optional[Dict[str, Any]]:
        """Create inference input for ObjectDetection models
        :param args:
        :param kwargs:
        :rtype: dict[str, Any]
        """
        self._images = []
        # Where the lifted camera's picture sits in this tick's results
        self._lift_index: Optional[int] = None
        self._lift_msg = None
        self._lift_depth = None

        # set one image topic as query for event based trigger
        if trigger := kwargs.get("topic"):
            if msg := self.trig_callbacks[trigger.name].msg:
                self._images.append(msg)
                if trigger.name == self._lift_camera:
                    self._lift_index, self._lift_msg = 0, msg
                    self._lift_depth = self._depth_snapshot()
            images = [self.trig_callbacks[trigger.name].get_output(clear_last=True)]
        else:
            images = []

            for name, i in self.callbacks.items():
                # Inputs outside the inference set are ignored
                if name not in self._inference_set:
                    continue
                msg = i.msg
                if (item := i.get_output(clear_last=True)) is not None:
                    if name == self._lift_camera and msg is not None:
                        self._lift_index, self._lift_msg = len(images), msg
                        self._lift_depth = self._depth_snapshot()
                    images.append(item)
                    if msg is not None:
                        self._images.append(msg)

        if not images:
            return None

        return {"images": images, **self.inference_params}

    def _publish(self, result, **kwargs) -> None:
        """Publish the detections, giving each topic the shape it can carry.

        Inference returns one set of detections per image. A single source
        message describes one camera and takes that camera's set on its own,
        while a multi source message takes the whole list.
        """
        output = result.pop("output")
        images = kwargs.pop("images", None)
        boxes = self._lift(output) if self._lift_to_3d else None

        for publisher in self.publishers_dict.values():
            msg_type = publisher.output_topic.msg_type
            if issubclass(msg_type, Detections3D):
                # NOTE: None means the camera could not be used this tick
                # (3D lifting didn't work), and nothing goes out. A scene the camera
                # did see as empty comes through as empty fields and is published
                if boxes is not None:
                    publisher.publish(
                        boxes["output"],
                        **{k: v for k, v in boxes.items() if k != "output"},
                        frame_id=self.config.detections_frame,
                        time_stamp=kwargs.get("time_stamp"),
                    )
            elif issubclass(msg_type, (Detections, Trackings)):
                publisher.publish(
                    output[0],
                    images=images[0] if images else None,
                    **result,
                    **kwargs,
                )
            else:
                publisher.publish(output, images=images, **result, **kwargs)

    def _lift(self, output: List[Dict]) -> Optional[Dict[str, Any]]:
        """Turn the lifted camera's detections into metric boxes.

        :returns: Fields for the Detections3D converter, or None when the
            camera could not be used this tick
        """
        from ..utils.perception3d import (
            boxes_from_detections,
            detections_to_message_fields,
        )

        if self._lift_index is None or self._lift_index >= len(output):
            self._warn_once(
                "no_lift_frame",
                f"No picture arrived on '{self._lift_camera}', so nothing can "
                "be placed in space this tick.",
            )
            return None

        detections = output[self._lift_index] or {}
        pixels = detections.get("bboxes") or []
        if not pixels:
            # NOTE: The camera worked and saw nothing; unlike the None returns on
            # this path, this is an observation, published as an empty message
            # so consumers let go of objects that are no longer there
            return detections_to_message_fields([])

        # the color image, which an RGBD frame carries nested
        color = getattr(self._lift_msg, "rgb", self._lift_msg)

        detector, depth_mm = self._depth_detector()
        if detector is None:
            return None

        lifted = [
            box
            for box in boxes_from_detections(
                detector,
                depth_mm,
                pixels,
                image_size=(color.width, color.height),
                depth_range=(self.config.min_depth, self.config.max_depth),
            )
            # A box built from a handful of depth pixels is not trustworthy
            if box.validity >= self.config.min_depth_validity
        ]
        return detections_to_message_fields(
            lifted,
            labels=detections.get("labels"),
            scores=detections.get("scores"),
            boxes_2d=pixels,
        )

    def _source_frame(self) -> Optional[str]:
        """Frame of the camera the detections were made in, None unless
        exactly one camera contributed.
        """
        if len(self._images) != 1:
            return None
        return get_frame_id(getattr(self._images[0], "rgb", self._images[0])) or None

    def _execution_step(self, *args, **kwargs):
        """_execution_step.

        :param args:
        :param kwargs:
        """

        if self.run_type is ComponentRunType.EVENT and (trigger := kwargs.get("topic")):
            if not trigger:
                return
            self.get_logger().debug(f"Received trigger on topic {trigger.name}")
        else:
            time_stamp = self.get_ros_time().sec
            self.get_logger().debug(f"Sending at {time_stamp}")

        # create inference input
        inference_input = self._create_input(*args, **kwargs)
        # call model inference
        if not inference_input:
            self.get_logger().warning("Input not received, not calling model inference")
            return

        # conduct inference
        result = self._call_inference(inference_input, unpack=True)
        if not result:
            return

        # result acquired, publish inference result
        self._publish(
            result,
            images=self._images,
            frame_id=self._source_frame(),
            time_stamp=self.get_ros_time(),
        )
        if self.config.enable_visualization:
            result["images"] = inference_input["images"]
            self.queue.put_nowait(result)

    def _warmup(self):
        """Warm up and stat check"""
        import time
        from pathlib import Path

        if (
            hasattr(self, "trig_callbacks")
            and (image := list(self.trig_callbacks.values())[0].get_output())
            is not None
        ):
            self.get_logger().warning("Got image input from trigger topic")
        else:
            self.get_logger().warning(
                "Did not get image input from trigger topic. Camera device might not be working and topic is not being published to, using a test image."
            )
            image = cv2.imread(
                str(Path(__file__).parents[1] / Path("resources/test.jpeg"))
            )

        inference_input = {"images": [image], **self.inference_params}

        # Run inference once to warm up and once to measure time
        if self.model_client:
            self.model_client.inference(inference_input)
        elif self.local_model:
            self.local_model(inference_input)

        start_time = time.time()
        if self.model_client:
            result = self.model_client.inference(inference_input)
            elapsed_time = time.time() - start_time
            self.get_logger().warning(f"Model Output: {result}")
            self.get_logger().warning(
                f"Approximate Inference time: {elapsed_time} seconds"
            )
            self.get_logger().warning(
                f"Max throughput: {1 / elapsed_time} frames per second"
            )
        elif self.local_model:
            result = self.local_model(inference_input)
            elapsed_time = time.time() - start_time
            self.get_logger().warning(f"Model Output: {result}")
            self.get_logger().warning(
                f"Approximate Inference time: {elapsed_time} seconds"
            )
            self.get_logger().warning(
                f"Max throughput: {1 / elapsed_time} frames per second"
            )
        else:
            result = "Component was run without a client. Did not execute warmup"

from typing import Any, Union, Optional, List, Dict
import queue
import threading
import numpy as np
import cv2

from ..clients.model_base import ModelClient
from ..config import VisionConfig
from ..ros import (
    Detections,
    FixedInput,
    Image,
    RGBD,
    Topic,
    Trackings,
    ROSImage,
    ROSCompressedImage,
)
from ..utils import validate_func_args, load_model
from ..utils.vision import LocalVisionModel
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
        component_name: str,
        **kwargs,
    ):
        self.config: VisionConfig = config or VisionConfig()
        self.allowed_inputs = {"Required": [[Image, RGBD]]}
        self.handled_outputs = [Detections, Trackings]

        self._images: List[Union[np.ndarray, ROSImage, ROSCompressedImage]] = []

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
                model_client._model._num_trackers = len(inputs)
        else:
            if not self.config.enable_local_classifier:
                raise TypeError(
                    "Vision component either requires a model client or enable_local_classifier needs to be set True in the VisionConfig."
                )

    def custom_on_configure(self):
        # configure parent component
        super().custom_on_configure()

        # create visualization thread if enabled
        if self.config.enable_visualization:
            self.queue = queue.Queue()
            self.stop_event = threading.Event()
            self.visualization_thread = threading.Thread(target=self._visualize)
            self.visualization_thread.start()

        # deploy local model if enabled
        if not self.model_client and self.config.enable_local_classifier:
            self.local_classifier = LocalVisionModel(
                model_path=load_model(
                    "local_classifier", self.config.local_classifier_model_path
                ),
                ncpu=self.config.ncpu_local_classifier,
                device=self.config.device_local_classifier,
            )

    def custom_on_deactivate(self):
        # if visualization is enabled, shutdown the thread
        if self.config.enable_visualization:
            if self.visualization_thread:
                self.stop_event.set()
                self.visualization_thread.join()
        # deactivate component
        super().custom_on_deactivate()

    def _visualize(self):
        """CV2 based visualization of infereance results"""
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
            tracked_objects = data["output"][0].get("tracked_points", [])

            for bbox in bounding_boxes:
                # Assuming bbox format: (x1, y1, x2, y2)
                cv2.rectangle(
                    image,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (0, 255, 0),
                    2,
                )

            for point_list in tracked_objects:
                # Each point_list is a list of points on one tracked object
                for point in point_list:
                    # Assuming point format: (x, y)
                    cv2.circle(
                        image,
                        (int(point[0]), int(point[1])),
                        radius=4,
                        color=(0, 0, 255),
                        thickness=-1,
                    )

            cv2.imshow(self.node_name, image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.get_logger().warning("User pressed 'q', stopping visualization.")
                break

        cv2.destroyAllWindows()

    def _create_input(self, *_, **kwargs) -> Optional[Dict[str, Any]]:
        """Create inference input for ObjectDetection models
        :param args:
        :param kwargs:
        :rtype: dict[str, Any]
        """
        self._images = []
        # set one image topic as query for event based trigger
        if trigger := kwargs.get("topic"):
            images = [self.trig_callbacks[trigger.name].get_output()]
            if msg := kwargs.get("msg"):
                self._images.append(msg)
        else:
            images = []

            for i in self.callbacks.values():
                if (item := i.get_output(clear_last=True)) is not None:
                    images.append(item)
                    if i.msg:
                        self._images.append(i.msg)  # Collect all images for publishing

        if not images:
            return None

        return {"images": images, **self.inference_params}

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
        if self.model_client:
            result = self._call_inference(inference_input, unpack=True)
        elif self.config.enable_local_classifier:
            result = self.local_classifier(
                inference_input,
                self.config.input_height,
                self.config.input_width,
                self.config.dataset_labels,
            )
        else:
            raise TypeError(
                "Vision component either requires a model client or enable_local_classifier needs to be set True in the VisionConfig. If latter was done, make sure no errors occured during initialization of the local classifier model."
            )

        # raise a fallback trigger via health status
        if result:
            # publish inference result
            self._publish(
                result,
                images=self._images,
                time_stamp=self.get_ros_time(),
            )
            if self.config.enable_visualization:
                result["images"] = inference_input["images"]
                self.queue.put_nowait(result)
        else:
            self.health_status.set_failure()

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

        start_time = time.time()
        if self.model_client:
            result = self.model_client.inference(inference_input)
        else:
            result = "Component was run without a client. Did not execute warmup"
        elapsed_time = time.time() - start_time

        self.get_logger().warning(f"Model Output: {result}")
        self.get_logger().warning(f"Approximate Inference time: {elapsed_time} seconds")
        self.get_logger().warning(
            f"Max throughput: {1 / elapsed_time} frames per second"
        )

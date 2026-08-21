import time
import json
from typing import Any, Union, Optional, List, Dict, Literal, MutableMapping

import numpy as np
from ..clients.db_base import DBClient
from ..clients.model_base import ModelClient
from ..config import MLLMConfig
from ..ros import (
    CameraInfo,
    FixedInput,
    Event,
    Image,
    String,
    StreamingString,
    Topic,
    DetectionsMultiSource,
    Detections,
    Detections3D,
    RGBD,
    PointsOfInterest,
    ComponentRunType,
    ROSImage,
    ROSCompressedImage,
    component_action,
)
from ..utils import validate_func_args
from ..utils.perception3d import resolve_lift_camera
from .depth_lift import DepthLiftMixin
from .llm import LLM


class MLLM(DepthLiftMixin, LLM):
    """
    This component utilizes multi-modal large language models (e.g. Llava) that can be used to process text and image data.

    :param inputs: The input topics or fixed inputs for the MLLM component.
        This should be a list of Topic objects or FixedInput instances, limited to String and Image types.
    :type inputs: list[Topic | FixedInput]
    :param outputs: The output topics for the MLLM component.
        This should be a list of Topic objects. String, Detections2D and PointsOfInterest2D types is handled automatically.
        With the "grounding" or "affordance" task, a Detections3D output additionally lifts the
        grounded boxes into metric space (see the `depth` and `camera_info` parameters), labeled
        with the query that grounded them.
    :type outputs: list[Topic]
    :param model_client: The model client for the MLLM component.
        This should be an instance of ModelClient. Optional if ``enable_local_model`` is set to True in the config.
    :type model_client: Optional[ModelClient]
    :param config: Optional configuration for the MLLM component.
        This should be an instance of MLLMConfig. If not provided, defaults to MLLMConfig().
    :type config: MLLMConfig
    :param trigger: The trigger value or topic for the MLLM component.
        This can be a single Topic object, a list of Topic objects, or a float value for a timed component. Defaults to 1.
    :type trigger: Union[Topic, list[Topic], float]
    :param depth: Depth image topic registered to the camera the VLM grounds on, used together with `camera_info` to lift grounded boxes into metric 3D boxes when a Detections3D output is given (requires the "grounding" or "affordance" task). The depth frame is latched when the picture is captured for inference, so the boxes measure the scene the VLM actually saw.
    :type depth: Optional[Topic]
    :param camera_info: Camera intrinsics topic of the stream `depth` was measured on. Required alongside `depth` for a Detections3D output.
    :type camera_info: Optional[Topic]
    :param component_name: The name of the MLLM component.
        This should be a string and defaults to "mllm_component".
    :type component_name: str

    Example usage:
    ```python
    text0 = Topic(name="text0", msg_type="String")
    image0 = Topic(name="image0", msg_type="Image")
    text0 = Topic(name="text1", msg_type="String")
    config = MLLMConfig()
    model = TransformersMLLM(name='idefics')
    model_client = ModelClient(model=model)
    mllm_component = MLLM(inputs=[text0, image0],
                          outputs=[text1],
                          model_client=model_client,
                          config=config,
                          component_name='mllm_component')
    ```

    Example usage with local model:
    ```python
    text0 = Topic(name="text0", msg_type="String")
    image0 = Topic(name="image0", msg_type="Image")
    text1 = Topic(name="text1", msg_type="String")
    config = MLLMConfig(enable_local_model=True)
    mllm_component = MLLM(inputs=[text0, image0],
                          outputs=[text1],
                          config=config,
                          component_name='local_vlm')
    ```
    """

    @validate_func_args
    def __init__(
        self,
        *,
        inputs: List[Union[Topic, FixedInput]],
        outputs: List[Topic],
        model_client: Optional[ModelClient] = None,
        config: Optional[MLLMConfig] = None,
        db_client: Optional[DBClient] = None,
        trigger: Union[Topic, List[Topic], float, Event] = 1.0,
        depth: Optional[Topic] = None,
        camera_info: Optional[Topic] = None,
        component_name: str,
        **kwargs,
    ):
        self.allowed_inputs = {
            "Required": [String, [Image, RGBD]],
            "Optional": [DetectionsMultiSource, Detections, CameraInfo],
        }

        config = config or MLLMConfig()

        if not model_client and not config.enable_local_model:
            raise RuntimeError(
                "MLLM/VLM component requires a model_client or enable_local_model=True in MLLMConfig."
            )

        depth = depth or config._depth_topic
        camera_info = camera_info or config._camera_info_topic
        config._depth_topic = depth
        config._camera_info_topic = camera_info
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
                f"MLLM was given CameraInfo topic(s) {stray_info} among its "
                "inputs. Intrinsics describe a camera rather than deliver "
                "pictures to reason on, so they are passed as "
                "`camera_info=Topic(...)`, as depth is passed as `depth=Topic(...)`."
            )
        self._aux_inputs = {t.name for t in (depth, camera_info) if t}

        # Asking for a Detections3D output turns 3D lifting on
        self._lift_to_3d = any(issubclass(t.msg_type, Detections3D) for t in outputs)
        self._lift_camera = None
        if self._lift_to_3d:
            # Lifting to 3d only works for bounding box outputs
            if config.task not in ("grounding", "affordance"):
                raise TypeError(
                    "MLLM was given a Detections3D output, which only the "
                    "'grounding' and 'affordance' tasks can produce boxes for. "
                    f"Set `task` in the MLLMConfig (currently {config.task!r})."
                )
            # The contract check names the picture stream the lift applies to
            self._lift_camera = resolve_lift_camera(
                inputs,
                depth,
                camera_info,
                frame=config.detections_frame,
                component="MLLM",
            )
            # Intrinsics and depth only get subscribed to when they feed a lift
            for topic in (depth, camera_info):
                if topic and all(t.name != topic.name for t in inputs):
                    inputs = [*inputs, topic]

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            model_client=model_client,
            config=config,
            db_client=db_client,
            trigger=trigger,
            component_name=component_name,
            allowed_inputs=self.allowed_inputs,
            **kwargs,
        )

        self.handled_outputs = [
            String,
            StreamingString,
            Detections,
            DetectionsMultiSource,
            PointsOfInterest,
            Detections3D,
        ]
        self._images: List[Union[np.ndarray, ROSImage, ROSCompressedImage]] = []

        # Initialize detector for 3D lift and its parse calibration
        self._init_lift_state()
        # For capturing grounding query to name 3D boxes
        self._lift_label: str = ""

    def custom_on_configure(self):
        # deploy local VLM if enabled
        if not self.model_client and self.config.enable_local_model:
            self._deploy_local_model()

        # configure the rest
        super().custom_on_configure()

        # Setup task
        self._task = self.config.task
        if self._task:
            # Initialize the topic type lists
            self._string_publishers: List = []
            self._poi_publishers: List = []
            self._detections_publishers: List = []
            self._detections3d_publishers: List = []

            # Loop through the list of topics and categorize them
            for topic in self.out_topics:
                if topic.msg_type in [String, StreamingString]:
                    self._string_publishers.append(topic.name)
                elif topic.msg_type is PointsOfInterest:
                    self._poi_publishers.append(topic.name)
                elif topic.msg_type in [Detections, DetectionsMultiSource]:
                    self._detections_publishers.append(topic.name)
                elif topic.msg_type is Detections3D:
                    self._detections3d_publishers.append(topic.name)
                else:
                    pass

    def _deploy_local_model(self):
        """Deploy local VLM model on demand."""
        if self.local_model is not None:
            return  # already deployed
        from ..utils.local_vlm import LocalVLM

        self.local_model = LocalVLM(
            model_path=self.config.local_model_path,
            device=self.config.device_local_model,
            ncpu=self.config.ncpu_local_model,
            model_options=self.config.local_model_options,
        )

    def _create_input(self, *_, **kwargs) -> Optional[Dict[str, Any]]:
        """Create inference input for MLLM models
        :param args:
        :param kwargs:
        :rtype: dict[str, Any]
        """
        self._images = []  # image msgs for publishing
        images = []  # image msg outputs as np arrays
        # The 3D latched msg and depth
        self._lift_msg = None
        self._lift_depth = None

        # context dict to gather all String inputs for use in system prompt
        context = {}
        # set mllm query as trigger
        query = self._extract_query_and_context(kwargs, context)
        if self._should_reset_chat(query):
            self.messages = []
            return None

        # aggregate all inputs that are available. Depth and camera_info
        # only feed the 3D lift, not the model
        for i in self.callbacks.values():
            if (
                i.input_topic.name in self._aux_inputs
                or (item := i.get_output()) is None
            ):
                continue
            msg = i.msg
            msg_type = i.input_topic.msg_type
            # set trigger equal to a topic with type String if trigger not found
            if msg_type == String:
                query = query or item
                context[i.input_topic.name] = item
            elif msg_type in [DetectionsMultiSource, Detections]:
                context[i.input_topic.name] = item
            # get images from image topics
            if issubclass(msg_type, (Image, RGBD)):
                images.append(item)
                if msg is not None:
                    self._images.append(msg)  # Collect all images for publishing
                    if i.input_topic.name == self._lift_camera:
                        self._lift_msg = msg
                        self._lift_depth = self._depth_snapshot()

        if not query or not images:
            return None

        # the query as given, before templates and RAG dress it up
        self._lift_label = query

        # get RAG results if enabled in config and if docs retrieved
        rag_result = self._handle_rag_query(query) if self.config.enable_rag else None

        # set system prompt template
        query = (
            self.component_prompt.render(context) if self.component_prompt else query
        )

        # get RAG results if enabled in config and if docs retreived
        query = f"{rag_result}\n{query}" if rag_result else query

        message = {"role": "user", "content": query}
        self._handle_chat_history(message)

        self.get_logger().debug(f"Input from component: {self.messages}")

        input = {
            "query": self.messages,
            "images": images,
            **self.inference_params,
        }

        # Add any tools, if registered
        if self.config._tool_descriptions:
            input["tools"] = self.config._tool_descriptions

        return input

    @validate_func_args
    def set_task(
        self,
        task: Literal["general", "pointing", "affordance", "trajectory", "grounding"],
    ) -> None:
        """Set a task for the MLLM component. This is useful when using a multimodal LLM model that has been trained on specific tasks. This method can be invoked as an action, in response to an event, to change the task at runtime.
            For an example checkout [RoboBrain2.0](https://github.com/FlagOpen/RoboBrain2.0), available on [RoboML](https://github.com/automatika-robotics/roboml).

        :param task: A task that is one of the following "general", "pointing", "affordance", "trajectory", "grounding".
        :type text: Literal
        """
        if task not in ["general", "pointing", "affordance", "trajectory", "grounding"]:
            raise ValueError(
                'Task value should be one of the following "general", "pointing", "affordance", "trajectory", "grounding"'
            )
        self._task = task
        self.config.task = task
        self.config.stream = False
        self.inference_params = self.config.get_inference_params()

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "describe",
                "description": (
                    "Use this method when asked to describe something in robot's "
                    "surroundings. Captures a frame from a camera topic and describe what "
                    "is visible in the image using the vision-language model. "
                    "Returns a text description to the caller, which can be used in a"
                    " subsequent tool call if required."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic_name": {
                            "type": "string",
                            "description": (
                                "Name of the image input topic to capture from. "
                                "Should be one of the component's image input topics."
                            ),
                        },
                        "query": {
                            "type": "string",
                            "description": (
                                "Question or instruction for the model about the image. "
                                "Defaults to 'Describe what you see in the image.'"
                            ),
                        },
                    },
                    "required": ["topic_name"],
                },
            },
        }
    )
    def describe(
        self,
        topic_name: str,
        query: str = "Describe what you see in the image.",
        timeout: float = 0.5,
    ) -> str:
        """Capture a frame from an image topic and describe it.

        Grabs the latest frame from the specified image input topic,
        runs VLM inference with the given query, and publishes the
        text result to the component's String output topics.

        :param topic_name: Name of the image input topic to capture from.
        :type topic_name: str
        :param query: Question or instruction about the image.
        :type query: str
        :param timeout: Seconds to wait for a frame. Defaults to 0.5.
        :type timeout: float
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        try:
            image = self._grab_frame(topic_name, timeout)
            if image is None:
                self.get_logger().error(
                    "Describe: could not get image from image topic."
                )
                raise Exception("Could not get image from image topic.")

            inference_input = {
                "query": [{"role": "user", "content": query}],
                "images": [image],
                **self.config._get_inference_params(),
            }

            result = self._call_inference(inference_input)
            if not result or not (output := result.get("output")):
                self.get_logger().error("Describe: inference returned no output.")
                raise Exception("Inference failed and returned no output.")

            # return text output to caller
            return json.dumps(output)

        except Exception as e:
            self.get_logger().error(f"Failed to describe: {e}")
            raise

    def _grab_frame(self, topic_name: str, timeout: float) -> Optional[np.ndarray]:
        """Grab a single frame from an image callback.

        :param topic_name: Name of the image input topic
        :param timeout: Seconds to wait for a frame
        :returns: Image as numpy array, or None on failure
        """
        trig_dict = getattr(self, "trig_callbacks", {})
        target_callback = trig_dict.get(topic_name) or self.callbacks.get(topic_name)
        if not target_callback:
            self.get_logger().error(
                f"Topic '{topic_name}' is not one of the component inputs."
            )
            return None

        # Check that this is an image topic
        if not issubclass(target_callback.input_topic.msg_type, (Image, RGBD)):
            self.get_logger().error(f"Topic '{topic_name}' is not an image topic.")
            return None

        # Save and swap callback to intercept a frame
        original_callback = target_callback._extra_callback
        original_get_processed = target_callback._get_processed
        frames = []

        def frame_interceptor(msg, topic, output=None):
            if output is not None and not frames:
                frames.append(output.copy())

        try:
            target_callback.on_callback_execute(frame_interceptor, get_processed=True)
            start_time = time.time()
            while (time.time() - start_time) < timeout and not frames:
                time.sleep(0.01)
        finally:
            if original_callback:
                target_callback.on_callback_execute(
                    original_callback, get_processed=original_get_processed
                )
            else:
                target_callback._extra_callback = None

        if not frames:
            self.get_logger().warning(
                f"Describe: timeout waiting for image on '{topic_name}'."
            )
            return None

        return frames[0]

    def _publish_task_specific_outputs(self, result: MutableMapping) -> None:
        """Publish outputs based on task type"""
        if self._task == "general":
            result["output"] = self._strip_think_tokens(result["output"])
            self.messages.append({"role": "assistant", "content": result["output"]})
            for pub_name in self._string_publishers:
                self.publishers_dict[pub_name].publish(
                    **result, time_stamp=self.get_ros_time()
                )
        elif self._task == "pointing":
            for pub_name in self._poi_publishers:
                self.publishers_dict[pub_name].publish(
                    **result,
                    image=self._images[0],  # POI msg takes only one image
                    time_stamp=self.get_ros_time(),
                )
        elif self._task in ("grounding", "affordance"):
            boxes = {"bboxes": result["output"], "labels": [], "scores": []}
            for pub_name in self._detections_publishers:
                publisher = self.publishers_dict[pub_name]
                # NOTE: The model grounds against the image set as a whole, so there
                # is one set of boxes: a multi source topic takes it as a list
                # of one, a single source topic takes it on its own
                multi = issubclass(
                    publisher.output_topic.msg_type, DetectionsMultiSource
                )
                publisher.publish(
                    [boxes] if multi else boxes,
                    images=self._images if multi else next(iter(self._images), None),
                    time_stamp=self.get_ros_time(),
                )
            self._publish_grounded_3d(result["output"])
        elif self._task == "trajectory":
            for pub_name in self._poi_publishers:
                self.publishers_dict[pub_name].publish(
                    **result,
                    image=self._images[0],  # POI msg takes only one image
                    time_stamp=self.get_ros_time(),
                )

    def _publish_grounded_3d(self, pixels: List) -> None:
        """Lift the grounded boxes into metric space and publish Detections3D."""

        if not self._detections3d_publishers:
            return
        from ..utils.perception3d import (
            boxes_from_detections,
            detections_to_message_fields,
        )

        if self._lift_msg is None:
            self._warn_once(
                "no_lift_frame",
                f"No picture was captured on '{self._lift_camera}', so the "
                "grounded boxes cannot be placed in space.",
            )
            return

        if not pixels:
            fields = detections_to_message_fields([])
        else:
            detector, depth_mm, camera_position = self._depth_detector()
            if detector is None:
                return
            # the color image, which an RGBD frame carries nested
            color = getattr(self._lift_msg, "rgb", self._lift_msg)
            lifted = [
                box
                for box in boxes_from_detections(
                    detector,
                    depth_mm,
                    pixels,
                    image_size=(color.width, color.height),
                    depth_range=(self.config.min_depth, self.config.max_depth),
                    camera_position=camera_position,
                )
                # A box built from a handful of depth pixels is not trustworthy
                if box.validity >= self.config.min_depth_validity
            ]
            # Every grounded box answers the same query, thus the same name
            fields = detections_to_message_fields(
                lifted,
                labels=[self._lift_label] * len(pixels),
                boxes_2d=pixels,
            )

        for pub_name in self._detections3d_publishers:
            self.publishers_dict[pub_name].publish(
                fields["output"],
                **{k: v for k, v in fields.items() if k != "output"},
                frame_id=self.config.detections_frame,
                time_stamp=self.get_ros_time(),
            )

    def _execution_step(self, *args, **kwargs):
        """_execution_step.

        :param args:
        :param kwargs:
        """
        if not self._task:
            super()._execution_step(*args, **kwargs)
            return

        # If a task has been specified then handle it here
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
        unpack = True if self._task != "general" else False
        result = self._call_inference(inference_input, unpack=unpack)

        # Publish results to output topics in accordance with the tasks
        if result:
            self._publish_task_specific_outputs(result)
            if result.get("thinking"):
                self.get_logger().info(f"<think>{result['thinking']}</think>")

    def _warmup(self):
        """Warm up and stat check"""
        import time
        from pathlib import Path
        import cv2

        image = cv2.imread(str(Path(__file__).parents[1] / Path("resources/test.jpeg")))

        message = {"role": "user", "content": "What do you see?"}
        inference_input = {
            "query": [message],
            "images": [image],
            **self.inference_params,
        }

        # Run inference once to warm up and once to measure time
        if self.model_client:
            self.model_client.inference(inference_input)
        elif hasattr(self, "local_model"):
            self.local_model(inference_input)

        inference_input = {
            "query": [message],
            "images": [image],
            **self.config._get_inference_params(),
        }
        start_time = time.time()
        if self.model_client:
            result = self.model_client.inference(inference_input)
        elif hasattr(self, "local_model"):
            result = self.local_model(inference_input)
        else:
            result = None
        elapsed_time = time.time() - start_time

        if result:
            self.get_logger().warning(f"Model Output: {result['output']}")
            self.get_logger().warning(
                f"Approximate Inference time: {elapsed_time} seconds"
            )
        else:
            self.get_logger().error("Model inference failed during warmup.")


# Alias
VLM = MLLM

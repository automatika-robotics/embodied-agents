from typing import Any, Union, Optional, List, Dict

from ..clients.db_base import DBClient
from ..clients.model_base import ModelClient
from ..config import MLLMConfig
from ..ros import FixedInput, Image, String, Topic, Detections
from ..utils import validate_func_args
from .llm import LLM


class MLLM(LLM):
    """
    This component utilizes multi-modal large language models (e.g. Llava) that can be used to process text and image data.

    :param inputs: The input topics or fixed inputs for the MLLM component.
        This should be a list of Topic objects or FixedInput instances, limited to String and Image types.
    :type inputs: list[Topic | FixedInput]
    :param outputs: The output topics for the MLLM component.
        This should be a list of Topic objects. String type is handled automatically.
    :type outputs: list[Topic]
    :param model_client: The model client for the MLLM component.
        This should be an instance of ModelClient.
    :type model_client: ModelClient
    :param config: Optional configuration for the MLLM component.
        This should be an instance of MLLMConfig. If not provided, defaults to MLLMConfig().
    :type config: MLLMConfig
    :param trigger: The trigger value or topic for the MLLM component.
        This can be a single Topic object, a list of Topic objects, or a float value for a timed component. Defaults to 1.
    :type trigger: Union[Topic, list[Topic], float]
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
    """

    @validate_func_args
    def __init__(
        self,
        *,
        inputs: List[Union[Topic, FixedInput]],
        outputs: List[Topic],
        model_client: ModelClient,
        config: Optional[MLLMConfig] = None,
        db_client: Optional[DBClient] = None,
        trigger: Union[Topic, List[Topic], float] = 1.0,
        component_name: str,
        **kwargs,
    ):
        self.allowed_inputs = {"Required": [String, Image], "Optional": [Detections]}

        config = config or MLLMConfig()

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

    def _create_input(self, *_, **kwargs) -> Optional[Dict[str, Any]]:
        """Create inference input for MLLM models
        :param args:
        :param kwargs:
        :rtype: dict[str, Any]
        """
        images = []
        # context dict to gather all String inputs for use in system prompt
        context = {}
        # set mllm query as trigger
        if trigger := kwargs.get("topic"):
            query = self.trig_callbacks[trigger.name].get_output()
            context[trigger.name] = query

            # handle chat reset
            if (
                self.config.chat_history
                and query.strip().lower() == self.config.history_reset_phrase
            ):
                self.messages = []
                return None

        else:
            query = None

        # aggregate all inputs that are available
        for i in self.callbacks.values():
            if (item := i.get_output()) is not None:
                # set trigger equal to a topic with type String if trigger not found
                if i.input_topic.msg_type is String:
                    if not query:
                        query = item
                    context[i.input_topic.name] = item
                elif i.input_topic.msg_type is Detections:
                    context[i.input_topic.name] = item
                # get images from image topics
                if issubclass(i.input_topic.msg_type, Image):
                    images.append(item)

        if not query or not images:
            return None

        # get RAG results if enabled in config and if docs retreived
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
        self.model_client.inference(inference_input)

        inference_input = {
            "query": [message],
            "images": [image],
            **self.config._get_inference_params(),
        }
        start_time = time.time()
        result = self.model_client.inference(inference_input)
        elapsed_time = time.time() - start_time

        self.get_logger().warning(f"Model Output: {result['output']}")
        self.get_logger().warning(f"Approximate Inference time: {elapsed_time} seconds")

from typing import Any, Dict, Generator, Optional, Union, List
import os
import json

import numpy as np
import httpx

from .model_base import ModelClient
from ..models import LLM, OllamaModel
from ..utils import encode_img_base64, validate_func_args


__all__ = ["GenericHTTPClient"]


class GenericHTTPClient(ModelClient):
    """
    A generic client for interacting with OpenAI-compatible APIs, including vLLM, ms-swift, lmdeploy, Google Gemini etc. This client works with LLM multimodal LLM models and supports both standard and streaming responses. It is designed to be compatible with any API that follows the OpenAI standard.
    """

    @validate_func_args
    def __init__(
        self,
        model: Union[LLM, Dict],
        host: str = "127.0.0.1",
        port: Optional[int] = 8000,
        inference_timeout: int = 30,
        api_key: Optional[str] = None,
        logging_level: str = "info",
        **kwargs,
    ):
        """
        Initializes the Client.
        :param model: The model to be used for inference.
        :type model: Union[Model, Dict]
        :param host: The hostname of the API server.
        :type host: str
        :param port: The port of the API server.
        :type port: Optional[int]
        :param inference_timeout: The timeout for inference requests.
        :type inference_timeout: int
        :param api_key: The API key for authentication. If not provided, it will be
                        retrieved from the OPENAI_API_KEY environment variable.
        :type api_key: Optional[str]
        :param logging_level: The logging level.
        :type logging_level: str
        """
        if isinstance(model, OllamaModel):
            raise TypeError(
                "An ollama model cannot be passed to a RoboML client. Please use the OllamaClient"
            )

        super().__init__(
            model=model,
            host=host,
            port=port,
            inference_timeout=inference_timeout,
            init_on_activation=False,
            logging_level=logging_level,
            **kwargs,
        )

        # try to get it from the environment variable otherwise default to empty string
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        header = {} if not self.api_key else {"Authorization": f"Bearer {self.api_key}"}

        self.url = f"http://{self.host}:{self.port}"

        # Create a synchronous httpx client
        self.client = httpx.Client(
            base_url=self.url,
            timeout=self.inference_timeout,
            headers=header,
        )

    def _check_connection(self) -> None:
        """
        Checks the connection to the API server by making a simple request.
        """
        self.logger.info("Checking connection with OpenAI-compatible API...")
        try:
            # Make a request to the models endpoint to verify the connection
            self.client.get("/v1/models").raise_for_status()
        except Exception as e:
            self.__handle_exceptions(e)
            raise

    def _initialize(self) -> None:
        """Initializes the client.

        For this client, model initialization cannot be handled by the client.
        """
        pass

    def _inference(
        self, inference_input: Dict[str, Any]
    ) -> Optional[Dict[str, Union[str, Generator]]]:
        """Performs inference using the specified model and input.

        :param inference_input: The input for the inference. This should be a
                                dictionary containing the messages and other
                                parameters for the API request.
        :type inference_input: Dict[str, Any]
        :return: A dictionary containing the output of the inference. If
                 streaming is enabled, the output will be a generator.
        :rtype: Optional[Dict[str, Union[str, Generator]]]
        :raises httpx.RequestError: If there is an error during the inference request.
        :raises httpx.HTTPStatusError: If the API returns an error during inference.
        """
        # Construct the payload for the API request
        inference_input = self.__handle_images(inference_input)

        payload = {
            "model": self.model_init_params["checkpoint"],
            "messages": inference_input.pop("query"),
            **inference_input,
        }

        try:
            # If streaming is enabled, return a generator
            if payload["stream"]:

                def stream_generator():
                    with self.client.stream(
                        "POST", "/v1/chat/completions", json=payload
                    ) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if line.startswith("data:"):
                                data_str = line.removeprefix("data: ").strip()
                                if data_str == "[DONE]":
                                    break
                                if data_str:
                                    try:
                                        yield json.loads(data_str)
                                    except json.JSONDecodeError:
                                        self.logger.warning(
                                            f"Failed to decode JSON from stream: {data_str}"
                                        )

                return {"output": stream_generator()}
            # Otherwise, make a standard request
            else:
                response = self.client.post(
                    "/v1/chat/completions", json=payload
                ).raise_for_status()
                r_json = response.json()

                self.logger.debug(f"Response from API: {r_json}")

                if not r_json.get("choices"):
                    self.logger.debug(
                        f"API Response does not contain any model output: {r_json}"
                    )
                    return

                model_resp = {}
                # if tool calls exist
                if tool_calls := r_json["choices"][0]["message"].get("tool_calls"):
                    model_resp["tool_calls"] = tool_calls

                model_resp["output"] = r_json["choices"][0]["message"].get(
                    "content", ""
                )
                return model_resp

        except Exception as e:
            self.__handle_exceptions(e)

    def _deinitialize(self) -> None:
        """
        Deinitializes the client by closing the httpx client.
        """
        self.logger.info("Deinitializing GenericHTTPClient...")
        self.client.close()

    def __handle_images(self, inference_input: Dict[str, Any]) -> Dict[str, Any]:
        """Handles images in multimodal input"""
        if images := inference_input.get("images"):
            b64_images = [encode_img_base64(img) for img in images]
            inference_input.pop("images")  # Remove images from inference_input
            # Add images to last message in query
            if inference_input["query"][-1].get("role") == "user":
                last_message = inference_input["query"][-1]
                content_parts = []

                # Add the original text prompt as the first part
                content_parts.append({"type": "text", "text": last_message["content"]})

                # Add each image as a subsequent part
                for img_b64 in b64_images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    })

                # Replace the original content with the new list of parts
                inference_input["query"][-1]["content"] = content_parts
                return inference_input
            else:
                self.logger.warning(
                    "Images provided, but no final user message found to attach them to. Ignoring images."
                )
        return inference_input

    def __handle_exceptions(self, excep: Exception) -> None:
        """__handle_exceptions.

        :param excep:
        :type excep: Exception
        :rtype: None
        """
        if isinstance(excep, httpx.RequestError):
            self.logger.error(
                f"{excep}. API inaccessible. Might not be running. Make sure remote is correctly configured."
            )
        elif isinstance(excep, httpx.TimeoutException):
            self.logger.error(
                f"{excep}. Request to API server timed out. Make sure the server is configured correctly."
            )
        elif isinstance(excep, httpx.HTTPStatusError):
            try:
                excep_json = excep.response.json()
                self.logger.error(
                    f"API server returned an invalid status code. Error: {excep_json}"
                )
            except Exception:
                self.logger.error(
                    f"API server returned an invalid status code. Error: {excep}"
                )
        else:
            self.logger.error(str(excep))

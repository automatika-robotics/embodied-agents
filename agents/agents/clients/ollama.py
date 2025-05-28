from types import GeneratorType
from typing import Any, Optional, Dict, Union

import httpx

from ..models import LLM
from ..utils import encode_arr_base64
from .model_base import ModelClient

__all__ = ["OllamaClient"]


class OllamaClient(ModelClient):
    """An HTTP client for interaction with ML models served on ollama"""

    def __init__(
        self,
        model: Union[LLM, Dict],
        host: str = "127.0.0.1",
        port: int = 11434,
        inference_timeout: int = 30,
        init_on_activation: bool = True,
        logging_level: str = "info",
        **kwargs,
    ):
        try:
            from ollama import Client

            self.client = Client(host=f"{host}:{port}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "In order to use the OllamaClient, you need ollama-python package installed. You can install it with 'pip install ollama'"
            ) from e

        if not isinstance(model, LLM):
            raise TypeError(
                "OllamaClient can only be used with LLM and MLLM components"
            )

        model._set_ollama_checkpoint()

        super().__init__(
            model=model,
            host=host,
            port=port,
            inference_timeout=inference_timeout,
            init_on_activation=init_on_activation,
            logging_level=logging_level,
            **kwargs,
        )
        self._check_connection()

    def _check_connection(self) -> None:
        """Check if the platfrom is being served on specified IP and port"""
        # Ping remote server to check connection
        self.logger.info("Checking connection with remote_host Ollama")
        try:
            httpx.get(f"http://{self.host}:{self.port}").raise_for_status()
        except Exception as e:
            self.logger.error(str(e))
            raise

    def _initialize(self) -> None:
        """
        Initialize the model on platform using the paramters provided in the model specification class
        """
        self.logger.info(f"Initializing {self.model_name} on ollama")
        try:
            # set timeout on underlying httpx client
            self.client._client.timeout = self.init_timeout
            r = self.client.pull(self.model_init_params["checkpoint"])
            if r.get("status") != "success":  # type: ignore
                raise Exception(
                    f"Could not pull model {self.model_init_params['checkpoint']}"
                )
            # load model in memory with empty request
            self.client.generate(
                model=self.model_init_params["checkpoint"], keep_alive=10
            )
            self.logger.info(f"{self.model_name} model initialized")
        except Exception as e:
            self.logger.error(str(e))

    def _inference(self, inference_input: Dict[str, Any]) -> Optional[Dict]:
        """Call inference on the model using data and inference parameters from the component"""
        # create input
        input = {
            "model": self.model_init_params["checkpoint"],
            "messages": inference_input.pop("query"),
            "stream": inference_input.pop("stream"),
        }

        # make images part of the latest message in message list
        if images := inference_input.get("images"):
            input["messages"][-1]["images"] = [encode_arr_base64(img) for img in images]
            inference_input.pop("images")

        # Add tools as part of input, if available
        if tools := inference_input.get("tools"):
            input["tools"] = tools
            inference_input.pop("tools")

        # ollama uses num_predict for max_new_tokens
        if max_new_tokens := inference_input.get("max_new_tokens"):
            inference_input["num_predict"] = max_new_tokens
            inference_input.pop("max_new_tokens")
        input["options"] = inference_input

        # call inference method
        try:
            # set timeout on underlying httpx client
            self.client._client.timeout = self.inference_timeout
            ollama_result = self.client.chat(**input)
        except Exception as e:
            self.logger.error(str(e))
            return

        if isinstance(ollama_result, GeneratorType):
            input["output"] = ollama_result
            return input

        self.logger.debug(str(ollama_result))

        # make result part of the input
        if output := ollama_result["message"].get("content"):
            input["output"] = output  # type: ignore
            # if tool calls exist
            if tool_calls := ollama_result["message"].get("tool_calls"):  # type: ignore
                input["tool_calls"] = tool_calls
            return input
        else:
            # if tool calls exist
            if tool_calls := ollama_result["message"].get("tool_calls"):  # type: ignore
                input["output"] = ""  # Add empty output for tool calls
                input["tool_calls"] = tool_calls
                return input

            # no output or tool calls
            self.logger.debug("Output not received")

    def _deinitialize(self):
        """Deinitialize the model on the platform"""

        self.logger.error(f"Deinitializing {self.model_name} model on ollama")
        try:
            self.client.generate(
                model=self.model_init_params["checkpoint"], keep_alive=0
            )
        except Exception as e:
            self.logger.error(str(e))

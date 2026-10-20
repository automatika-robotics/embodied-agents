import threading
from typing import Any, Optional, Dict, Union, List, Generator, Tuple

import httpx

from ..models import OllamaModel
from ..utils import encode_img_base64, tls_verify
from .model_base import ModelClient

__all__ = ["OllamaClient"]

# Decision model warm up query
DECISION_WARMUP = {
    "state": "Warming up.",
    "questions": {"warmup": {"type": "noul", "instructions": "Is this a test?"}},
}


class OllamaClient(ModelClient):
    """An HTTP client for interaction with ML models served on ollama."""

    def __init__(
        self,
        model: Union[OllamaModel, Dict],
        host: str = "127.0.0.1",
        port: int = 11434,
        inference_timeout: int = 30,
        init_on_activation: bool = True,
        logging_level: str = "info",
        ca_cert: Optional[str] = None,
        **kwargs,
    ):
        try:
            from ollama import Client
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "In order to use the OllamaClient, you need ollama-python package installed. You can install it with 'pip install ollama'"
            ) from e

        if not isinstance(model, OllamaModel) and model["model_type"] != "OllamaModel":
            raise TypeError("OllamaClient can only be used with an OllamaModel")

        super().__init__(
            model=model,
            host=host,
            port=port,
            inference_timeout=inference_timeout,
            init_on_activation=init_on_activation,
            logging_level=logging_level,
            ca_cert=ca_cert,
            **kwargs,
        )
        self.client = Client(host=self._build_url(), verify=tls_verify(self.ca_cert))
        self._check_connection()
        self._capabilities: Optional[List[str]] = None

        # A pull of a tag that is not installed, run in the background
        self._pull: Optional[threading.Thread] = None
        self._probe_capabilities()

    @property
    def supports_tool_calls(self) -> bool:
        """
        Ollama chat models support tool calling, decision models do not.
        Does not wait for a pull: a model still being pulled is taken for
        a chat model.

        :rtype: bool
        """
        return self._capabilities is None or "decision" not in self._capabilities

    @property
    def supports_decisions(self) -> bool:
        """
        Whether the model is a decision model, by its Ollama capabilities.
        Waits for a pull in progress.

        :rtype: bool
        """
        return "decision" in self._model_capabilities()

    def _probe_capabilities(self, pull: bool = True) -> None:
        """Read the model's capabilities from Ollama. A tag that is not
        installed yet is pulled in the background"""
        from ollama import ResponseError

        checkpoint = self.model_init_params["checkpoint"]
        try:
            self._capabilities = self.client.show(checkpoint).capabilities or []
        except ResponseError as e:
            if e.status_code != 404:
                self.logger.debug(
                    f"Could not read the capabilities of {checkpoint}: {e}"
                )
            elif pull:
                self.logger.info(
                    f"{checkpoint} is not pulled locally in Ollama, pulling it in the background"
                )
                self._pull = threading.Thread(target=self._pull_model, daemon=True)
                self._pull.start()
            else:
                self.logger.error(
                    f"{checkpoint} does not exist locally in Ollama and could not be pulled"
                )
        except Exception as e:
            self.logger.debug(f"Could not read the capabilities of {checkpoint}: {e}")

    def _pull_model(self) -> None:
        """Pull the model. Logs the stages of the pull, and the progress of
        each layer once per tenth."""
        checkpoint = self.model_init_params["checkpoint"]
        stage, decile = None, 0
        try:
            for progress in self.client.pull(checkpoint, stream=True):
                completed, total = (
                    progress.get("completed") or 0,
                    progress.get("total") or 0,
                )
                if progress.get("status") != stage:
                    stage, decile = progress.get("status"), 0
                    # show per layer progress
                    if not total:
                        self.logger.info(f"Pulling {checkpoint}: {stage}")
                # do not log below a display unit
                if total >= 1e6 and int(10 * completed / total) > decile:
                    decile = int(10 * completed / total)
                    self.logger.info(
                        f"Pulling {checkpoint}: {completed / 1e6:.0f} of {total / 1e6:.0f} MB "
                        f"({10 * decile}%)"
                    )
        except Exception as e:
            self.logger.error(f"Could not pull model {checkpoint}: {e}")

    def _model_capabilities(self) -> List[str]:
        """The model's capabilities, after any pull has finished"""
        if self._capabilities is None:
            if self._pull is None:
                self._probe_capabilities()
            if self._pull is not None:
                if self._pull.is_alive():
                    self.logger.info(
                        f"Waiting for the pull of {self.model_init_params['checkpoint']} to finish"
                    )
                    self._pull.join()
                self._probe_capabilities(pull=False)
        return self._capabilities or []

    def _check_connection(self) -> None:
        """Check if the platfrom is being served on specified IP and port"""
        # Ping remote server to check connection
        self.logger.info("Checking connection with remote_host Ollama")
        try:
            httpx.get(self._build_url()).raise_for_status()
        except Exception:
            self.logger.error(
                f"""Failed to connect to Ollama server at {self.host}:{self.port}

                Make sure an Ollama instance is running on the given url by executing the following command:

                `export OLLAMA_HOST={self.host}:{self.port}  # if not using default`
                `ollama serve`

                To install ollama, follow instructions on https://ollama.com/download
                """
            )
            raise

    @property
    def context_length(self) -> Optional[int]:
        """The context length the model is served with: ``num_ctx`` from the
        model options when set, since requests carry it, else what Ollama
        reports for the loaded model, else the length the model was trained
        with. None when nothing can be read"""
        if num_ctx := (self.model_init_params.get("options") or {}).get("num_ctx"):
            return num_ctx
        served, trained = self._context_lengths()
        return served or trained

    def _context_lengths(self) -> Tuple[Optional[int], Optional[int]]:
        """The context length the model is served with, from Ollama's process
        list, and the one it was trained with, from its model info."""
        checkpoint = self.model_init_params["checkpoint"]
        served = trained = None
        try:
            wanted = checkpoint.removesuffix(":latest")
            for running in self.client.ps().models:
                names = {running.model, running.name}
                if wanted in {name.removesuffix(":latest") for name in names}:
                    served = running.context_length
            info = self.client.show(checkpoint).modelinfo or {}
            trained = next(
                (v for k, v in info.items() if k.endswith(".context_length")), None
            )
        except Exception as e:
            self.logger.debug(f"Could not read the context length of {checkpoint}: {e}")
        return served, trained

    def _warn_on_a_small_context(self) -> None:
        """Ollama serves a model with a few thousand tokens of context unless
        told otherwise, whatever the model supports, and truncates the prompt
        silently past it"""
        served, trained = self._context_lengths()
        if served and trained and served < trained:
            self.logger.warning(
                f"{self.model_init_params['checkpoint']} is served with a context "
                f"of {served} tokens of the {trained} it supports. Set num_ctx in "
                "the OllamaModel options to use more."
            )

    def _is_embedding_model(self) -> bool:
        """Detect whether the served model is an embedding model: the name
        ChromaClient gives its embedding model, or the embedding capability
        reported by Ollama."""
        if self.model_name == "internal_ollama_embeddings":
            return True
        return "embedding" in self._model_capabilities()

    def _initialize(self) -> None:
        """
        Initialize the model on platform using the paramters provided in the model specification class.
        """
        self.logger.info(f"Initializing {self.model_name} on ollama")
        checkpoint = self.model_init_params["checkpoint"]
        try:
            # set timeout on underlying httpx client
            self.client._client.timeout = self.init_timeout
            # Check capabilities, will also pull a model that doesnt exist
            self._capabilities = self._model_capabilities()
            # Cache capability check so _deinitialize can use it too
            self._is_embedding = self._is_embedding_model()
            # load embedding model in memory with an empty request
            if self._is_embedding:
                self.client.embed(
                    model=self.model_init_params["checkpoint"],
                    input="",
                    keep_alive=10,
                )
            # load a decision model in memory with a stub request
            elif self.supports_decisions:
                self.client.systemone(
                    model=checkpoint, keep_alive=10, **DECISION_WARMUP
                )
            # load a text generation model in memory
            else:
                self.client.generate(
                    model=self.model_init_params["checkpoint"], keep_alive=10
                )
        except Exception as e:
            self.logger.error(
                f"Failed to initialize model {self.model_init_params['checkpoint']} on Ollama, please ensure that the checkpoint name is correct. Received the following error: {e}"
            )
            raise
        self.logger.info(
            f"{self.model_name} with {self.model_init_params['checkpoint']} model initialized"
        )
        if not self._is_embedding and not self.supports_decisions:
            self._warn_on_a_small_context()

    def _inference(
        self, inference_input: Dict[str, Any]
    ) -> Optional[Dict[str, Union[str, Generator]]]:
        """Call inference on the model using data and inference parameters from the component"""
        # decision model inference
        if self.supports_decisions:
            return self._decide(inference_input)
        # create input
        input = {
            "model": self.model_init_params["checkpoint"],
            "messages": inference_input.pop("query"),
            "stream": inference_input.pop("stream"),
        }

        # Encode the images the chat history kept on earlier messages, and make
        # the images of this call part of the latest message
        messages = [
            {**message, "images": [encode_img_base64(img) for img in message["images"]]}
            if message.get("images")
            else message
            for message in input["messages"]
        ]
        if images := inference_input.pop("images", None):
            messages[-1] = {
                **messages[-1],
                "images": messages[-1].get("images", [])
                + [encode_img_base64(img) for img in images],
            }
        input["messages"] = messages

        # Add tools as part of input, if available
        if tools := inference_input.get("tools"):
            input["tools"] = tools
            inference_input.pop("tools")

        # ollama uses num_predict for max_new_tokens
        inference_input["num_predict"] = inference_input.pop("max_new_tokens")

        # merge any options specified during OllamaModel definition
        input["options"] = (
            {**self.model_init_params["options"], **inference_input}
            if self.model_init_params.get("options")
            else inference_input
        )
        # pass the thinking option
        if (think := self.model_init_params.get("think")) is not None:
            input["think"] = think

        self.logger.debug(f"Sending to ollama server: {input}")

        # call inference method
        try:
            # set timeout on underlying httpx client
            self.client._client.timeout = self.inference_timeout
            ollama_result = self.client.chat(**input)
        except Exception as e:
            self.logger.error(str(e))
            return

        if isinstance(ollama_result, Generator):
            input["output"] = ollama_result
            return input

        if not ollama_result.get("message"):
            self.logger.debug(
                f"Ollama Response does not contain any model output: {ollama_result}"
            )
            return

        # if tool calls exist, convert client return pydantic models to plain dicts
        if tool_calls := ollama_result["message"].get("tool_calls"):  # type: ignore
            input["tool_calls"] = [
                call.model_dump() if hasattr(call, "model_dump") else call
                for call in tool_calls
            ]

        input["output"] = ollama_result["message"].get("content", "")
        return input

    def _decide(self, inference_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Ask a decision model the typed questions about a state, and return
        the answers keyed by question id"""
        request = {
            "model": self.model_init_params["checkpoint"],
            "state": inference_input["state"],
            "questions": inference_input["questions"],
        }
        if images := inference_input.get("images"):
            request["images"] = [encode_img_base64(img) for img in images]
        try:
            self.client._client.timeout = self.inference_timeout
            response = self.client.systemone(**request)
        except Exception as e:
            self.logger.error(str(e))
            return
        return {
            "output": {
                question_id: answer.model_dump()
                for question_id, answer in response.answers.items()
            }
        }

    def _embed(
        self, input: Union[str, List[str]], truncate: bool = False
    ) -> Optional[List[List]]:
        # create input
        embedding_input = {
            "model": self.model_init_params["checkpoint"],
            "input": input,
            "truncate": truncate,
        }

        # call embedding method
        try:
            # set timeout on underlying httpx client
            self.client._client.timeout = self.inference_timeout
            ollama_result = self.client.embed(**embedding_input)
        except Exception as e:
            self.logger.error(str(e))
            return

        self.logger.debug(
            f"Created embeddings of length: {len(ollama_result['embeddings'])}"
        )

        # make result part of the input
        if not (result := ollama_result.get("embeddings")) or not result:
            self.logger.error("Embeddings not generated")
            return

        return ollama_result["embeddings"]

    def _deinitialize(self):
        """Deinitialize the model on the platform"""

        self.logger.info(f"Deinitializing {self.model_name} model on ollama")
        try:
            if getattr(self, "_is_embedding", False):
                self.client.embed(
                    model=self.model_init_params["checkpoint"],
                    input="",
                    keep_alive=0,
                )
            elif self.supports_decisions:
                self.client.systemone(
                    model=self.model_init_params["checkpoint"],
                    keep_alive=0,
                    **DECISION_WARMUP,
                )
            else:
                self.client.generate(
                    model=self.model_init_params["checkpoint"], keep_alive=0
                )
        except Exception as e:
            self.logger.error(str(e))

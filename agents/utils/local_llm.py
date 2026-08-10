import inspect
import json
import re
import threading
from typing import Dict, Generator, List, Optional, Tuple, Union

# Reserved keys in model_options that are not llama_cpp.Llama parameters
_FILENAME_KEY = "filename"

# Qwen/Hermes-style inline tool call emitted as text
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class LocalLLM:
    """Local LLM inference using llama-cpp-python.

    :param model_path: HuggingFace repository ID for a GGUF model
        (e.g. ``Qwen/Qwen3-0.6B-GGUF``) or a local path to a ``.gguf`` file.
    :param device: Device to run on ('cpu' or 'cuda')
    :param ncpu: Number of CPU threads
    :param model_options: Additional keyword options for ``llama_cpp.Llama``,
        validated against its signature (e.g. ``n_ctx``, ``n_batch``,
        ``flash_attn``, ``chat_format``). The reserved key ``filename``
        selects the GGUF file when a repository ships several quantizations
        (e.g. ``"*q4_k_m*.gguf"``).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        ncpu: int = 1,
        model_options: Optional[Dict] = None,
    ):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "Local LLM model deployment requires llama-cpp-python. "
                "Install it with: pip install llama-cpp-python\n"
                "For NVIDIA GPUs build with: "
                'CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python\n'
            ) from e

        self.device = device
        self.ncpu = ncpu

        options = dict(model_options or {})
        filename = options.pop(_FILENAME_KEY, "*.gguf")

        # Validate user options against the Llama signature
        llama_params = inspect.signature(Llama.__init__).parameters
        for key in options:
            if key not in llama_params:
                valid = sorted(
                    p for p in llama_params if p not in ("self", "kwargs")
                )
                raise ValueError(
                    f"Unknown local_model_options key '{key}' for llama-cpp. "
                    f"Valid options: {valid}. Reserved: '{_FILENAME_KEY}'."
                )

        kwargs = {
            "n_gpu_layers": -1 if device == "cuda" else 0,
            "n_threads": ncpu,
            # NOTE: 0 = use the model's trained context length. llama-cpp's own
            # default (512) overflows immediately with chat history, system
            # prompts and tool descriptions
            "n_ctx": 0,
            "verbose": False,
            **options,
        }

        if model_path.endswith(".gguf"):
            self.llm = Llama(model_path=model_path, **kwargs)
        else:
            self.llm = Llama.from_pretrained(
                repo_id=model_path, filename=filename, **kwargs
            )

        # NOTE: llama-cpp contexts are stateful and not thread-safe. Concurrent
        # calls corrupt the shared batch/KV cache and abort the process. We will
        # use thread locks for each call.
        self._lock = threading.Lock()

    def __call__(
        self, inference_input: Dict, stream=False
    ) -> Union[Dict, Generator[str, None, None]]:
        """Run inference and return complete response.

        :param inference_input: Dict with 'query' (messages list) and optional
            'temperature', 'max_new_tokens', 'tools'
        :returns: Dict with 'output' (str) and optionally 'tool_calls'
        """
        kwargs = {
            "messages": inference_input["query"],
            "stream": stream,
        }
        if temperature := inference_input.get("temperature"):
            kwargs["temperature"] = temperature
        if max_new_tokens := inference_input.get("max_new_tokens"):
            kwargs["max_tokens"] = max_new_tokens

        if tools := inference_input.get("tools"):
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if stream:
            return {"output": self._stream_tokens(kwargs)}

        with self._lock:
            response = self.llm.create_chat_completion(**kwargs)

        choice = response["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls")

        # parse tool calls (Qwen/Hermes style)
        if tools and not tool_calls:
            content, tool_calls = self._extract_inline_tool_calls(content)

        result = {"output": content}

        if tool_calls:
            result["tool_calls"] = [
                {
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": json.loads(tc["function"]["arguments"])
                        if isinstance(tc["function"]["arguments"], str)
                        else tc["function"]["arguments"],
                    }
                }
                for tc in tool_calls
            ]

        return result

    @staticmethod
    def _extract_inline_tool_calls(
        content: str,
    ) -> Tuple[str, Optional[List[Dict]]]:
        """Extract Qwen/Hermes-style ``<tool_call>{...}</tool_call>`` tags.

        :param content: Model output text
        :returns: Tuple of (text without tool call tags, tool calls or None)
        """
        calls = []
        for block in _TOOL_CALL_RE.findall(content):
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                continue
            if name := parsed.get("name"):
                calls.append(
                    {
                        "function": {
                            "name": name,
                            "arguments": parsed.get("arguments") or {},
                        }
                    }
                )
        if calls:
            content = _TOOL_CALL_RE.sub("", content).strip()
        return content, calls or None

    def _stream_tokens(self, kwargs: Dict) -> Generator[str, None, None]:
        """Yield decoded text tokens from a streaming response.

        The model lock is held for the whole generation, from inside the
        generator so it is acquired and released by the consuming thread
        The component should guarantee closure of abandoned streams.

        :param kwargs: Keyword arguments for create_chat_completion
        :yields: Decoded text strings, one per chunk
        """
        with self._lock:
            for chunk in self.llm.create_chat_completion(**kwargs):
                delta = chunk["choices"][0]["delta"]
                if "content" in delta and delta["content"]:
                    yield delta["content"]

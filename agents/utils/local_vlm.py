import inspect
import threading
from pathlib import Path
from typing import Dict, Generator, List, Optional, Union

import numpy as np
from .utils import encode_img_base64

# VLM families supported through llama-cpp-python chat handlers.
_VLM_FAMILIES: Dict[str, Dict] = {
    "moondream": {
        "hints": ("moondream",),
        "handlers": ["MoondreamChatHandler"],
        "filename": "*text-model*.gguf",
    },
    # NOTE: MTMD is llama-cpp's generic multimodal handler for newer model families.
    # TODO: When most new models switch to MTMD, we can simplify things here
    "qwen_vl": {
        "hints": ("qwen2-vl", "qwen2.5-vl", "qwen2_5-vl", "qwen25-vl", "qwen3-vl", "qwen3vl"),
        "handlers": ["MTMDChatHandler", "Qwen25VLChatHandler"],
        "filename": "Qwen*.gguf",
    },
    "gemma": {
        "hints": ("gemma",),
        "handlers": ["MTMDChatHandler", "Gemma4ChatHandler"],
        "filename": "gemma*.gguf",
    },
    "minicpm": {
        "hints": ("minicpm",),
        "handlers": ["MiniCPMv26ChatHandler"],
        "filename": "*.gguf",
    },
    "llava16": {
        "hints": ("llava-v1.6", "llava-1.6", "llava16"),
        "handlers": ["Llava16ChatHandler"],
        "filename": "*.gguf",
    },
    "llava": {
        "hints": ("llava", "bakllava"),
        "handlers": ["Llava15ChatHandler"],
        "filename": "*.gguf",
    },
    "nanollava": {
        "hints": ("nanollava",),
        "handlers": ["NanoLlavaChatHandler"],
        "filename": "*.gguf",
    },
}

# Reserved keys in model_options that are not llama_cpp.Llama parameters
_MODEL_TYPE_KEY = "model_type"
_FILENAME_KEY = "filename"


def detect_vlm_family(model_path: str, model_type: Optional[str] = None) -> str:
    """Detect the VLM family from the model path/repository name.

    :param model_path: HuggingFace repo ID or local path.
    :param model_type: Optional family name to force (skips detection).
    :raises ValueError: If no family matches.
    """
    if model_type is not None:
        if model_type not in _VLM_FAMILIES:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Available families: "
                f"{sorted(_VLM_FAMILIES)}"
            )
        return model_type

    name = model_path.lower()
    for family, spec in _VLM_FAMILIES.items():
        if any(hint in name for hint in spec["hints"]):
            return family

    raise ValueError(
        f"Could not detect a VLM family from '{model_path}'. Supported "
        f"families: {sorted(_VLM_FAMILIES)}. Force one by setting "
        f"local_model_options={{'{_MODEL_TYPE_KEY}': '<family>'}} in the config."
    )


class LocalVLM:
    """Local VLM inference using llama-cpp-python multimodal chat handlers.

    The VLM family (moondream, qwen_vl, minicpm, llava/llava16, nanollava) is
    detected from the model name.

    :param model_path: HuggingFace repository ID for a GGUF VLM model
        (e.g. ``ggml-org/Qwen3-VL-2B-Instruct-GGUF``), a local directory
        containing the GGUF and mmproj files, or a local path to a ``.gguf``
        file (with the mmproj file next to it).
    :param device: Device to run on ('cpu' or 'cuda')
    :param ncpu: Number of CPU threads
    :param model_options: Additional keyword options for ``llama_cpp.Llama``,
        validated against its signature (e.g. ``n_ctx``, ``n_batch``,
        ``flash_attn``). Reserved keys: ``filename`` selects the GGUF file
        in multi-file repositories, ``model_type`` forces the VLM family.
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
            from llama_cpp import llama_chat_format
        except ImportError as e:
            raise ImportError(
                "Local VLM model deployment requires llama-cpp-python. "
                "Install it with: pip install llama-cpp-python\n"
                "For NVIDIA GPUs build with: "
                'CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python\n'
            ) from e

        self.device = device
        self.ncpu = ncpu

        options = dict(model_options or {})
        model_type = options.pop(_MODEL_TYPE_KEY, None)
        family = detect_vlm_family(model_path, model_type)
        self.model_family = family
        spec = _VLM_FAMILIES[family]
        filename = options.pop(_FILENAME_KEY, spec["filename"])

        # Resolve a chat handler class available in this llama-cpp version
        handler_cls = None
        for handler_name in spec["handlers"]:
            handler_cls = getattr(llama_chat_format, handler_name, None)
            if handler_cls is not None:
                break
        if handler_cls is None:
            raise ValueError(
                f"This llama-cpp-python version provides no chat handler for "
                f"the '{family}' family (tried: {spec['handlers']}). "
                "Upgrade with: pip install -U llama-cpp-python"
            )

        # Validate user options against the Llama signature
        llama_params = inspect.signature(Llama.__init__).parameters
        for key in options:
            if key not in llama_params:
                valid = sorted(
                    p for p in llama_params if p not in ("self", "kwargs")
                )
                raise ValueError(
                    f"Unknown local_model_options key '{key}' for llama-cpp. "
                    f"Valid options: {valid}. "
                    f"Reserved: ['{_FILENAME_KEY}', '{_MODEL_TYPE_KEY}']."
                )

        kwargs = {
            "n_gpu_layers": -1 if device == "cuda" else 0,
            "n_threads": ncpu,
            "n_ctx": 4096,
            "verbose": False,
            **options,
        }

        local_path = Path(model_path)
        if local_path.is_dir() or model_path.endswith(".gguf"):
            model_dir = local_path if local_path.is_dir() else local_path.parent
            mmproj = next(iter(sorted(model_dir.glob("*mmproj*"))), None)
            if mmproj is None:
                raise FileNotFoundError(
                    f"No multimodal projector (*mmproj*) found in '{model_dir}' "
                    "— required for VLM inference."
                )
            if local_path.is_dir():
                gguf = next(
                    (
                        f
                        for f in sorted(model_dir.glob(filename))
                        if "mmproj" not in f.name
                    ),
                    None,
                )
                if gguf is None:
                    raise FileNotFoundError(
                        f"No GGUF model matching '{filename}' found in '{model_dir}'."
                    )
            else:
                gguf = local_path
            self.llm = Llama(
                model_path=str(gguf),
                chat_handler=handler_cls(clip_model_path=str(mmproj)),
                **kwargs,
            )
        else:
            chat_handler = handler_cls.from_pretrained(
                repo_id=model_path,
                filename="*mmproj*",
            )
            self.llm = Llama.from_pretrained(
                repo_id=model_path,
                filename=filename,
                chat_handler=chat_handler,
                **kwargs,
            )

        # NOTE: llama-cpp contexts are stateful and not thread-safe. Concurrent
        # calls corrupt the shared batch/KV cache and abort the process. We will
        # use thread locking per call
        self._lock = threading.Lock()

    def __call__(
        self, inference_input: Dict, stream: bool = False
    ) -> Union[Dict, Generator[str, None, None]]:
        """Run VLM inference.

        :param inference_input: Dict with 'query' (messages list) and
            'images' (list of RGB numpy arrays), and optional 'temperature',
            'max_new_tokens'
        :param stream: Yield the response as a generator of text chunks
        :returns: Dict with 'output' (str, or a generator when streaming)
        """
        images: List[np.ndarray] = inference_input.get("images", [])
        if not images:
            return {"output": "No image provided."}

        image_parts = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_img_base64(image)}"
                },
            }
            for image in images
        ]

        # Keep the full conversation; attach the images to the last user
        # message
        messages = [dict(message) for message in inference_input["query"]]
        for message in reversed(messages):
            if message["role"] == "user":
                text = message["content"]
                message["content"] = image_parts + [{"type": "text", "text": text}]
                break
        else:
            messages.append({"role": "user", "content": image_parts})

        kwargs = {"messages": messages, "stream": stream}
        if temperature := inference_input.get("temperature"):
            kwargs["temperature"] = temperature
        if max_new_tokens := inference_input.get("max_new_tokens"):
            kwargs["max_tokens"] = max_new_tokens

        if stream:
            return {"output": self._stream_tokens(kwargs)}

        with self._lock:
            response = self.llm.create_chat_completion(**kwargs)
        return {"output": response["choices"][0]["message"]["content"]}

    def _stream_tokens(self, kwargs: Dict) -> Generator[str, None, None]:
        """Yield decoded text tokens from a streaming response.

        The model lock is held for the whole generation, from inside the
        generator so it is acquired and released by the consuming thread.

        :param kwargs: Keyword arguments for create_chat_completion
        :yields: Decoded text strings, one per chunk
        """
        with self._lock:
            for chunk in self.llm.create_chat_completion(**kwargs):
                delta = chunk["choices"][0]["delta"]
                if "content" in delta and delta["content"]:
                    yield delta["content"]

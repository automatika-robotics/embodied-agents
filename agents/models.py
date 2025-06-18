"""
The following model specification classes are meant to define a comman interface for initialization parameters for ML models across supported model serving platforms.
"""

from typing import Optional, Dict, Any

from attrs import define, field
from .ros import BaseAttrs, base_validators

__all__ = [
    "TransformersLLM",
    "TransformersMLLM",
    "OllamaModel",
    "Whisper",
    "SpeechT5",
    "Bark",
    "MeloTTS",
    "VisionModel",
]


@define(kw_only=True)
class Model(BaseAttrs):
    """Model configuration base class"""

    name: str
    checkpoint: str
    init_timeout: Optional[int] = field(default=None)

    def get_init_params(self) -> Dict:
        """Get init params from models"""
        return self._get_init_params()

    def _get_init_params(self) -> Dict:
        raise NotImplementedError(
            "This method needs to be implemented by model definition classes"
        )


@define(kw_only=True)
class LLM(Model):
    """LLM/MLLM model configurations base class.

    :param name: An arbitrary name given to the model.
    :type name: str
    :type quantization: str or None
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional
    """

    quantization: Optional[str] = field(
        default="4bit", validator=base_validators.in_(["4bit", "8bit", None])
    )

    def _get_init_params(self) -> Dict:
        """Get init params for model initialization."""
        return {
            "checkpoint": self.checkpoint,
            "quantization": self.quantization,
        }


@define(kw_only=True)
class OllamaModel(LLM):
    """An Ollama model that needs to be initialized with an ollama tag as checkpoint.

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. For available checkpoints consult [Ollama Models](https://ollama.com/library)
    :type checkpoint: str
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional
    :param options: Optional dictionary to configure generation behavior. Options that conflict with component config options such as (num_predict and temperature) will be overridden if set in component config. Only the following keys with their specified value types are allowed. For details check [Ollama api documentation](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-request-with-options):
        - num_keep: int
        - seed: int
        - num_predict: int
        - top_k: int
        - top_p: float
        - min_p: float
        - typical_p: float
        - repeat_last_n: int
        - temperature: float
        - repeat_penalty: float
        - presence_penalty: float
        - frequency_penalty: float
        - penalize_newline: bool
        - stop: list of strings
        - numa: bool
        - num_ctx: int
        - num_batch: int
        - num_gpu: int
        - main_gpu: int
        - use_mmap: bool
        - num_thread: int
    :type options: dict, optional

     Example usage:
    ```python
    llm = OllamaModel(
        name='ollama1',
        checkpoint="gemma2:latest",
        options={"temperature": 0.7, "num_predict": 50}
    )
    ```"""

    checkpoint: str = field(default="llama3.2:3b")
    port: Optional[int] = field(default=11434)
    options: Optional[Dict[str, Any]] = field(default=None)

    @options.validator
    def _validate_options(self, _, value):
        if value is None:
            return

        allowed_keys = {
            "num_keep": int,
            "seed": int,
            "num_predict": int,
            "top_k": int,
            "top_p": float,
            "min_p": float,
            "typical_p": float,
            "repeat_last_n": int,
            "temperature": float,
            "repeat_penalty": float,
            "presence_penalty": float,
            "frequency_penalty": float,
            "penalize_newline": bool,
            "stop": list,
            "numa": bool,
            "num_ctx": int,
            "num_batch": int,
            "num_gpu": int,
            "main_gpu": int,
            "use_mmap": bool,
            "num_thread": int,
        }

        for key, val in value.items():
            if key not in allowed_keys:
                raise ValueError(f"Invalid key in options: {key}")
            expected_type = allowed_keys[key]
            if key == "stop":
                if not isinstance(val, list) or not all(
                    isinstance(item, str) for item in val
                ):
                    raise TypeError(f"Value for key '{key}' must be a list of strings")
            elif not isinstance(val, expected_type):
                raise TypeError(
                    f"Value for key '{key}' must be of type {expected_type.__name__}"
                )

    def _get_init_params(self) -> Dict:
        """Get init params for model initialization."""
        return {
            "checkpoint": self.checkpoint,
            "options": self.options,
        }


@define(kw_only=True)
class TransformersLLM(LLM):
    """An LLM model that needs to be initialized with any LLM checkpoint available on HuggingFace transformers. This model can be used with a roboml client.

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. Default is "microsoft/Phi-3-mini-4k-instruct". For available checkpoints consult [HuggingFace LLM Models](https://huggingface.co/models?other=LLM)
    :type checkpoint: str
    :param quantization: The quantization scheme used by the model. Can be one of "4bit", "8bit" or None (default is "4bit").
    :type quantization: str or None
    :param system_prompt: The system prompt used to initialize the model. If not provided, defaults to None.
    :type system_prompt: str or None
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    llm = TransformersLLM(name='llm', checkpoint="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ```
    """

    checkpoint: str = field(default="microsoft/Phi-3-mini-4k-instruct")


@define(kw_only=True)
class TransformersMLLM(LLM):
    """An MLLM model that needs to be initialized with any MLLM checkpoint available on HuggingFace transformers. This model can be used with a roboml client.

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. Default is "HuggingFaceM4/idefics2-8b". For available checkpoints consult [HuggingFace Image-Text to Text Models](https://huggingface.co/models?pipeline_tag=image-text-to-text)
    :type checkpoint: str
    :param quantization: The quantization scheme used by the model. Can be one of "4bit", "8bit" or None (default is "4bit").
    :type quantization: str or None
    :param system_prompt: The system prompt used to initialize the model. If not provided, defaults to None.
    :type system_prompt: str or None
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    mllm = TransformersMLLM(name='mllm', checkpoint="gemma2:latest")
    ```
    """

    checkpoint: str = field(default="HuggingFaceM4/idefics2-8b")


@define(kw_only=True)
class Whisper(Model):
    """Whisper is an automatic speech recognition (ASR) system by OpenAI trained on 680,000 hours of multilingual and multitask supervised data collected from the web. [Details](https://openai.com/index/whisper/)

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: Size of the model to use (tiny, tiny.en, base, base.en, small, small.en, distil-small.en, medium, medium.en, distil-medium.en, large-v1, large-v2, large-v3, large, distil-large-v2, distil-large-v3, large-v3-turbo, or turbo). For more information check [here](https://github.com/SYSTRAN/faster-whisper/blob/d3bfd0a305eb9d97c08047c82149c1998cc90fcb/faster_whisper/transcribe.py#L606)
    :type checkpoint: str
    :param compute_type: The compute type used by the model. Can be one of "int8", "fp16", "fp32", None (default is "int8").
    :type compute_type: str or None
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    whisper = Whisper(name='s2t', checkpoint="small") # Initialize with a different checkpoint
    ```
    """

    checkpoint: str = field(default="small.en")
    compute_type: Optional[str] = field(
        default="int8",
        validator=base_validators.in_(["int8", "float16", "float32", None]),
    )

    def _get_init_params(self) -> Dict:
        """Get init params for model initialization."""
        return {"checkpoint": self.checkpoint, "compute_type": self.compute_type}


@define(kw_only=True)
class SpeechT5(Model):
    """A model for text-to-speech synthesis developed by Microsoft. [Details](https://github.com/microsoft/SpeechT5)

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. Default is "microsoft/speecht5_tts".
    :type checkpoint: str
    :param voice: The voice to use for synthesis. Can be one of "awb", "bdl", "clb", "jmk", "ksp", "rms", or "slt". Default is "clb".
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    speecht5 = SpeechT5(name='t2s1', voice="bdl")  # Initialize with a different voice
    ```
    """

    checkpoint: str = field(default="microsoft/speecht5_tts")
    voice: str = field(
        default="clb",
        validator=base_validators.in_([
            "awb",
            "bdl",
            "clb",
            "jmk",
            "ksp",
            "rms",
            "slt",
        ]),
    )

    def _get_init_params(self) -> Dict:
        """Get init params for model initialization."""
        return {"checkpoint": self.checkpoint, "voice": self.voice}


@define(kw_only=True)
class Bark(Model):
    """A model for text-to-speech synthesis developed by SunoAI. [Details](https://github.com/suno-ai/bark)

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. [Bark checkpoints on HuggingFace](https://huggingface.co/collections/suno/bark-6502bdd89a612aa33a111bae). Default is "suno/bark-small".
    :type checkpoint: str
    :param attn_implementation: The attention implementation to use for the model. Default is "flash_attention_2".
    :param voice: The voice to use for synthesis. More choices are available [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c). Default is "v2/en_speaker_6".
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    bark = Bark(name='t2s2', voice="v2/en_speaker_1")  # Initialize with a different voice
    ```
    """

    checkpoint: str = field(default="suno/bark-small")
    voice: str = field(default="v2/en_speaker_6")

    def _get_init_params(self) -> Dict:
        """Get init params for model initialization."""
        return {
            "checkpoint": self.checkpoint,
            "voice": self.voice,
        }


@define(kw_only=True)
class MeloTTS(Model):
    """A model for text-to-speech synthesis developed by MyShell AI using the MeloTTS engine.

    :param name: An arbitrary name given to the model.
    :type name: str
    :param language: The language for speech synthesis. Supported values: ["EN", "ES", "FR", "ZH", "JP", "KR"]. Default is "EN".
    :type language: str
    :param speaker_id: The speaker ID for the chosen language. Default is "EN-US". For details check [here](https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md#python-api)
    :type speaker_id: str
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    melotts = MeloTTS(name='melo1', language='JP', speaker_id='JP-1')
    ```
    """

    language: str = field(default="EN")
    speaker_id: str = field(default="EN-US")

    def _get_init_params(self) -> Dict:
        """Get init params for model initialization."""
        return {
            "language": self.language,
            "speaker_id": self.speaker_id,
        }


@define(kw_only=True)
class VisionModel(Model):
    """Object Detection Model with Optional Tracking.

    This vision model provides a flexible framework for object detection and tracking using the [mmdet framework](https://github.com/open-mmlab/mmdetection). It can be used as a standalone detector or as a tracker to follow detected objects over time. It can be initizaled with any checkpoint available in the mmdet framework.

    :param name: An arbitrary name given to the model.
    :type name: str
    :param checkpoint: The name of the pre-trained model's checkpoint. [All available checkpoints in the mmdet framework](https://github.com/open-mmlab/mmdetection?tab=readme-ov-file#overview-of-benchmark-and-model-zoo). Default is "dino-4scale_r50_8xb2-12e_coco".
    :type checkpoint: str
    :param cache_dir: The directory where downloaded models are cached. Default is 'mmdet'.
    :type cache_dir: str
    :param setup_trackers: Whether to set up trackers using norfair or not. Default is False.
    :type setup_trackers: bool
    :param tracking_distance_function: The function used to calculate the distance between detected objects. This can be any distance metric string available in [scipy.spatial.distance.cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) Default is "euclidean".
    :type tracking_distance_function: str
    :param tracking_distance_threshold: The threshold for determining whether two object in consecutive frames are considered close enough to be considered the same object. Default is 30, with a minimum value of 1.
    :type tracking_distance_threshold: int
    :param deploy_tensorrt: Deploy the vision model using NVIDIA TensorRT. To utilize this feature with roboml, checkout the instructions [here](https://github.com/automatika-robotics/roboml). Default is False.
    :type deploy_tensorrt: bool
    :param _num_trackers: The number of trackers to use. This number depends on the number of inputs image streams being given to the component. It is set automatically if **setup_trackers** is True.
    :type _num_trackers: int
    :param init_timeout: The timeout in seconds for the initialization process. Defaults to None.
    :type init_timeout: int, optional

    Example usage:
    ```python
    model = DetectionModel(name='detection1', setup_trackers=True, num_trackers=1, tracking_distance_threshold=20)  # Initialize the model for tracking one object
    ```
    """

    checkpoint: str = field(default="dino-4scale_r50_8xb2-12e_coco")
    cache_dir: str = field(default="mmdet")
    setup_trackers: bool = field(default=False)
    tracking_distance_function: str = field(default="euclidean")
    tracking_distance_threshold: int = field(
        default=30, validator=base_validators.gt(0)
    )
    deploy_tensorrt: bool = field(default=False)
    _num_trackers: int = field(default=1, validator=base_validators.gt(0))

    def _get_init_params(self) -> Dict:
        """Get init params for model initialization."""
        return {
            "checkpoint": self.checkpoint,
            "cache_dir": self.cache_dir,
            "setup_trackers": self.setup_trackers,
            "num_trackers": self._num_trackers,
            "tracking_distance_function": self.tracking_distance_function,
            "tracking_distance_threshold": self.tracking_distance_threshold,
            "deploy_tensorrt": self.deploy_tensorrt,
        }

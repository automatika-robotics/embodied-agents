from typing import Optional, Union, Dict, List
from pathlib import Path

from attrs import define, field, Factory

from .ros import base_validators, BaseComponentConfig, Topic, Route
from .utils import validate_kwargs

__all__ = [
    "LLMConfig",
    "MLLMConfig",
    "SpeechToTextConfig",
    "TextToSpeechConfig",
    "SemanticRouterConfig",
    "MapConfig",
    "VideoMessageMakerConfig",
    "VisionConfig",
]


@define(kw_only=True)
class ModelComponentConfig(BaseComponentConfig):
    warmup: Optional[bool] = field(default=False)


@define(kw_only=True)
class LLMConfig(ModelComponentConfig):
    """
    Configuration for the Large Language Model (LLM) component.

    It defines various settings that control how the LLM component operates, including
    whether to enable chat history, retreival augmented generation (RAG) and more.

    :param enable_rag: Enables or disables Retreival Augmented Generation.
    :type enable_rag: bool
    :param collection_name: The name of the vectordb collection to use for RAG.
    :type collection_name: Optional[str]
    :param distance_func: The distance metric used for nearest neighbor search for RAG.
        Supported values are "l2", "ip", and "cosine".
    :type distance_func: str
    :param n_results: The maximum number of results to return for RAG. Defaults to 1.
        For numbers greater than 1, results will be concatenated together in a single string.
    :type n_results: int
    :param chat_history: Whether to include chat history in the LLM's prompt.
    :type chat_history: bool
    :param history_reset_phrase: Phrase to reset chat history. Defaults to 'chat reset'
    :type history_reset_phrase: str
    :param history_size: Number of user messages to keep in chat history. Defaults to 10
    :type history_size: int
    :param temperature: Temperature used for sampling tokens during generation.
        Default is 0.8 and must be greater than 0.0.
    :type temperature: float
    :param max_new_tokens: The maximum number of new tokens to generate.
        Default is 100 and must be greater than 0.
    :type max_new_tokens: int
    :param stream: Publish the llm output as a stream of tokens, useful when sending llm output to a user facing client or to a TTS component. Cannot be used in conjunction with tool calling.
        Default is false
    :type stream: bool
    :param break_character: A string character marking that the output thus far received in a stream should be published. This parameter only takes effect when stream is set to True. As stream output is received token by token, it is useful to publish full sentences instead of individual tokens as the components output (for example, for downstream text to speech conversion). This value can be set to an empty string to publish output token by token.
        Default is '.' (period)
    :type break_character: str
    :param response_terminator: A string token marking that the end of a single response from the model. This token is only used in case of a persistent clients, such as a websocket client and it is not published. This value cannot be an empty string.
        Default is '<<Response Ended>>'
    :type response_terminator: str

    Example of usage:
    ```python
    config = LLMConfig(enable_rag=True, collection_name="my_collection", distance_func="l2")
    ```
    """

    enable_rag: bool = field(default=False)
    collection_name: Optional[str] = field(default=None)
    distance_func: str = field(
        default="l2", validator=base_validators.in_(["l2", "ip", "cosine"])
    )
    n_results: int = field(default=1)
    add_metadata: bool = field(default=False)
    chat_history: bool = field(default=False)
    history_reset_phrase: str = field(default="chat reset")
    history_size: int = field(
        default=10, validator=base_validators.gt(4)
    )  # number of user messages
    temperature: float = field(default=0.8, validator=base_validators.gt(0.0))
    max_new_tokens: int = field(default=100, validator=base_validators.gt(0))
    stream: bool = field(default=False)
    break_character: str = field(default=".")
    response_terminator: str = field(default="<<Response Ended>>")
    _system_prompt: Optional[str] = field(default=None, alias="_system_prompt")
    _component_prompt: Optional[Union[str, Path]] = field(
        default=None, alias="_component_prompt"
    )
    _topic_prompts: Dict[str, Union[str, Path]] = field(
        default=Factory(dict), alias="_topic_prompts"
    )
    _tool_descriptions: List[Dict] = field(
        default=Factory(list), alias="_tool_descriptions"
    )
    _tool_response_flags: Dict[str, bool] = field(
        default=Factory(dict), alias="_tool_response_flags"
    )

    @response_terminator.validator
    def not_empty(self, _, value):
        if not value:
            raise ValueError("response_terminator must not be an empty string")

    def _get_inference_params(self) -> Dict:
        """get_inference_params.
        :rtype: dict
        """
        return {
            "chat_history": self.chat_history,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "stream": self.stream,
        }


@define(kw_only=True)
class MLLMConfig(LLMConfig):
    """
    Configuration for the Multi-Modal Large Language Model (MLLM) component.

    It defines various settings that control how the LLM component operates, including
    whether to enable chat history, retreival augmented generation (RAG) and more.

    :param enable_rag: Enables or disables Retreival Augmented Generation.
    :type enable_rag: bool
    :param collection_name: The name of the vectordb collection to use for RAG.
    :type collection_name: Optional[str]
    :param distance_func: The distance metric used for nearest neighbor search for RAG.
        Supported values are "l2", "ip", and "cosine".
    :type distance_func: str
    :param n_results: The maximum number of results to return for RAG.
    :type n_results: int
    :param chat_history: Whether to include chat history in the MLLM's prompt.
    :type chat_history: bool
    :param temperature: Temperature used for sampling tokens during generation.
        Default is 0.7 and must be greater than 0.0.
    :type temperature: float
    :param max_new_tokens: The maximum number of new tokens to generate.
        Default is 100 and must be greater than 0.

    Example of usage:
    ```python
    config = MLLMConfig(enable_rag=True, collection_name="my_collection", distance_func="l2")
    ```
    """

    pass


@define(kw_only=True)
class VisionConfig(ModelComponentConfig):
    """Configuration for a detection component.

    The config allows you to customize the detection and/or tracking process.

    :param threshold: The confidence threshold for object detection, ranging from 0.1 to 1.0 (default: 0.5).
    :type threshold: float
    :param get_data_labels: Whether to return data labels along with detections (default: True).
    :type get_data_labels: bool
    :param labels_to_track: A list of specific labels to track, when the model is used as a tracker (default: None).
    :type labels_to_track: Optional[list]

    Example of usage:
    ```python
    config = DetectionConfig(threshold=0.3)
    ```
    """

    threshold: float = field(
        default=0.5, validator=base_validators.in_range(min_value=0.1, max_value=1.0)
    )
    get_data_labels: bool = field(default=True)
    labels_to_track: Optional[List[str]] = field(default=None)
    enable_visualization: Optional[bool] = field(default=False)

    def _get_inference_params(self) -> Dict:
        """get_inference_params.
        :rtype: dict
        """
        return {
            "threshold": self.threshold,
            "get_data_labels": self.get_data_labels,
            "labels_to_track": self.labels_to_track,
        }


@define(kw_only=True)
class TextToSpeechConfig(ModelComponentConfig):
    """Configuration for a Text-To-Speech component.

    This class defines the configuration options for a Text-To-Speech component.

    :param play_on_device: Whether to play the audio on available audio device (default: False).
    :type play_on_device: bool
    :param device: Device id (int) or name (sub-string) for playing the audio. Only effective if play_on_device is True (default: 'default').
    :type play_on_device: bool
    :param buffer_size: Size of the buffer for playing audio on device. Only effective if play_on_device is True (default: 20).
    :type buffer_size: int
    :param block_size: Size of the audio block to be read for playing audio on device. Only effective if play_on_device is True (default: 1024).
    :type block_size: int
    :param get_bytes: Whether the model should return the speech data as bytes instead of base64 encoded string(default: False).
    :type get_bytes: bool

    Example of usage:
    ```python
    config = TextToSpeechConfig(play_on_device=True, get_bytes=False)
    ```
    """

    play_on_device: bool = field(default=False)
    device: Union[int, str] = field(default="default")
    buffer_size: int = field(default=20)
    block_size: int = field(default=1024)
    get_bytes: bool = field(default=False)

    def _get_inference_params(self) -> Dict:
        """get_inference_params.
        :rtype: dict
        """
        return {"get_bytes": self.get_bytes}


@define(kw_only=True)
class SpeechToTextConfig(ModelComponentConfig):
    """
    Configuration for a Speech-To-Text component.

    This class defines the configuration options for a Speech-To-Text component.

    :param enable_vad: Enable Voice Activity Detection (VAD) to identify when speech is present in continuous input stream from an input audio device. Uses silero-vad model and requires onnxruntime to be installed.
                       Defaults to False.
    :type enable_vad: bool
    :param enable_wakeword: Enable Wakeword Detection to identify a specific key phrase in the audio stream, e.g 'Hey Jarvis'.
                            Defaults to False.
    :type enable_wakeword: bool
    :param device_audio: Device id (int) to use for audio input. Only effective if `enable_vad` is set to true. Defaults to 0.
    :type device_audio: int
    :param vad_threshold: Minimum threshold above which speech is considered present. Only effective if `enable_vad` is set to true. Defaults to 0.5 (50%).
    :type vad_threshold: float
    :param wakeword_threshold: Minimum threshold for detecting the wake word phrase. Only effective if `enable_wakeword` is set to true. Defaults to 0.6 (60%).
    :type wakeword_threshold: float
    :param min_silence_duration_ms: Minimum duration of silence in milliseconds before considering it as a speaker pause. Only effective if `enable_vad` is set to true. Defaults to 1000 ms.
    :type min_silence_duration_ms: int
    :param speech_pad_ms: Duration in milliseconds to pad silence at the start and end of detected speech regions. Only effective if `enable_vad` is set to true. Defaults to 30 ms.
    :type speech_pad_ms: int
    :param speech_buffer_max_len: Maximum length of the speech buffer in milliseconds. Defaults to 8000ms. Only effective if `enable_vad` is set to true.
    :type speech_buffer_max_len: int
    :param device_vad: Device type for VAD processing ('cpu' or 'gpu'). Only effective if `enable_vad` is set to true. Defaults to 'cpu'.
    :type device_vad: str
    :param device_wakeword: Device type for Wakeword detection ('cpu' or 'gpu'). Only effective if `enable_wakeword` is set to true. Defaults to 'cpu'.
    :type device_wakeword: str
    :param ncpu_vad: Number of CPU cores to use for VAD processing. Only effective if `device_vad` is 'cpu'. Defaults to 1.
    :type ncpu_vad: int
    :param ncpu_wakeword: Number of CPU cores to use for Wakeword detection. Only effective if `device_wakeword` is 'cpu'. Defaults to 1.
    :type ncpu_wakeword: int
    :param vad_model_path: File path or URL to the VAD model file. Defaults to the URL for Silero VAD ONNX model.
    :type vad_model_path: str
    :param melspectrogram_model_path: File path or URL to the melspectrogram model file used by the Wakeword detection. Defaults to the URL for melspectrogram ONNX model provided by openWakeWord.
    :type melspectrogram_model_path: str
    :param embedding_model_path: File path or URL to the audio embedding model file used by the Wakeword detection. Defaults to the URL for embedding ONNX model provided by openWakeWord, which is a reimplmentation of audio embeddings model provided by Google. License Apache-2.0.
    :type embedding_model_path: str
    :param wakeword_model_path: File path or URL to the Wakeword model file. Defaults to the URL for pretrained 'Hey Jarvis' wakeword ONNX model provided by openWakeWord. To train your custom wakeword model, follow the [tutorial](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb) provided by openWakeWord.
    :type wakeword_model_path: str

    Example of usage:
    ```python
    config = SpeechToTextConfig(
        enable_vad=True,
        enable_wakeword=True,
        device_audio=1,
        vad_threshold=0.5,
        wakeword_threshold=0.6,
        min_silence_duration_ms=1000,
        speech_pad_ms=30,
        speech_buffer_max_len=8000,
    )
    """

    enable_vad: bool = field(default=False)
    enable_wakeword: bool = field(default=False)
    device_audio: int = field(default=0)
    vad_threshold: float = field(
        default=0.5, validator=base_validators.in_range(min_value=0.0, max_value=1.0)
    )
    wakeword_threshold: float = field(
        default=0.6, validator=base_validators.in_range(min_value=0.0, max_value=1.0)
    )
    min_silence_duration_ms: int = field(default=1000)
    speech_pad_ms: int = field(default=30)
    speech_buffer_max_len: int = field(default=8000)
    device_vad: str = field(
        default="cpu", validator=base_validators.in_(["cpu", "gpu"])
    )
    device_wakeword: str = field(
        default="cpu", validator=base_validators.in_(["cpu", "gpu"])
    )
    ncpu_vad: int = field(default=1)
    ncpu_wakeword: int = field(default=1)
    vad_model_path: str = field(
        default="https://raw.githubusercontent.com/snakers4/silero-vad/refs/heads/master/src/silero_vad/data/silero_vad.onnx"
    )
    melspectrogram_model_path: str = field(
        default="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx"
    )
    embedding_model_path: str = field(
        default="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx"
    )
    wakeword_model_path: str = field(
        default="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx"
    )
    _sample_rate: int = field(default=16000)
    _block_size: int = field(default=1280)

    @enable_wakeword.validator
    def check_wakeword(self, _, value):
        if value and not self.enable_vad:
            raise ValueError(
                "enable_vad (voice activity detection) must be set to True when enable_wakeword is True"
            )

    def _get_inference_params(self) -> Dict:
        """get_inference_params.
        :rtype: dict
        """
        return {}


def _get_optional_topic(topic: Union[Topic, Dict]) -> Optional[Topic]:
    if not topic:
        return
    if isinstance(topic, Topic):
        return topic
    return Topic(**topic)


@define(kw_only=True)
class MapConfig(BaseComponentConfig):
    """Configuration for a MapEncoding component.

    :param map_name: The name of the map.
    :type map_name: str
    :param distance_func: The function used to calculate distance when retreiving information from the map collection. Can be one of "l2" (L2 distance), "ip" (Inner Product), or "cosine" (Cosine similarity). Default is "l2".
    :type distance_func: str

    Example of usage:
    ```python
    config = MapConfig(map_name="my_map", distance_func="ip")
    ```
    """

    map_name: str = field()
    distance_func: str = field(
        default="l2", validator=base_validators.in_(["l2", "ip", "cosine"])
    )
    _position: Optional[Union[Topic, Dict]] = field(
        default=None, converter=_get_optional_topic, alias="_position"
    )
    _map_topic: Optional[Union[Topic, Dict]] = field(
        default=None, converter=_get_optional_topic, alias="_map_topic"
    )


def _get_optional_route(route: Union[Route, Dict]) -> Optional[Route]:
    if not route:
        return
    if isinstance(route, Route):
        return route
    return Route(**route)


@define(kw_only=True)
class SemanticRouterConfig(BaseComponentConfig):
    """Configuration parameters for a semantic router component.

    :param router_name: The name of the router.
    :type router_name: str
    :param distance_func: The function used to calculate distance from route samples in vectordb. Can be one of "l2" (L2 distance), "ip" (Inner Product), or "cosine" (Cosine similarity). Default is "l2".
    :type distance_func: str
    :param maximum_distance: The maximum distance threshold for routing. A value between 0.1 and 1.0. Defaults to 0.4
    :type maximum_distance: float

    Example of usage:
    ```python
    config = SemanticRouterConfig(router_name="my_router")
    # or
    config = SemanticRouterConfig(router_name="my_router", distance_func="ip", maximum_distance=0.7)
    ```
    """

    router_name: str = field()
    distance_func: str = field(
        default="l2", validator=base_validators.in_(["l2", "ip", "cosine"])
    )
    maximum_distance: float = field(
        default=0.4, validator=base_validators.in_range(min_value=0.1, max_value=1.0)
    )
    _default_route: Optional[Union[Route, Dict]] = field(
        default=None, converter=_get_optional_route, alias="_default_route"
    )


@define(kw_only=True)
class VideoMessageMakerConfig(BaseComponentConfig):
    """Configuration parameters for a video message maker component.

    :param min_video_frames: The minimum number of frames in a video segment. Default is 15, assuming a 0.5 second video at 30 fps.
    :type min_video_frames: int
    :param max_video_frames: The maximum number of frames in a video segment. Default is 600, assuming a 20 second video at 30 fps.
    :type max_video_frames: int
    :param motion_estimation_func: The function used for motion estimation. Can be one of "frame_difference" or "optical_flow". Default is None.
    :type motion_estimation_func: Optional[str]
    :param threshold: The threshold value for motion detection. A float between 0.1 and 5.0. Default is 0.3.
    :type threshold: float
    :param flow_kwargs: Additional keyword arguments for the optical flow algorithm. Default is a dictionary with reasonable values.

    Example of usage:
    ```python
    config = VideoMessageMakerConfig()
    # or
    config = VideoMessageMakerConfig(min_video_frames=30, motion_estimation_func="optical_flow", threshold=0.5)
    ```
    """

    min_video_frames: int = field(default=15)  # assuming 0.5 second video at 30 fps
    max_video_frames: int = field(default=600)  # assuming 20 second video at 30 fps
    motion_estimation_func: Optional[str] = field(
        default=None,
        validator=base_validators.in_(["frame_difference", "optical_flow"]),
    )
    threshold: float = field(
        default=0.3, validator=base_validators.in_range(min_value=0.1, max_value=5.0)
    )
    flow_kwargs: Dict = field(
        default={
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 15,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.1,
            "flags": 0,
        },
        validator=validate_kwargs,
    )

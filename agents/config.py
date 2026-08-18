from typing import Optional, Union, Dict, List, Literal, Mapping
from pathlib import Path

from attrs import define, field, Factory, validators

from .ros import base_validators, BaseComponentConfig, Topic
from .utils import validate_kwargs_from_default, _LANGUAGE_CODES

__all__ = [
    "LLMConfig",
    "MLLMConfig",
    "VLMConfig",
    "CortexConfig",
    "VLAConfig",
    "MoveItConfig",
    "SpeechToTextConfig",
    "TextToSpeechConfig",
    "SemanticRouterConfig",
    "MapConfig",
    "MemoryConfig",
    "MotionDetectorConfig",
    "VideoMessageMakerConfig",
    "VisionConfig",
]


# --- HELPERS ---
def _get_optional_topic(topic: Union[Topic, Dict]) -> Optional[Topic]:
    if not topic:
        return
    if isinstance(topic, Topic):
        return topic
    return Topic(**topic)


@define(kw_only=True)
class ModelComponentConfig(BaseComponentConfig):
    warmup: Optional[bool] = field(default=False)

    def get_inference_params(self) -> Dict:
        """Get inference params from model components"""
        return self._get_inference_params()

    def _get_inference_params(self) -> Dict:
        raise NotImplementedError(
            "_get_inference_params method needs to be implemented by model config classes"
        )


@define(kw_only=True)
class LLMConfig(ModelComponentConfig):
    """
    Configuration for the Large Language Model (LLM) component.

    It defines various settings that control how the LLM component operates, including
    whether to enable chat history, retrieval augmented generation (RAG) and more.

    :param enable_rag: Enables or disables Retrieval Augmented Generation.
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
        Default is 512 and must be greater than 0.
    :type max_new_tokens: int
    :param stream: Publish the llm output as a stream of tokens, useful when sending llm output to a user facing client or to a TTS component. Cannot be used in conjunction with tool calling.
        Default is false
    :type stream: bool
    :param break_character: A string character marking that the output thus far received in a stream should be published. This parameter only takes effect when stream is set to True. As stream output is received token by token, it is useful to publish full sentences instead of individual tokens as the components output (for example, for downstream text to speech conversion). This value can be set to an empty string to publish output token by token.
        Default is '.' (period)
    :type break_character: str
    :param response_terminator: A string token marking that the end of a single response from the model. This token is only used in case of a persistent clients, such as a websocket client and when stream is set to True. It is not published. This value cannot be an empty string.
        Default is '<<Response Ended>>'
    :type response_terminator: str
    :param strip_think_tokens: Whether to strip ``<think>...</think>`` blocks from model output. Reasoning models emit these blocks which are useful for debugging but should typically not be forwarded to downstream components such as TTS or UI. Applies to both streaming and non-streaming output. Default is True.
    :type strip_think_tokens: bool
    :param enable_local_model: Whether to enable a local LLM model via llama.cpp, allowing the component to run without a remote model client. Requires the ``llama-cpp-python`` package. Default is False.
    :type enable_local_model: bool
    :param device_local_model: Device to run the local model on, either "cpu" or "cuda" (default: "cuda"). This parameter is only effective when ``enable_local_model`` is True.
    :type device_local_model: str
    :param ncpu_local_model: Number of CPU cores to allocate to the local model when using CPU (default: 1). This parameter is only effective when ``enable_local_model`` is True.
    :type ncpu_local_model: int
    :param local_model_path: HuggingFace repository ID for a GGUF model (default: ``Qwen/Qwen3-0.6B-GGUF``), or a local path to a ``.gguf`` file. This parameter is only effective when ``enable_local_model`` is True.
    :type local_model_path: Optional[str]
    :param local_model_options: Additional options for the local model, validated at load time against the ``llama_cpp.Llama`` signature (e.g. ``n_ctx``, ``n_batch``, ``flash_attn``, ``chat_format``). Reserved keys: ``filename`` selects the GGUF file when a repository ships several quantizations (e.g. ``"*q4_k_m*.gguf"``); for VLM components ``model_type`` additionally forces the VLM family (moondream, qwen_vl, minicpm, llava, llava16, nanollava) instead of detecting it from the model name. An unknown key raises an error listing the valid keys. Only effective when ``enable_local_model`` is True. Default is ``{}``.
    :type local_model_options: Dict

    Example of usage:
    ```python
    config = LLMConfig(enable_rag=True, collection_name="my_collection", distance_func="l2")
    ```

    Example of usage with local model:
    ```python
    config = LLMConfig(enable_local_model=True)
    ```
    """

    enable_rag: bool = field(default=False)
    collection_name: Optional[str] = field(default=None)
    distance_func: Literal["l2", "ip", "cosine"] = field(
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
    max_new_tokens: int = field(default=512, validator=base_validators.gt(0))
    stream: bool = field(default=False)
    break_character: str = field(default=".")
    response_terminator: str = field(default="<<Response Ended>>")
    strip_think_tokens: bool = field(default=True)
    enable_local_model: bool = field(default=False)
    device_local_model: Literal["cpu", "cuda"] = field(
        default="cuda", validator=base_validators.in_(["cpu", "cuda"])
    )
    ncpu_local_model: int = field(default=1)
    local_model_path: Optional[str] = field(default="Qwen/Qwen3-0.6B-GGUF")
    local_model_options: Dict = field(default=Factory(dict))
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
    _component_tool_names: List[str] = field(
        default=Factory(list), alias="_component_tool_names"
    )
    # Only used when LLM is used as a router
    _default_route: Optional[str] = field(default=None, alias="_default_route")

    @response_terminator.validator
    def _not_empty(self, _, value):
        if not value:
            raise ValueError("response_terminator must not be an empty string")

    def _get_inference_params(self) -> Dict:
        """get_inference_params.
        :rtype: dict
        """
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "stream": self.stream,
        }


@define(kw_only=True)
class CortexConfig(LLMConfig):
    """
    Configuration for the Cortex task planning and execution component.

    The Cortex component uses an LLM to decompose high-level tasks into sub-tasks
    and executes them by dispatching Actions registered on other components.

    The task execution follows a two-phase approach:

    1. **Planning** — A multi-step conversational loop where the LLM can call
       ``inspect_component`` to research available components and their capabilities.
       Once the LLM has enough context, it returns action tool calls which become
       the execution plan. RAG context from a vector DB is also available during
       this phase. Controlled by ``max_planning_steps``.
    2. **Execution** — Each planned step is executed sequentially. Before each
       step, a brief LLM confirmation call decides: EXECUTE, SKIP, or ABORT,
       based on the original plan and results so far. After a plan is fully
       executed, Cortex feeds the results back to the planner and may produce
       a follow-up plan, repeating the plan-execute loop until the planner
       signals completion. Both the per-plan length and the number of
       plan-execute iterations are capped by ``max_execution_steps``.

    The ``chat_history`` and ``stream`` fields are enforced by the component
    (``chat_history=True``, ``stream=False``) and cannot be overridden.

    :param max_planning_steps: Maximum number of LLM calls allowed during the
        planning phase (e.g. inspect_component calls). Default is 10.
    :type max_planning_steps: int
    :param max_execution_steps: Caps two things at once: (1) the maximum
        number of action steps allowed in any single execution plan, plans
        with more steps are truncated; and (2) the maximum number of
        plan-execute iterations Cortex will run before giving up if the
        planner never signals completion. The worst-case total number of
        actions executed for one task is therefore ``max_execution_steps²``.
        Default is 10.
    :type max_execution_steps: int
    :param confirmation_temperature: Temperature for the per-step confirmation LLM
        calls. Used for both the decision and resolving tool call arguments
        from prior step results. Default is 0.3.
    :type confirmation_temperature: float
    :param confirmation_max_tokens: Maximum tokens for confirmation responses.
        Must be large enough to accommodate a tool call with resolved arguments
        when the LLM returns EXECUTE. Default is 500.
    :type confirmation_max_tokens: int
    :param temperature: Temperature used for the planning LLM call.
        Default is 0.8 and must be greater than 0.0.
    :type temperature: float
    :param max_new_tokens: The maximum number of new tokens to generate during planning.
        Default is 1000 (inherited from LLMConfig) and must be greater than 0.
    :type max_new_tokens: int
    :param enable_rag: Enable Retrieval Augmented Generation to provide context
        during planning. Requires a ``db_client`` to be passed to the Cortex component.
        Default is False.
    :type enable_rag: bool
    :param strip_think_tokens: Whether to strip ``<think>...</think>`` blocks from model output. Default is True.
    :type strip_think_tokens: bool
    :param enable_local_model: Whether to enable a local LLM via llama.cpp. Requires ``llama-cpp-python``. Default is False.
    :type enable_local_model: bool
    :param device_local_model: Device to run the local model on, either "cpu" or "cuda" (default: "cuda").
    :type device_local_model: str
    :param ncpu_local_model: Number of CPU cores for the local model (default: 1).
    :type ncpu_local_model: int
    :param local_model_path: HuggingFace repository ID for a GGUF model (default: ``Qwen/Qwen3-0.6B-GGUF``), or a local path to a ``.gguf`` file.
    :type local_model_path: Optional[str]

    Example of usage:
    ```python
    config = CortexConfig(max_planning_steps=10, max_execution_steps=15, temperature=0.2)
    ```

    Example of usage with local model:
    ```python
    config = CortexConfig(enable_local_model=True, max_execution_steps=20)
    ```
    """

    max_planning_steps: int = field(default=10, validator=base_validators.gt(0))
    max_execution_steps: int = field(default=10, validator=base_validators.gt(0))
    confirmation_temperature: float = field(
        default=0.3, validator=base_validators.gt(0.0)
    )
    confirmation_max_tokens: int = field(default=500, validator=base_validators.gt(0))
    monitoring_interval: float = field(default=2.0, validator=base_validators.gt(0.0))

    def _get_inference_params(self) -> Dict:
        """get_inference_params.
        :rtype: dict
        """
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "stream": False,
        }


@define(kw_only=True)
class MLLMConfig(LLMConfig):
    """
    Configuration for the Multi-Modal LLM (VLM) component.

    It defines various settings that control how the VLM component operates, including
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
    :param response_terminator: A string token marking that the end of a single response from the model. This token is only used in case of a persistent clients, such as a websocket client and when stream is set to True. It is not published. This value cannot be an empty string.
        Default is '<<Response Ended>>'
    :type response_terminator: str
    :param strip_think_tokens: Whether to strip ``<think>...</think>`` blocks from model output. Reasoning models emit these blocks which are useful for debugging but should typically not be forwarded to downstream components such as TTS or UI. Applies to both streaming and non-streaming output. Default is True.
    :type strip_think_tokens: bool
     :param task: The specific task the VLM should perform. This can help tailor model behavior and is useful when the VLM being used with the component has been trained on specific tasks. For an example of such a model check out RoboBrain2 in models.
        Supported values are: "general", "pointing", "affordance", "trajectory", and "grounding".
        Default is None.
    :type task: Optional[Literal["general", "pointing", "affordance", "trajectory", "grounding"]]
    :param enable_local_model: Whether to enable a local VLM via llama.cpp (Qwen3-VL by default), allowing the component to run without a remote model client. Requires the ``llama-cpp-python`` package. Default is False.
    :type enable_local_model: bool
    :param device_local_model: Device to run the local model on, either "cpu" or "cuda" (default: "cuda"). This parameter is only effective when ``enable_local_model`` is True.
    :type device_local_model: str
    :param ncpu_local_model: Number of CPU cores to allocate to the local model when using CPU (default: 1). This parameter is only effective when ``enable_local_model`` is True.
    :type ncpu_local_model: int
    :param local_model_path: HuggingFace repository ID for a GGUF VLM model (default: ``ggml-org/Qwen3-VL-2B-Instruct-GGUF``), a local directory with the GGUF and mmproj files, or a local path to a ``.gguf`` file. The VLM family (qwen_vl, gemma, moondream, minicpm, llava, llava16, nanollava) is detected from the model name. This parameter is only effective when ``enable_local_model`` is True.
    :type local_model_path: Optional[str]
    :param detections_frame: Frame that 3D detections are published in, usually the frame the consumer plans in, e.g. "base_link". Boxes are axis aligned in this frame and it is chosen before they are measured, so it cannot be changed after the fact. Required when a Detections3D output topic is given. Only meaningful with the "grounding" or "affordance" task.
    :type detections_frame: str
    :param static_camera_tf: Whether the transform from the camera to `detections_frame` is fixed, which it is for a camera bolted to the robot. Set False for a camera that moves with a moving joint, so the transform keeps being looked up. Default is True.
    :type static_camera_tf: bool
    :param depth_scale: Multiplier from the depth image's units to millimeters, overriding what its encoding implies. Default is None, which derives it from the encoding.
    :type depth_scale: Optional[float]
    :param min_depth: Closest depth reading to treat as usable, in meters. Default is 0.05.
    :type min_depth: float
    :param max_depth: Furthest depth reading to treat as usable, in meters. Default is 5.0.
    :type max_depth: float
    :param max_depth_age: How far apart in time the color and depth frames may be before the pair is treated as mismatched, in seconds. Default is 0.2.
    :type max_depth_age: float
    :param min_depth_validity: Minimum fraction of usable depth pixels inside a detection's 2D box for its 3D box to be published, in [0, 1]. A box built from a handful of readings is not a shape, and a planner would take it for one. Default is 0.1.
    :type min_depth_validity: float

    Example of usage:
    ```python
    config = MLLMConfig(enable_rag=True, collection_name="my_collection", distance_func="l2", task=grounding)
    ```

    Example of usage with local model:
    ```python
    config = MLLMConfig(enable_local_model=True)
    ```
    """

    task: Optional[
        Literal["general", "pointing", "affordance", "trajectory", "grounding"]
    ] = field(
        default=None,
        validator=validators.optional(
            base_validators.in_(
                ["general", "pointing", "affordance", "trajectory", "grounding"]
            )
        ),
    )
    local_model_path: Optional[str] = field(
        default="ggml-org/Qwen3-VL-2B-Instruct-GGUF"
    )
    # NOTE: 3D lift fields, kept identical to VisionConfig
    detections_frame: str = field(default="")
    static_camera_tf: bool = field(default=True)
    depth_scale: Optional[float] = field(default=None)
    min_depth: float = field(default=0.05, validator=base_validators.gt(0.0))
    max_depth: float = field(default=5.0, validator=base_validators.gt(0.0))
    max_depth_age: float = field(default=0.2, validator=base_validators.gt(0.0))
    min_depth_validity: float = field(
        default=0.1, validator=base_validators.in_range(min_value=0.0, max_value=1.0)
    )
    # serialized topics
    _depth_topic: Optional[Topic] = field(
        default=None, converter=_get_optional_topic, alias="_depth_topic"
    )
    _camera_info_topic: Optional[Topic] = field(
        default=None, converter=_get_optional_topic, alias="_camera_info_topic"
    )

    @max_depth.validator
    def _check_depth_range(self, _, value):
        """A sensor cannot see further than it can see"""
        if value <= self.min_depth:
            raise ValueError(
                f"max_depth ({value}) must be greater than min_depth ({self.min_depth})"
            )

    @task.validator
    def _check_task(self, _, value):
        """Task validator"""
        if value and self.stream:
            raise ValueError(
                "stream cannot be set to True when a task is set in VLMConfig"
            )
        if value and value != "general" and self.enable_local_model:
            raise ValueError(
                f"Local VLM model only supports general VQA. "
                f"Task '{value}' requires a remote model client."
            )

    def _get_inference_params(self) -> Dict:
        """get_inference_params.
        :rtype: dict
        """
        llm_params = super()._get_inference_params()
        return {**llm_params, "task": self.task} if self.task else llm_params


# Alias
VLMConfig = MLLMConfig


@define(kw_only=True)
class VLAConfig(ModelComponentConfig):
    """
    Configuration for the Vision-Language-Action (VLA) component.

    It defines settings that control how the VLA component maps sensor inputs to the model,
    manages the frequency of observation and action loops, and enforces safety constraints
    through URDF limits.

    :param joint_names_map: A dictionary mapping the joint names expected by the model
        (keys) to the actual joint names in the robot's URDF/ROS system (values).
    :type joint_names_map: Dict[str, str]
    :param camera_inputs_map: A mapping of camera names expected by the model (keys)
        to the corresponding ROS topics (values). A camera whose dataset feature is
        single channel is treated as a depth camera and fetches depth frames from its
        topic (a depth Image topic or the depth part of an RGBD topic). Depth cameras
        require ``dataset_info_file`` to be set on the LeRobotPolicy, as the
        auto-generated feature spec assumes 3-channel RGB for every camera.
    :type camera_inputs_map: Mapping[str, Union[Topic, Dict]]
    :param state_input_type: The type of state data to extract from the joint state inputs.
        Supported values are "positions", "velocities", "accelerations", and "efforts".
        Default is "positions".
    :type state_input_type: Literal["positions", "velocities", "accelerations", "efforts"]
    :param action_output_type: The type of action data to publish to the robot controller.
        Supported values are "positions", "velocities", "accelerations", and "efforts".
        Default is "positions".
    :type action_output_type: Literal["positions", "velocities", "accelerations", "efforts"]
    :param observation_sending_rate: The frequency (in Hz) at which observations are
        captured and sent to the model for inference. Default is 10.0 Hz.
    :type observation_sending_rate: float
    :param action_sending_rate: The frequency (in Hz) at which action commands are
        published to the robot's controllers. Default is 10.0 Hz.
    :type action_sending_rate: float
    :param input_timeout: The maximum time (in seconds) to wait for all required inputs
        (joints, images) to become available before aborting an action after an action request.
        Default is 30.0s.
    :type input_timeout: float
    :param robot_urdf_file: Path to the robot's URDF file. This is strongly recommended
        for safety, as it allows the component to read joint limits and cap generated
        actions within safe bounds.
    :type robot_urdf_file: Optional[str]
    :param joint_limits: A manual dictionary of joint limits to be used if a URDF file
        is not provided. Format should match parsed URDF limits. When a URDF file is
        also provided, entries in this dictionary override the URDF-derived limits for
        those joints.
    :type joint_limits: Optional[Dict]
    :param policy_action_units: The unit space of the policy's actions, used to
        convert URDF-derived joint limits before capping. URDF `<limit>` values are
        always radians (per the URDF spec); this option converts them into the unit
        space of the policy's actions:
        - `"radians"` (default) — use URDF values as-is. Correct only when the policy
          outputs radians.
        - `"degrees"` — convert lower/upper (and velocity) to degrees.
        - `"normalized"` — the LeRobot SO-10x motor-unit convention: each joint's
          [lower, upper] range is mapped to [-100, 100]; joints whose name contains
          "gripper" or "jaw" are mapped to [0, 100]. Use this for policies trained on
          LeRobot datasets with normalized motor positions.
        Only applies to URDF-derived limits; the manual `joint_limits` dict is always
        used verbatim.
    :type policy_action_units: Literal["radians", "degrees", "normalized"]
    :param aggregate_fn_name: The strategy used to merge actions when newly received
        action chunks overlap timesteps already in the queue (chunks from consecutive
        inferences overlap). Presets mirror the LeRobot client:
        "latest_only" (new action wins), "weighted_average" (0.3 * old + 0.7 * new),
        "average" (0.5 * old + 0.5 * new) and "conservative" (0.7 * old + 0.3 * new).
        A custom callable set with `set_aggregation_function` on the component takes
        precedence over this preset. Default is "latest_only".
    :type aggregate_fn_name: Literal["latest_only", "weighted_average", "average", "conservative"]

    Example of usage:
    ```python
    joints_map = {"shoulder_pan": "joint1", "elbow_flex": "joint2"}
    camera_map = {"front_view": camera_topic}

    config = VLAConfig(
        joint_names_map=joints_map,
        camera_inputs_map=camera_map,
        observation_sending_rate=5.0,
        robot_urdf_file="/path/to/robot.urdf"
    )
    ```
    """

    joint_names_map: Dict[str, str] = field()
    camera_inputs_map: Mapping[str, Union[Topic, Dict]] = field()
    # TODO: One can make models that take multiple state input types.
    # This parameter would have to be revised in that case
    state_input_type: Literal["positions", "velocities", "accelerations", "efforts"] = (
        field(
            default="positions",
            validator=base_validators.in_(
                ["positions", "velocities", "accelerations", "efforts"]
            ),
        )
    )
    # TODO: One can make models that produce multiple action output types.
    # This parameter would have to be revised in that case
    action_output_type: Literal[
        "positions", "velocities", "accelerations", "efforts"
    ] = field(
        default="positions",
        validator=base_validators.in_(
            ["positions", "velocities", "accelerations", "efforts"]
        ),
    )
    observation_sending_rate: float = field(
        default=10.0, validator=base_validators.in_range(min_value=1e-6, max_value=1e6)
    )
    action_sending_rate: float = field(
        default=10.0, validator=base_validators.in_range(min_value=1e-6, max_value=1e6)
    )
    input_timeout: float = field(
        default=30.0, validator=base_validators.in_range(min_value=1e-6, max_value=1e6)
    )  # seconds
    robot_urdf_file: Optional[str] = field(default=None)
    joint_limits: Optional[Dict] = field(default=None)
    policy_action_units: Literal["radians", "degrees", "normalized"] = field(
        default="radians",
        validator=base_validators.in_(["radians", "degrees", "normalized"]),
    )
    aggregate_fn_name: Literal[
        "latest_only", "weighted_average", "average", "conservative"
    ] = field(
        default="latest_only",
        validator=base_validators.in_(
            ["latest_only", "weighted_average", "average", "conservative"]
        ),
    )
    _termination_mode: Literal["timesteps", "keyboard", "event"] = field(
        default="timesteps", alias="_termination_mode"
    )
    _termination_timesteps: int = field(
        default=50, validator=base_validators.gt(0), alias="_termination_timesteps"
    )
    _termination_key: str = field(default="q", alias="_termination_key")

    def __attrs_post_init__(self):
        """Post Init"""
        # Main action loop is executed at the loop rate
        # So we set the loop rate equal to observation sending rate
        self.loop_rate = self.observation_sending_rate

    def _get_inference_params(self) -> Dict:
        return {}


@define(kw_only=True)
class MoveItConfig(BaseComponentConfig):
    """
    Configuration for the MoveIt manipulation component.

    It defines which planning groups to command on a running MoveIt 2 `move_group`
    node, how to plan (planner selection, effort, tolerances) and how the gripper
    is controlled.

    :param arm_group_name: Name of the SRDF planning group of the arm (e.g. "panda_arm"). Group names are defined in the robot's MoveIt (SRDF) configuration.
    :type arm_group_name: str
    :param gripper_group_name: Name of the SRDF planning group of the gripper (e.g. "hand"). Required for gripper control when `gripper_mode` is "move_group". Default is None.
    :type gripper_group_name: Optional[str]
    :param cartesian_group_name: Planning group used for Cartesian path requests (straight-line motions, including the descend/retreat steps of pick and place). Default is None, which uses `arm_group_name`. Setting a separate group lets an underactuated (e.g. 5-DOF) arm pair a position-only IK configuration on the arm group (for pose goals) with an orientation-tracking IK configuration on a twin group over the same chain. MoveIt's Cartesian interpolator validates the achieved orientation of every step against a fixed precision the request cannot override, which a position-only solver cannot meet.
    :type cartesian_group_name: Optional[str]
    :param end_effector_link: End-effector link that pose targets and Cartesian waypoints refer to. Empty (default) uses the planning group's default tip link.
    :type end_effector_link: str
    :param pose_reference_frame: Default reference frame for pose targets that carry an empty `header.frame_id`. Empty (default) uses move_group's planning frame.
    :type pose_reference_frame: str
    :param planning_pipeline: Planning pipeline to use (e.g. "ompl", "pilz_industrial_motion_planner"). Empty (default) uses move_group's default pipeline. Validated against the pipelines advertised by move_group at activation.
    :type planning_pipeline: str
    :param planner_id: Planner algorithm within the pipeline (e.g. "RRTConnect", "RRTstar" for OMPL). Empty (default) uses the pipeline's default planner. Validated against the planners advertised by move_group at activation.
    :type planner_id: str
    :param num_planning_attempts: Number of planning attempts before the best solution is returned. Default is 5.
    :type num_planning_attempts: int
    :param allowed_planning_time: Maximum planning time in seconds. Default is 5.0.
    :type allowed_planning_time: float
    :param max_velocity_scaling: Fraction of the joint velocity limits used when timing trajectories, in (0, 1]. Default is 0.1 — deliberately slow; increase once a setup is trusted.
    :type max_velocity_scaling: float
    :param max_acceleration_scaling: Fraction of the joint acceleration limits used when timing trajectories, in (0, 1]. Default is 0.1.
    :type max_acceleration_scaling: float
    :param goal_position_tolerance: Position tolerance in meters for pose targets. Default is 1e-3.
    :type goal_position_tolerance: float
    :param goal_orientation_tolerance: Orientation tolerance in radians for pose targets — a single value applied to all axes, or a list of 3 per-axis values (x, y, z). Relaxing a single axis (e.g. z to ~3.14) makes many poses reachable for underactuated (e.g. 5-DOF) arms. Default is 1e-2.
    :type goal_orientation_tolerance: Union[float, List[float]]
    :param goal_joint_tolerance: Position tolerance in radians for joint targets. Default is 1e-3.
    :type goal_joint_tolerance: float
    :param cartesian_max_step: End-effector interpolation step in meters for Cartesian paths. Default is 0.0025.
    :type cartesian_max_step: float
    :param cartesian_jump_threshold: Maximum allowed joint-space jump between consecutive Cartesian points (0.0 disables the check). Default is 0.0.
    :type cartesian_jump_threshold: float
    :param cartesian_avoid_collisions: Whether Cartesian paths must avoid collisions. Default is True.
    :type cartesian_avoid_collisions: bool
    :param cartesian_fraction_threshold: Minimum fraction of the requested Cartesian path that must be achievable for the goal to be executed, in [0, 1]. Default is 0.95.
    :type cartesian_fraction_threshold: float
    :param gripper_mode: How the gripper is controlled: "move_group" (default) sends named targets on `gripper_group_name`; "gripper_command" sends a control_msgs GripperCommand to `gripper_command_action`.
    :type gripper_mode: Literal["move_group", "gripper_command"]
    :param gripper_command_action: Action name of the gripper controller (e.g. "/gripper_controller/gripper_cmd"). Required when `gripper_mode` is "gripper_command".
    :type gripper_command_action: str
    :param gripper_open_target: SRDF named target used by `open_gripper` in "move_group" mode. Default is "open".
    :type gripper_open_target: str
    :param gripper_close_target: SRDF named target used by `close_gripper` in "move_group" mode. Default is "close".
    :type gripper_close_target: str
    :param gripper_open_position: Gripper position used by `open_gripper` in "gripper_command" mode. Default is 0.04.
    :type gripper_open_position: float
    :param gripper_close_position: Gripper position used by `close_gripper` in "gripper_command" mode. Default is 0.0.
    :type gripper_close_position: float
    :param gripper_max_effort: Maximum effort for GripperCommand goals (0.0 lets the controller decide). Default is 0.0.
    :type gripper_max_effort: float
    :param move_group_namespace: Namespace prefix of the move_group interfaces (e.g. "/my_robot" if the actions are at "/my_robot/move_action"). Default is "".
    :type move_group_namespace: str
    :param move_group_node_name: Name of the move_group node, used to fetch the SRDF (named targets) from its parameters. Default is "move_group".
    :type move_group_node_name: str
    :param named_targets: Manual named-target definitions per group, overriding the SRDF: {group: {target_name: {joint: position}}}. Default is None.
    :type named_targets: Optional[Dict]
    :param srdf_file: Path or URL of a local SRDF file used as fallback for named targets when the SRDF cannot be fetched from move_group. Default is None.
    :type srdf_file: Optional[str]
    :param server_timeout: Time in seconds to wait for move_group's servers to become available. Default is 30.0.
    :type server_timeout: float
    :param execution_timeout: Maximum wall time in seconds for a single plan+execute goal. Default is 120.0.
    :type execution_timeout: float
    :param scene_update_mode: WHEN detected objects are pushed into the planning scene: "manual" (default) only on the `update_planning_scene` component action, "on_goal" additionally refreshes the scene right before each motion goal is planned, "continuous" keeps refreshing at `scene_update_rate` while detections arrive. Only effective when the component is given a Detections3D input topic. In every mode the scene freezes while an object is held. The first refresh after release reconciles the scene.
    :type scene_update_mode: Literal["manual", "on_goal", "continuous"]
    :param scene_update_rate: Scene refreshes per second in "continuous" mode. Default is 1.0.
    :type scene_update_rate: float
    :param scene_detection_labels: Detection labels allowed into the planning scene. Default is None, which admits every label.
    :type scene_detection_labels: Optional[List[str]]
    :param scene_object_ttl: Seconds a detection-sourced object stays in the scene after the detector stops reporting it, before a refresh removes it. Bridges detection dropouts without keeping ghost obstacles around. Default is 5.0.
    :type scene_object_ttl: float
    :param object_padding: Margin in meters added on every side of detected objects, for planning clearance around imperfectly measured geometry. Default is 0.0.
    :type object_padding: float
    :param min_object_thickness: Floor in meters for each extent of a detected object. Surfaces seen head-on are lifted with no measurable extent along the view axis, and a zero-thickness box is invisible to collision checking. Default is 0.01.
    :type min_object_thickness: float
    :param touch_links: Links allowed to stay in contact with an attached object, e.g. the gripper's links. Default is None, which resolves them from the robot SRDF at attach time.
    :type touch_links: Optional[List[str]]
    :param approach_clearance: Height in meters above a pick or place target at which the collision-aware approach motion ends and the straight-line descent begins; also the height objects are lifted or retreated to. Default is 0.1.
    :type approach_clearance: float
    :param target_match_radius: How far in meters a scene object may be from a pick goal's target_pose and still be taken as the intended target. Tighten for dense scenes, loosen for coarse target coordinates such as language-model guesses. Default is 0.2.
    :type target_match_radius: float

    Example of usage:
    ```python
    config = MoveItConfig(arm_group_name="panda_arm", gripper_group_name="hand")
    ```
    """

    arm_group_name: str = field()
    gripper_group_name: Optional[str] = field(default=None)
    cartesian_group_name: Optional[str] = field(default=None)
    end_effector_link: str = field(default="")
    pose_reference_frame: str = field(default="")
    planning_pipeline: str = field(default="")
    planner_id: str = field(default="")
    num_planning_attempts: int = field(default=5, validator=base_validators.gt(0))
    allowed_planning_time: float = field(
        default=5.0, validator=base_validators.gt(0.0)
    )
    max_velocity_scaling: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-3, max_value=1.0)
    )
    max_acceleration_scaling: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-3, max_value=1.0)
    )
    goal_position_tolerance: float = field(
        default=1e-3, validator=base_validators.gt(0.0)
    )
    goal_orientation_tolerance: Union[float, List[float]] = field(default=1e-2)
    goal_joint_tolerance: float = field(default=1e-3, validator=base_validators.gt(0.0))
    cartesian_max_step: float = field(default=0.0025, validator=base_validators.gt(0.0))
    cartesian_jump_threshold: float = field(default=0.0)
    cartesian_avoid_collisions: bool = field(default=True)
    cartesian_fraction_threshold: float = field(
        default=0.95, validator=base_validators.in_range(min_value=0.0, max_value=1.0)
    )
    gripper_mode: Literal["move_group", "gripper_command"] = field(
        default="move_group",
        validator=base_validators.in_(["move_group", "gripper_command"]),
    )
    gripper_command_action: str = field(default="")
    gripper_open_target: str = field(default="open")
    gripper_close_target: str = field(default="close")
    gripper_open_position: float = field(default=0.04)
    gripper_close_position: float = field(default=0.0)
    gripper_max_effort: float = field(default=0.0)
    move_group_namespace: str = field(default="")
    move_group_node_name: str = field(default="move_group")
    named_targets: Optional[Dict] = field(default=None)
    srdf_file: Optional[str] = field(default=None)
    server_timeout: float = field(default=30.0, validator=base_validators.gt(0.0))
    execution_timeout: float = field(default=120.0, validator=base_validators.gt(0.0))
    scene_update_mode: Literal["manual", "on_goal", "continuous"] = field(
        default="manual",
        validator=base_validators.in_(["manual", "on_goal", "continuous"]),
    )
    scene_update_rate: float = field(default=1.0, validator=base_validators.gt(0.0))
    scene_detection_labels: Optional[List[str]] = field(default=None)
    scene_object_ttl: float = field(default=5.0, validator=base_validators.gt(0.0))
    object_padding: float = field(
        default=0.0, validator=base_validators.in_range(min_value=0.0, max_value=1.0)
    )
    min_object_thickness: float = field(default=0.01, validator=base_validators.gt(0.0))
    touch_links: Optional[List[str]] = field(default=None)
    approach_clearance: float = field(default=0.1, validator=base_validators.gt(0.0))
    target_match_radius: float = field(default=0.2, validator=base_validators.gt(0.0))

    @goal_orientation_tolerance.validator
    def _check_orientation_tolerance(self, _, value):
        """Orientation tolerance validator"""
        if isinstance(value, (int, float)):
            if value <= 0:
                raise ValueError("goal_orientation_tolerance must be greater than 0")
            return
        if len(value) != 3 or any(v <= 0 for v in value):
            raise ValueError(
                "goal_orientation_tolerance must be a single positive value or "
                "a list of 3 positive per-axis (x, y, z) values"
            )

    @gripper_command_action.validator
    def _check_gripper_command_action(self, _, value):
        """Gripper command action validator.

        Defined on gripper_command_action (declared after gripper_mode)."""
        if self.gripper_mode == "gripper_command" and not value:
            raise ValueError(
                "gripper_command_action must be set when gripper_mode is "
                "'gripper_command' (e.g. '/gripper_controller/gripper_cmd')"
            )


@define(kw_only=True)
class VisionConfig(ModelComponentConfig):
    """Configuration for a detection component.

       The config allows you to customize the detection and/or tracking process.

       :param threshold: The confidence threshold for object detection, ranging from 0.1 to 1.0 (default: 0.5).
       :type threshold: float
       :param get_dataset_labels: Whether to return data labels along with detections (default: True).
       :type get_dataset_labels: bool
       :param labels_to_track: A list of specific labels to track, when the model is used as a tracker (default: None).
       :type labels_to_track: Optional[list]
    :param enable_visualization: Whether to enable visualization of detections (default: False). Useful for testing vision component output.
       :type enable_visualization: Optional[bool]
       :param enable_local_classifier: Whether to enable a local classifier model for detections (default: False). If a model client is given to the component, than this has no effect.
       :type enable_local_classifier: bool
       :param input_height: Height of the input to local classifier model in pixels (default: 640). This parameter is only effective when enable_local_classifier is set to True.
       :type input_height: int
       :param input_width: Width of the input to local classifier in pixels (default: 640). This parameter is only effective when enable_local_classifier is set to True.
       :type input_width: int
       :param dataset_labels: A dictionary mapping label indices to names, used to interpret model outputs (default: COCO labels). This parameter is only effective when enable_local_classifier is set to True.
       :type dataset_labels: Dict
       :param device_local_classifier: Device to run the local classifier on, either "cpu" or "gpu" (default: "gpu"). This parameter is only effective when enable_local_classifier is set to True.
       :type device_local_classifier: str
       :param ncpu_local_classifier: Number of CPU cores to allocate to the local classifier when using CPU (default: 1). This parameter is only effective when enable_local_classifier is set to True.
       :type ncpu_local_classifier: int
       :param local_classifier_model_path: Path or URL to the ONNX model used by the local classifier (default: DEIM, Huang et al. CVPR 2025). Other models based on [DEIM](https://github.com/ShihuaHuang95/DEIM?tab=readme-ov-file#deim-d-fine) can be checked [here](https://github.com/automatika-robotics/embodied-agents/releases/tag/0.3.3). This parameter is only effective when enable_local_classifier is set to True.
       :type local_classifier_model_path: str
       :param detections_frame: Frame that 3D detections are published in, usually the frame the consumer plans in, e.g. "base_link". Boxes are axis aligned in this frame and it is chosen before they are measured, so it cannot be changed after the fact. Required when a Detections3D output topic is given.
       :type detections_frame: str
       :param static_camera_tf: Whether the transform from the camera to `detections_frame` is fixed, which it is for a camera bolted to the robot. Set False for a camera that moves with a moving joint, so the transform keeps being looked up. Default is True.
       :type static_camera_tf: bool
       :param depth_scale: Multiplier from the depth image's units to millimeters, overriding what its encoding implies. Default is None, which derives it from the encoding.
       :type depth_scale: Optional[float]
       :param min_depth: Closest depth reading to treat as usable, in meters. Default is 0.05.
       :type min_depth: float
       :param max_depth: Furthest depth reading to treat as usable, in meters. Default is 5.0.
       :type max_depth: float
       :param max_depth_age: How far apart in time the color and depth frames may be before the pair is treated as mismatched, in seconds. Only applies to depth arriving on its own topic, since an RGBD frame carries both halves together. Default is 0.2.
       :type max_depth_age: float
       :param min_depth_validity: Minimum fraction of usable depth pixels inside a detection's 2D box for its 3D box to be published, in [0, 1]. A box built from a handful of readings is not a shape, and a planner would take it for one. Default is 0.1.
       :type min_depth_validity: float

       Example of usage:
       ```python
       config = DetectionConfig(threshold=0.3)
       ```
    """

    threshold: float = field(
        default=0.5, validator=base_validators.in_range(min_value=0.1, max_value=1.0)
    )
    get_dataset_labels: bool = field(default=True)
    labels_to_track: Optional[List[str]] = field(default=None)
    enable_visualization: Optional[bool] = field(default=False)
    enable_local_classifier: bool = field(default=False)
    input_height: int = field(default=640)
    input_width: int = field(default=640)
    dataset_labels: Optional[Dict] = field(default=None)
    device_local_classifier: Literal["cpu", "cuda", "tensorrt"] = field(
        default="cuda", validator=base_validators.in_(["cpu", "cuda", "tensorrt"])
    )
    ncpu_local_classifier: int = field(default=1)
    local_classifier_model_path: str = field(
        default="https://github.com/automatika-robotics/embodied-agents/releases/download/0.3.3/deim_dfine_hgnetv2_n_coco_160e.onnx"
    )
    # NOTE: 3D lift fields, kept identical to MLLMConfig
    detections_frame: str = field(default="")
    static_camera_tf: bool = field(default=True)
    depth_scale: Optional[float] = field(default=None)
    min_depth: float = field(default=0.05, validator=base_validators.gt(0.0))
    max_depth: float = field(default=5.0, validator=base_validators.gt(0.0))
    max_depth_age: float = field(default=0.2, validator=base_validators.gt(0.0))
    min_depth_validity: float = field(
        default=0.1, validator=base_validators.in_range(min_value=0.0, max_value=1.0)
    )
    # serialized topics
    _depth_topic: Optional[Topic] = field(
        default=None, converter=_get_optional_topic, alias="_depth_topic"
    )
    _camera_info_topic: Optional[Topic] = field(
        default=None, converter=_get_optional_topic, alias="_camera_info_topic"
    )

    @max_depth.validator
    def _check_depth_range(self, _, value):
        """A sensor cannot see further than it can see"""
        if value <= self.min_depth:
            raise ValueError(
                f"max_depth ({value}) must be greater than min_depth ({self.min_depth})"
            )

    def _get_inference_params(self) -> Dict:
        """get_inference_params.
        :rtype: dict
        """
        return {
            "threshold": self.threshold,
            "get_dataset_labels": self.get_dataset_labels,
            "labels_to_track": self.labels_to_track,
        }


@define(kw_only=True)
class TextToSpeechConfig(ModelComponentConfig):
    """Configuration for a Text-To-Speech component.

    This class defines the configuration options for a Text-To-Speech component.

    :param enable_local_model: Whether to enable a local TTS model via ``sherpa-onnx`` (Kyutai's Pocket TTS by default), allowing the component to run without a remote model client. Requires the ``sherpa-onnx`` pip package. Default is False.
    :type enable_local_model: bool
    :param device_local_model: Device to run the local model on, either "cpu" or "cuda" (default: "cuda"). This parameter is only effective when ``enable_local_model`` is True.
    :type device_local_model: str
    :param ncpu_local_model: Number of CPU cores to allocate to the local model when using CPU (default: 1). This parameter is only effective when ``enable_local_model`` is True.
    :type ncpu_local_model: int
    :param local_model_path: HuggingFace repository ID for a sherpa-onnx compatible TTS model (default: ``csukuangfj2/sherpa-onnx-pocket-tts-int8-2026-01-26``, Kyutai's Pocket TTS), or a path to a local directory containing an already-downloaded bundle. For available models see https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html. This parameter is only effective when ``enable_local_model`` is True.
    :type local_model_path: Optional[str]
    :param speaker_id: Voice index used by multi-voice local models (e.g. Kokoro ships several voices; single-voice models ignore it). Only effective when ``enable_local_model`` is True. Default is 0.
    :type speaker_id: int
    :param local_model_options: Additional options for the local model, validated at load time against the fields of the detected sherpa-onnx model family (e.g. ``length_scale``, ``noise_scale`` (vits/matcha), ``lang`` (kokoro), ``voice_style`` (supertonic), ``guidance_scale`` (zipvoice)) and the top-level sherpa-onnx TTS options (``silence_scale``, ``max_num_sentences``, ``rule_fsts``, ``rule_fars``). Voice-prompted families (pocket, zipvoice) also accept generation options: ``voice`` (a wav file path for voice cloning, or a voice name shipped in the bundle — Pocket TTS bundles include several; defaults to the bundle's first voice), ``num_steps``, ``reference_text`` and ``max_reference_audio_len``. An unknown key raises an error listing the valid keys for the detected family. The reserved key ``model_type`` forces the model family instead of detecting it from the bundle contents. Only effective when ``enable_local_model`` is True. Default is ``{}``.
    :type local_model_options: Dict
    :param play_on_device: Whether to play the audio on available audio device (default: False).
    :type play_on_device: bool
    :param device: Optional device id (int) for playing the audio. Only effective if play_on_device is True (default: None).
    :type device: int
    :param stream_to_ip: If set, streams the audio to this IP address via UDP instead of playing locally. Requires `play_on_device` to be True.
    :type stream_to_ip: Optional[str]
    :param stream_to_port: The target port for UDP streaming. Must be set if `stream_to_ip` is set.
    :type stream_to_port: Optional[int]
    :param buffer_size: Size of the buffer for playing audio on device. Only effective if play_on_device is True (default: 20).
    :type buffer_size: int
    :param block_size: Size of the audio block to be read for playing audio on device. Only effective if play_on_device is True (default: 4096).
    :type block_size: int
    :param thread_shutdown_timeout: Timeout to shutdown a playback thread, if data is not received for more than a certain number of seconds. Only effective if play_on_device is True (default: 5 seconds).
    :type thread_shutdown_timeout: int
    :param stream: Stream output audio in chunks. With a WebSocketClient, chunks are streamed by the server; with a local model, audio chunks are yielded as the model synthesizes them (all sherpa-onnx families support this). Useful for playing audio while long text is still being synthesized. (default: True).
    :type stream: bool

    Example of usage for local playback:
    ```python
    config = TextToSpeechConfig(play_on_device=True)
    ```

    Example of usage for UDP streaming:
    ```python
    config = TextToSpeechConfig(play_on_device=True, stream_to_ip="192.168.1.100", stream_to_port=12345)
    ```

    Example of usage with local model:
    ```python
    config = TextToSpeechConfig(enable_local_model=True, play_on_device=True)
    ```
    """

    enable_local_model: bool = field(default=False)
    device_local_model: Literal["cpu", "cuda"] = field(
        default="cuda", validator=base_validators.in_(["cpu", "cuda"])
    )
    ncpu_local_model: int = field(default=1)
    local_model_path: Optional[str] = field(
        default="csukuangfj2/sherpa-onnx-pocket-tts-int8-2026-01-26"
    )
    speaker_id: int = field(default=0, validator=base_validators.gt(-1))
    local_model_options: Dict = field(default=Factory(dict))
    play_on_device: bool = field(default=False)
    device: Optional[int] = field(default=None)
    stream_to_ip: Optional[str] = field(default=None)
    stream_to_port: Optional[int] = field(default=None)
    buffer_size: int = field(default=20)
    block_size: int = field(default=4096)
    thread_shutdown_timeout: int = field(default=5)
    stream: bool = field(default=True)
    _get_bytes: bool = field(default=False, alias="_get_bytes")

    @stream_to_ip.validator
    def _check_stream_to_ip(self, _, value):
        """Stream to IP validator"""
        if value and not self.play_on_device:
            raise ValueError(
                "play_on_device must be set to True when stream_to_ip and stream_to_port are set."
            )
        if value and not self.stream_to_port:
            raise ValueError(
                "stream_to_ip is set, but stream_to_port is not. stream_to_port must be set."
            )

    @stream_to_port.validator
    def _check_stream_to_port(self, _, value):
        """Stream to Port validator"""
        if value and not self.play_on_device:
            raise ValueError(
                "play_on_device must be set to True when stream_to_ip and stream_to_port are set."
            )
        if value and not self.stream_to_ip:
            raise ValueError(
                "stream_to_port is set, but stream_to_ip is not. stream_to_ip must be set."
            )

    def _get_inference_params(self) -> Dict:
        """get_inference_params.
        :rtype: dict
        """
        return {"get_bytes": self._get_bytes}


@define(kw_only=True)
class SpeechToTextConfig(ModelComponentConfig):
    """
    Configuration for a Speech-To-Text component.

    This class defines the configuration options for speech transcription, voice activity detection,
    wakeword detection, and audio streaming.

    --
    Local Model
    --
    :param enable_local_model: Whether to enable a local STT model via ``sherpa-onnx`` (NVIDIA Parakeet TDT 0.6B by default), allowing the component to run without a remote model client. Requires the ``sherpa-onnx`` pip package. Default is False.
    :type enable_local_model: bool
    :param device_local_model: Device to run the local model on, either "cpu" or "cuda" (default: "cuda"). This parameter is only effective when ``enable_local_model`` is True.
    :type device_local_model: str
    :param ncpu_local_model: Number of CPU cores to allocate to the local model when using CPU (default: 1). This parameter is only effective when ``enable_local_model`` is True.
    :type ncpu_local_model: int
    :param local_model_path: HuggingFace repository ID for a sherpa-onnx compatible STT model (default: ``csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8``, NVIDIA Parakeet TDT), or a path to a local directory containing an already downloaded model. For available models see https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html. This parameter is only effective when ``enable_local_model`` is True.
    :type local_model_path: Optional[str]
    :param local_model_options: Additional options for the local model, validated at load time against the detected sherpa-onnx model family's loader signature (e.g. ``decoding_method``, ``hotwords_file``, ``hotwords_score`` for transducers; ``task`` for whisper; ``use_itn`` for sense_voice). An unknown key raises an error listing the valid keys for the detected family. The reserved key ``model_type`` forces the model family instead of detecting it from the bundle contents. Only effective when ``enable_local_model`` is True. Default is ``{}``.
    :type local_model_options: Dict

    --
    Transcription
    --
    :param initial_prompt: Optional initial prompt to guide transcription (e.g. speaker name or topic).
                           Defaults to None.
    :type initial_prompt: str or None

    :param language: Language code for transcription (e.g. "en", "zh"). Must be one of the supported language codes.
                     Defaults to "en".
    :type language: str

    :param max_new_tokens: Maximum number of tokens to generate. If None, no limit is applied.
                           Defaults to None.
    :type max_new_tokens: int or None

    --
    Voice Activity Detection (VAD)
    --
    :param enable_vad: Enable VAD to detect when speech is present in audio input.
                       Requires onnxruntime and silero-vad model.
                       Defaults to False.
    :type enable_vad: bool

    :param device_audio: Audio input device ID. Only used if `enable_vad` is True.
                         Defaults to None.
    :type device_audio: Optional[int]

    :param vad_threshold: Threshold above which speech is considered present.
                          Only used if `enable_vad` is True. Range: 0.0–1.0.
                          Defaults to 0.5.
    :type vad_threshold: float

    :param min_silence_duration_ms: Minimum silence duration (ms) before it's treated as a pause.
                                    Only used if `enable_vad` is True.
                                    Defaults to 300.
    :type min_silence_duration_ms: int

    :param speech_pad_ms: Silence padding (ms) added to start and end of detected speech regions.
                          Only used if `enable_vad` is True.
                          Defaults to 30.
    :type speech_pad_ms: int

    :param speech_buffer_max_len: Max length of speech buffer in ms.
                                  Only used if `enable_vad` is True.
                                  Defaults to 30000.
    :type speech_buffer_max_len: int

    :param device_vad: Device for VAD ('cpu' or 'gpu').
                       Only used if `enable_vad` is True.
                       Defaults to 'cpu'.
    :type device_vad: str

    :param ncpu_vad: Number of CPU cores to use for VAD (if `device_vad` is 'cpu').
                     Defaults to 1.
    :type ncpu_vad: int

    --
    Wakeword Detection
    --
    :param enable_wakeword: Enable detection of a wake phrase before transcription.
                            Requires `enable_vad` to be True and the
                            ``sentencepiece`` package for encoding the phrase.
                            Defaults to False.
    :type enable_wakeword: bool

    :param wakeword_phrase: The wake phrase (or list of phrases) to detect,
                            as plain text (e.g. 'hey jarvis', 'ok robot').
                            Only used if `enable_wakeword` is True.
                            Defaults to 'ok robot'.
    :type wakeword_phrase: Union[str, List[str]]

    :param wakeword_threshold: Keyword spotting trigger threshold (sherpa-onnx
                               `keywords_threshold`). Lower values trigger more
                               easily. Only used if `enable_wakeword` is True.
                               Defaults to 0.25.
    :type wakeword_threshold: float

    :param device_wakeword: Device for Wakeword Detection ('cpu' or 'gpu').
                             Only used if `enable_wakeword` is True.
                             Defaults to 'cpu'.
    :type device_wakeword: str

    :param ncpu_wakeword: Number of CPU cores for Wakeword Detection (if `device_wakeword` is 'cpu').
                          Defaults to 1.
    :type ncpu_wakeword: int

    --
    Streaming
    --
    :param stream: Send audio as a stream to a persistent client (e.g., websockets).
                   Requires `enable_vad` to be True.
                   Useful for real-time transcription.
                   Defaults to False.
    :type stream: bool

    :param min_chunk_size: Audio chunk size in ms to send when streaming.
                       Requires `stream` to be True. Must be > 100 ms.
                       Defaults to 2000.
    :type min_chunk_size: int

    --
    Model Paths
    --
    :param vad_model_path: Path or URL to VAD ONNX model.
                           Defaults to the Silero VAD model URL.
    :type vad_model_path: str

    :param wakeword_model_path: Source of the sherpa-onnx keyword spotting bundle:
                                a model archive URL (.tar.bz2/.tar.gz), a HuggingFace
                                repository ID, or a local directory. Defaults to the
                                official English zipformer KWS bundle (3.3M params)
                                from the sherpa-onnx releases. For other languages
                                see https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models
    :type wakeword_model_path: str

    --
    Example
    --
    Example usage:
    ```python
    config = SpeechToTextConfig(
        enable_vad=True,
        enable_wakeword=True,
        vad_threshold=0.5,
        wakeword_threshold=0.6,
        min_silence_duration_ms=1000,
        speech_pad_ms=30,
        speech_buffer_max_len=8000,
    )
    ```

    Example of usage with local model:
    ```python
    config = SpeechToTextConfig(enable_local_model=True, enable_vad=True)
    ```
    """

    enable_local_model: bool = field(default=False)
    device_local_model: Literal["cpu", "cuda"] = field(
        default="cuda", validator=base_validators.in_(["cpu", "cuda"])
    )
    ncpu_local_model: int = field(default=1)
    local_model_path: Optional[str] = field(
        default="csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
    )
    local_model_options: Dict = field(default=Factory(dict))
    initial_prompt: Optional[str] = field(default=None)
    language: Optional[str] = field(
        default="en",
        validator=validators.optional(base_validators.in_(_LANGUAGE_CODES)),
    )
    max_new_tokens: Optional[int] = field(default=None)
    enable_vad: bool = field(default=False)
    enable_wakeword: bool = field(default=False)
    device_audio: Optional[int] = field(default=None)
    vad_threshold: float = field(
        default=0.5, validator=base_validators.in_range(min_value=0.0, max_value=1.0)
    )
    wakeword_threshold: float = field(
        default=0.25, validator=base_validators.in_range(min_value=0.0, max_value=1.0)
    )
    wakeword_phrase: Union[str, List[str]] = field(default="ok robot")
    min_silence_duration_ms: int = field(default=500)
    speech_pad_ms: int = field(default=30)
    speech_buffer_max_len: int = field(default=30000)
    stream: bool = field(default=False)
    min_chunk_size: int = field(default=2000, validator=base_validators.gt(500))
    device_vad: Literal["cpu", "cuda", "tensorrt"] = field(
        default="cpu", validator=base_validators.in_(["cpu", "cuda", "tensorrt"])
    )
    device_wakeword: Literal["cpu", "cuda", "tensorrt"] = field(
        default="cpu", validator=base_validators.in_(["cpu", "cuda", "tensorrt"])
    )
    ncpu_vad: int = field(default=1)
    ncpu_wakeword: int = field(default=1)
    vad_model_path: str = field(
        default="https://raw.githubusercontent.com/snakers4/silero-vad/refs/heads/master/src/silero_vad/data/silero_vad.onnx"
    )
    wakeword_model_path: str = field(
        default="https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2"
    )
    _sample_rate: int = field(default=16000, alias="_sample_rate")
    _block_size: int = field(default=1280, alias="_block_size")
    _vad_filter: bool = field(init=False, alias="_vad_filter")
    _word_timestamps: bool = field(init=False, alias="_word_timestamps")

    @enable_wakeword.validator
    def _check_wakeword(self, _, value):
        """Wakeword validator"""
        if value and not self.enable_vad:
            raise ValueError(
                "enable_vad (voice activity detection) must be set to True when enable_wakeword is set to True"
            )

    @stream.validator
    def _check_stream(self, _, value):
        """Stream validator"""
        if value and not self.enable_vad:
            raise ValueError(
                "enable_vad (voice activity detection) must be set to True when stream is set to True"
            )

    def __attrs_post_init__(self):
        """Set values of undefined privates"""
        self._word_timestamps = self.stream
        self._vad_filter = not self.enable_vad

    def _get_inference_params(self) -> Dict:
        """get_inference_params.
        :rtype: dict
        """
        return {
            "language": self.language,
            "initial_prompt": self.initial_prompt,
            "max_new_tokens": self.max_new_tokens,
            "word_timestamps": self._word_timestamps,
            "vad_filter": self._vad_filter,
        }


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
    distance_func: Literal["l2", "ip", "cosine"] = field(
        default="l2", validator=base_validators.in_(["l2", "ip", "cosine"])
    )
    _position: Optional[Topic] = field(
        default=None, converter=_get_optional_topic, alias="_position"
    )
    _map_topic: Optional[Topic] = field(
        default=None, converter=_get_optional_topic, alias="_map_topic"
    )


@define(kw_only=True)
class MemoryConfig(BaseComponentConfig):
    """Configuration for the Memory component.

    :param db_path: Path to the eMEM SQLite database file.
    :type db_path: str
    :param embedding_checkpoint: Model name for sentence-transformers fallback. Only used when no ``embedding_client`` is provided to the Memory component.
    :type embedding_checkpoint: str
    :param auto_store: Automatically store layer data on each execution step. If False, storage only happens via the ``store`` component action.
    :type auto_store: bool
    :param working_memory_size: Max observations held in the in-process buffer before the oldest are dropped. Observations are flushed to persistent storage well before this limit via ``flush_batch_size`` and ``flush_interval``.
    :type working_memory_size: int
    :param flush_interval: Seconds between auto-flushes of the working memory buffer to persistent storage. Lower values mean observations become searchable faster but increase write frequency.
    :type flush_interval: float
    :param flush_batch_size: Number of observations accumulated before an automatic flush is triggered, regardless of ``flush_interval``.
    :type flush_batch_size: int
    :param consolidation_window: Maximum temporal gap in seconds between consecutive observations within the same consolidation chunk. When an episode is consolidated, observations separated by more than this gap produce separate gist summaries. For example, 1800 (30 min) means a multi-session episode spanning days will get one gist per session, not one monolithic summary.
    :type consolidation_window: float
    :param consolidation_spatial_eps: DBSCAN epsilon in meters for spatial clustering during time-window consolidation. Observations farther apart than this are placed in separate clusters and produce separate gists. Only applies to non-episodic consolidation.
    :type consolidation_spatial_eps: float
    :param consolidation_min_samples: Minimum number of observations required to form a spatial cluster during time-window consolidation. Clusters smaller than this are left in short-term memory.
    :type consolidation_min_samples: int
    :param archive_after_seconds: How long (in seconds) observations remain in long-term memory (with full text preserved) before archival drops their text and embeddings, leaving only the gist searchable. Set higher to keep raw observations searchable longer at the cost of storage.
    :type archive_after_seconds: float
    :param entity_extract_flush_interval: Trigger entity extraction every N working-memory flushes.
    :type entity_extract_flush_interval: int
    :param entity_extract_time_interval: Trigger entity extraction every N seconds, whichever comes first with ``entity_extract_flush_interval``.
    :type entity_extract_time_interval: float
    :param entity_similarity_threshold: Cosine similarity threshold (0-1) for merging a newly detected entity with an existing one. Higher values require closer name matches before merging (e.g. 0.85 means "red chair" and "chair" may merge, but "chair" and "table" won't).
    :type entity_similarity_threshold: float
    :param entity_spatial_radius: Maximum distance in meters between an existing entity and a new detection for them to be considered the same object. Only entities within this radius AND above the similarity threshold are merged.
    :type entity_spatial_radius: float
    :param recency_weight: Alpha multiplier for recency-weighted semantic search. When > 0, recent observations are boosted over older ones at equal semantic distance. Set to 0.0 (default) for pure semantic ordering.
    :type recency_weight: float
    :param recency_halflife: Time constant in seconds for recency decay. An observation this many seconds old receives half the recency boost. Only effective when ``recency_weight`` > 0.
    :type recency_halflife: float
    :param hnsw_ef_construction: HNSW index build-time quality parameter. Higher values produce a better quality index but take longer to build. Default (200) is suitable for most use cases.
    :type hnsw_ef_construction: int
    :param hnsw_m: Number of bidirectional links per node in the HNSW graph. Higher values improve recall but increase memory usage. Default (16) is suitable for most use cases.
    :type hnsw_m: int
    :param hnsw_ef_search: HNSW search-time quality parameter. Higher values improve recall at the cost of query latency. Default (50) is suitable for most use cases.
    :type hnsw_ef_search: int
    :param hnsw_max_elements: Maximum number of vectors the HNSW index can hold. Should be set higher than the expected total number of observations + gists + entities over the system's lifetime.
    :type hnsw_max_elements: int

    Example of usage:
    ```python
    config = MemoryConfig(db_path="/tmp/robot_memory.db")
    ```
    """

    # Memory component specific
    db_path: str = field(default="memory.db")
    embedding_checkpoint: str = field(default="all-MiniLM-L6-v2")
    auto_store: bool = field(default=True)
    # eMEM config parameters (mirrors emem.SpatioTemporalMemoryConfig)
    working_memory_size: int = field(default=50, validator=base_validators.gt(0))
    flush_interval: float = field(default=2.0, validator=base_validators.gt(0.0))
    flush_batch_size: int = field(default=5, validator=base_validators.gt(0))
    consolidation_window: float = field(
        default=1800.0, validator=base_validators.gt(0.0)
    )
    consolidation_spatial_eps: float = field(
        default=3.0, validator=base_validators.gt(0.0)
    )
    consolidation_min_samples: int = field(default=3, validator=base_validators.gt(0))
    archive_after_seconds: float = field(
        default=3600.0, validator=base_validators.gt(0.0)
    )
    entity_extract_flush_interval: int = field(
        default=10, validator=base_validators.gt(0)
    )
    entity_extract_time_interval: float = field(
        default=60.0, validator=base_validators.gt(0.0)
    )
    entity_similarity_threshold: float = field(
        default=0.85,
        validator=base_validators.in_range(min_value=0.0, max_value=1.0),
    )
    entity_spatial_radius: float = field(default=5.0, validator=base_validators.gt(0.0))
    recency_weight: float = field(
        default=0.0,
        validator=base_validators.in_range(min_value=0.0, max_value=1.0),
    )
    recency_halflife: float = field(default=3600.0, validator=base_validators.gt(0.0))
    hnsw_ef_construction: int = field(default=200, validator=base_validators.gt(0))
    hnsw_m: int = field(default=16, validator=base_validators.gt(0))
    hnsw_ef_search: int = field(default=50, validator=base_validators.gt(0))
    hnsw_max_elements: int = field(default=100_000, validator=base_validators.gt(0))
    # internal - serialized topic
    _position: Optional[Topic] = field(
        default=None, converter=_get_optional_topic, alias="_position"
    )


@define(kw_only=True)
class SemanticRouterConfig(ModelComponentConfig):
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
    distance_func: Literal["l2", "ip", "cosine"] = field(
        default="l2", validator=base_validators.in_(["l2", "ip", "cosine"])
    )
    maximum_distance: float = field(
        default=0.4, validator=base_validators.in_range(min_value=0.1, max_value=1.0)
    )
    _default_route: Optional[str] = field(default=None, alias="_default_route")

    def _get_inference_params(self):
        """Dummy method to avoid check if semantic router is used in vector mode"""
        return {}


@define(kw_only=True)
class MotionDetectorConfig(BaseComponentConfig):
    """Configuration parameters for a motion detection component.

    --
    Common params
    --
    :param motion_stop_delay: Number of consecutive still inputs before declaring that motion has ended. Debounces flickery detections. Default is 8.
    :type motion_stop_delay: int
    :param publish_bool_on_change_only: Publish on Bool output topics only when the motion state changes, instead of on every processed input. Default is False.
    :type publish_bool_on_change_only: bool
    :param process_rate: Optional maximum processing rate in Hz. Inputs arriving faster are dropped. Default is None (process every input).
    :type process_rate: Optional[float]
    :param device: Device for point cloud voxelization, "cpu" or "cuda". "cuda" requires torch (an error with installation instructions is raised if it is missing); if torch has no available CUDA device, processing falls back to cpu with a warning. Default is "cpu".
    :type device: str

    --
    Image input params
    --
    :param min_video_frames: The minimum number of frames in a video segment. Default is 15, assuming a 0.5 second video at 30 fps.
    :type min_video_frames: int
    :param max_video_frames: The maximum number of frames in a video segment. Default is 600, assuming a 20 second video at 30 fps.
    :type max_video_frames: int
    :param motion_estimation_func: The function used for motion estimation. Can be one of "frame_difference" or "optical_flow". Default is None.
    :type motion_estimation_func: Optional[str]
    :param threshold: The threshold value for image motion detection. A float between 0.1 and 5.0. Default is 0.3.
    :type threshold: float
    :param flow_kwargs: Additional keyword arguments for the optical flow algorithm. Default is a dictionary with reasonable values.
    :param roi_ignore_polygon: Optional polygon of (x, y) pixel coordinates to ignore during image motion estimation (e.g. a visible robot arm). Default is None.
    :type roi_ignore_polygon: Optional[List]
    :param pause_on_ego_motion: When a position (odometry) topic is provided with image inputs, suppress motion detection while the robot itself is moving. Default is True.
    :type pause_on_ego_motion: bool
    :param ego_speed_threshold: Speed (m/s) above which the robot is considered moving for ``pause_on_ego_motion``. Default is 0.05.
    :type ego_speed_threshold: float

    --
    Point cloud input params
    --
    :param voxel_size: Edge length in meters of the voxel grid used for cloud differencing. Default is 0.15.
    :type voxel_size: float
    :param changed_voxel_threshold: Number of newly appearing voxels (relative to the accumulated occupancy history) that form spatially coherent clusters of at least ``min_cluster_size`` voxels, required to declare motion. Coherence filters out scattered appearances from sensor noise or people standing quasi-still. Default is 5.
    :type changed_voxel_threshold: int
    :param accumulation_window: Number of previous clouds accumulated into the occupancy history that new clouds are differenced against. A window makes detection robust to sparse and non-repetitive scan patterns (e.g. Livox lidars) where a single previous cloud does not cover the whole scene. Detection starts once the window is full. Default is 20 (i.e. 2 seconds of history for a 10 Hz sensor).
    :type accumulation_window: int
    :param min_cluster_size: Minimum number of newly appearing voxels in a spatially connected cluster for the cluster to count as motion evidence and produce a motion center. Default is 4.
    :type min_cluster_size: int
    :param max_clusters: Maximum number of motion centers published at a time (largest clusters first). Default is 5.
    :type max_clusters: int
    :param min_range: Minimum planar (xy) range in meters of cloud points considered. Default is 0.0.
    :type min_range: float
    :param max_range: Maximum planar (xy) range in meters of cloud points considered. Default is 20.0.
    :type max_range: float
    :param z_min: Minimum height of cloud points considered. Default is -1.0 (1 meter below the sensor).
    :type z_min: float
    :param z_max: Maximum height of cloud points considered. Default is 1.0 (1 meter above the sensor).
    :type z_max: float
    :param base_frame: The robot base frame used for ego-motion subtraction. The static transform from the cloud frame to this frame (the sensor mount) is looked up from TF automatically. Default is "base_link".
    :type base_frame: str

    Example of usage:
    ```python
    config = MotionDetectorConfig()
    # or
    config = MotionDetectorConfig(min_video_frames=30, motion_estimation_func="optical_flow", threshold=0.5)
    ```
    """

    motion_stop_delay: int = field(
        default=8, validator=base_validators.in_range(min_value=0, max_value=1e3)
    )
    publish_bool_on_change_only: bool = field(default=False)
    process_rate: Optional[float] = field(default=None)
    device: Literal["cpu", "cuda"] = field(
        default="cpu", validator=base_validators.in_(["cpu", "cuda"])
    )

    min_video_frames: int = field(default=15)  # assuming 0.5 second video at 30 fps
    max_video_frames: int = field(default=600)  # assuming 20 second video at 30 fps
    motion_estimation_func: Optional[Literal["frame_difference", "optical_flow"]] = (
        field(
            default=None,
            validator=validators.optional(
                base_validators.in_(["frame_difference", "optical_flow"])
            ),
        )
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
        validator=validate_kwargs_from_default,
    )
    roi_ignore_polygon: Optional[List] = field(default=None)
    pause_on_ego_motion: bool = field(default=True)
    ego_speed_threshold: float = field(
        default=0.05, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )

    voxel_size: float = field(
        default=0.15, validator=base_validators.in_range(min_value=1e-3, max_value=1e3)
    )
    changed_voxel_threshold: int = field(
        default=5, validator=base_validators.in_range(min_value=1, max_value=1e9)
    )
    accumulation_window: int = field(
        default=20, validator=base_validators.in_range(min_value=1, max_value=1e3)
    )
    min_cluster_size: int = field(
        default=4, validator=base_validators.in_range(min_value=1, max_value=1e9)
    )
    max_clusters: int = field(
        default=5, validator=base_validators.in_range(min_value=1, max_value=1e3)
    )
    min_range: float = field(
        default=0.0, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )
    max_range: float = field(
        default=20.0, validator=base_validators.in_range(min_value=1e-3, max_value=1e3)
    )
    z_min: float = field(default=-1.0)
    z_max: float = field(default=1.0)
    base_frame: str = field(default="base_link")


# Backwards-compatible alias for the renamed config
VideoMessageMakerConfig = MotionDetectorConfig

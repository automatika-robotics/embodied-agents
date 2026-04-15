from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import json

from ..clients.model_base import ModelClient
from ..config import MemoryConfig
from ..ros import (
    Odometry,
    String,
    Event,
    Topic,
    Detections,
    DetectionsMultiSource,
    MapLayer,
    ActionPhase,
    component_action,
)
from ..utils import validate_func_args
from .component_base import Component, ComponentRunType

# Pull tool schemas from eMEM if installed, else fallback to an empty
# stub per tool so decoration still succeeds
try:
    from emem.tools import TOOL_SCHEMAS as _EMEM_TOOL_SCHEMAS
except ImportError:
    _EMEM_TOOL_SCHEMAS = {
        name: {
            "name": name,
            "description": f"{name} (eMEM not installed)",
            "parameters": {"type": "object", "properties": {}},
        }
        for name in (
            "semantic_search",
            "spatial_query",
            "temporal_query",
            "episode_summary",
            "get_current_context",
            "search_gists",
            "entity_query",
            "locate",
            "recall",
            "body_status",
        )
    }


def _tool(name: str) -> Dict[str, Any]:
    """Wrap an eMEM tool schema in OpenAI-format for ``@component_action``."""
    return {"type": "function", "function": _EMEM_TOOL_SCHEMAS[name]}


class Memory(Component):
    """Spatio-temporal memory component powered by eMEM.

    Encodes perception layer data (text descriptions from vlms, detections) into
    a graph-based spatio-temporal memory indexed by meaning, location, and time.
    Provides 10 retrieval tools as component actions and supports episode-based
    memory consolidation.

    This component uses real-world coordinates from Odometry directly
    and provides consolidation, entity tracking, and structured retrieval tools
    instead of flat vector DB storage.

    :param layers: Perception layers to encode (output topics from other components).
    :type layers: list[MapLayer]
    :param position: Odometry topic providing the robot's current position.
    :type position: Topic
    :param model_client: Model client for memory consolidation (summarization, entity extraction). If not provided, consolidation uses simple text concatenation.
    :type model_client: Optional[ModelClient]
    :param embedding_client: Model client for generating embeddings (e.g. OllamaClient with an embedding model). If not provided, falls back to sentence-transformers.
    :type embedding_client: Optional[ModelClient]
    :param config: Memory configuration.
    :type config: Optional[MemoryConfig]
    :param trigger: Trigger for the execution step (frequency in Hz, topic, or event).
    :type trigger: Union[Topic, list[Topic], float, Event]
    :param component_name: ROS node name for this component.
    :type component_name: str

    Example usage:
    ```python
    position = Topic(name="odom", msg_type="Odometry")
    detections = Topic(name="detections", msg_type="Detections")
    room_type = Topic(name="room_type", msg_type="String")

    layer1 = MapLayer(subscribes_to=detections, temporal_change=True)
    layer2 = MapLayer(subscribes_to=room_type, resolution_multiple=3)

    memory = Memory(
        layers=[layer1, layer2],
        position=position,
        model_client=llama_client,
        embedding_client=embed_client,
        config=MemoryConfig(db_path="/tmp/robot_memory.db"),
        trigger=15.0,
        component_name="memory",
    )
    ```
    """

    @validate_func_args
    def __init__(
        self,
        *,
        layers: List[MapLayer],
        position: Topic,
        model_client: Optional[ModelClient] = None,
        embedding_client: Optional[ModelClient] = None,
        config: Optional[MemoryConfig] = None,
        trigger: Union[Topic, List[Topic], float, Event] = 10.0,
        component_name: str,
        **kwargs,
    ):
        from importlib.util import find_spec

        if find_spec("emem") is None:
            raise ImportError(
                "The Memory component requires the 'emem' package. "
                "Install it with: pip install emem"
            )

        self.config: MemoryConfig = config or MemoryConfig()
        self.allowed_inputs = {
            "Required": [Odometry],
            "Optional": [String, Detections, DetectionsMultiSource],
        }

        self.model_client = model_client
        self.embedding_client = embedding_client

        self.position = position
        self.config._position = position

        super().__init__(
            None,
            None,
            self.config,
            trigger,
            component_name,
            **kwargs,
        )

        self._layers(layers)

    def custom_on_configure(self):
        """Initialize eMEM and client connections."""
        self.get_logger().debug(f"Current Status: {self.health_status.value}")
        super().custom_on_configure()

        # Initialize embedding provider
        if self.embedding_client:
            self.embedding_client.check_connection()
            self.embedding_client.initialize()
            from emem.embeddings import CallableEmbeddingProvider

            embedding_provider = CallableEmbeddingProvider(self.embedding_client._embed)
        else:
            from emem.embeddings import SentenceTransformerProvider

            embedding_provider = SentenceTransformerProvider(
                self.config.embedding_checkpoint
            )

        # Initialize LLM client for consolidation
        llm_client = None
        if self.model_client:
            self.model_client.check_connection()
            self.model_client.initialize()
            from emem.consolidation import InferenceLLMClient

            llm_client = InferenceLLMClient(self.model_client.inference)
        else:
            self.get_logger().warning(
                "No model_client provided. Memory consolidation will use "
                "simple text concatenation instead of LLM summarization."
            )

        # Build eMEM config from component config
        from emem.config import SpatioTemporalMemoryConfig

        emem_config = SpatioTemporalMemoryConfig(
            db_path=self.config.db_path,
            hnsw_path=str(Path(self.config.db_path).with_suffix(".hnsw.bin")),
            embedding_dim=embedding_provider.dim,
            working_memory_size=self.config.working_memory_size,
            flush_interval=self.config.flush_interval,
            flush_batch_size=self.config.flush_batch_size,
            consolidation_window=self.config.consolidation_window,
            consolidation_spatial_eps=self.config.consolidation_spatial_eps,
            consolidation_min_samples=self.config.consolidation_min_samples,
            archive_after_seconds=self.config.archive_after_seconds,
            entity_extract_flush_interval=self.config.entity_extract_flush_interval,
            entity_extract_time_interval=self.config.entity_extract_time_interval,
            entity_similarity_threshold=self.config.entity_similarity_threshold,
            entity_spatial_radius=self.config.entity_spatial_radius,
            recency_weight=self.config.recency_weight,
            recency_halflife=self.config.recency_halflife,
            hnsw_ef_construction=self.config.hnsw_ef_construction,
            hnsw_m=self.config.hnsw_m,
            hnsw_ef_search=self.config.hnsw_ef_search,
            hnsw_max_elements=self.config.hnsw_max_elements,
        )

        # Create eMEM instance
        from emem import SpatioTemporalMemory

        self.memory = SpatioTemporalMemory(
            config=emem_config,
            embedding_provider=embedding_provider,
            llm_client=llm_client,
        )

    def custom_on_deactivate(self):
        """Close eMEM and deinitialize clients."""
        if hasattr(self, "memory"):
            self.memory.close()
        if self.embedding_client:
            self.embedding_client.check_connection()
            self.embedding_client.deinitialize()
        if self.model_client:
            self.model_client.check_connection()
            self.model_client.deinitialize()

    def inspect_component(self) -> str:
        """Return component info including configured perception layers.

        Appends a ``Perception layers:`` section listing each configured
        MapLayer (layer name, source topic, and message type). A consumer
        like Cortex can read this to learn which layer tags observations
        get stored under. Useful when planning retrieval calls that take
        a ``layer`` filter, or when deciding whether an LLM-injected memory
        should share a tag with an existing perception stream.
        """
        result = super().inspect_component()

        if self.layers_dict:
            lines = [
                "",
                "Perception layers (layer_name → source topic):",
                (
                    "  Each layer tags observations with a label derived from "
                    "its source topic. Use these names as the 'layer' filter in "
                    "memory retrieval tools (semantic_search, spatial_query, "
                    "locate, ...) to narrow results to one stream."
                ),
            ]
            for name, layer in self.layers_dict.items():
                topic = layer.subscribes_to
                msg_name = (
                    topic.msg_type.__name__
                    if hasattr(topic.msg_type, "__name__")
                    else str(topic.msg_type)
                )
                lines.append(
                    f"  - {name}  (from topic '{topic.name}', type {msg_name})"
                )
            result += "\n".join(lines)
        else:
            result += "\nPerception layers: none configured"

        result += f"\nPosition topic: {self.position.name}"
        return result

    def _layers(self, layers: List[MapLayer]):
        """Set up perception layers and callbacks"""
        self.layers_dict = {layer.subscribes_to.name: layer for layer in layers}
        layer_topics = [layer.subscribes_to for layer in layers]
        all_inputs = [*layer_topics, self.config._position]
        self._validate_topics(all_inputs, self.allowed_inputs, "Inputs")
        self.callbacks = {
            input.name: input.msg_type.callback(input) for input in all_inputs
        }

    def _store_layers(self, position, time_stamp) -> None:
        """Store all layer data at the given position and time."""
        for name in self.layers_dict:
            if item := self.callbacks[name].get_output():
                self.memory.add(
                    text=item,
                    x=float(position[0]),
                    y=float(position[1]),
                    z=float(position[2]) if len(position) > 2 else 0.0,
                    layer_name=name,
                    timestamp=float(time_stamp),
                )

    def _execution_step(self, **kwargs):
        """Periodic execution: read position and store layer data."""
        time_stamp = self.get_ros_time().sec
        if self.run_type is ComponentRunType.EVENT:
            trigger = kwargs.get("topic")
            if trigger:
                self.get_logger().debug(f"Received trigger on {trigger.name}")
            else:
                self.get_logger().debug("Memory got triggered by an event.")
        else:
            self.get_logger().debug(f"Executing at {time_stamp}")

        if not self.config.auto_store:
            return

        position = self.callbacks[self.position.name].get_output()
        if position is None:
            self.get_logger().warning(
                "Position not received, not storing observations."
            )
            return

        self._store_layers(position[:3], time_stamp)

    ### Storage actions ###

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "store",
                "description": (
                    "Force an immediate snapshot of the latest data on every "
                    "configured perception layer, tagged with the robot's "
                    "current odometry position and current time. Normally "
                    "snapshotting happens automatically at the component's "
                    "trigger rate — call this only when you need an out-of-band "
                    "capture (e.g. right after observing something important)."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    )
    def store(self) -> None:
        """Explicitly trigger storage of current layer data."""
        position = self.callbacks[self.position.name].get_output()
        if position is None:
            return
        time_stamp = self.get_ros_time().sec
        self._store_layers(position[:3], time_stamp)

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "store_specific_memory",
                "description": (
                    "Write a piece of text into memory as a new observation. "
                    "Use this to record things that did not come in through a "
                    "perception layer, for example a scene description the "
                    "planner just produced, a decision made during a task, or "
                    "a fact the operator just told the robot. "
                    "By default the note goes into the 'agent_notes' layer, "
                    "which is separate from perception streams. "
                    "Only override 'layer_name' if you are confident the note "
                    "belongs with an existing perception layer reported by "
                    "inspect_component (e.g. if you are adding a caption to a "
                    "'scene_description' layer that already stores VLM output); "
                    "otherwise leave it at the default so hallucinated text "
                    "does not contaminate detector streams. "
                    "If coordinates are not provided, the robot's current "
                    "odometry position is used."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The text to remember.",
                        },
                        "layer_name": {
                            "type": "string",
                            "description": (
                                "Layer tag to store the note under. Defaults "
                                "to 'agent_notes'. Only change this if the "
                                "note genuinely belongs with an existing "
                                "perception layer visible via inspect_component."
                            ),
                            "default": "agent_notes",
                        },
                        "x": {
                            "type": "number",
                            "description": (
                                "X coordinate in world-frame meters. Omit to "
                                "use the robot's current position."
                            ),
                        },
                        "y": {
                            "type": "number",
                            "description": (
                                "Y coordinate in world-frame meters. Omit to "
                                "use the robot's current position."
                            ),
                        },
                        "z": {
                            "type": "number",
                            "description": (
                                "Z coordinate in meters. Omit to use the "
                                "robot's current position (or 0 for 2D maps)."
                            ),
                        },
                    },
                    "required": ["content"],
                },
            },
        }
    )
    def store_specific_memory(
        self,
        content: str,
        layer_name: str = "agent_notes",
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
    ) -> bool:
        """Store an arbitrary piece of text at a given (or current) position.

        :param content: Text to record.
        :param layer_name: Layer tag to store under. Defaults to ``agent_notes``.
        :param x: Optional X in world-frame meters. If omitted, current odometry is used.
        :param y: Optional Y in world-frame meters. If omitted, current odometry is used.
        :param z: Optional Z in meters. If omitted, current odometry is used.
        :returns: True if the note was stored, False if position was unavailable.
        """
        if x is None or y is None:
            position = self.callbacks[self.position.name].get_output()
            if position is None:
                self.get_logger().warning(
                    "store_note: no coordinates given and no odometry "
                    "available, note not stored."
                )
                return False
            px = float(position[0])
            py = float(position[1])
            pz = float(position[2]) if len(position) > 2 else 0.0
        else:
            px = float(x)
            py = float(y)
            pz = float(z) if z is not None else 0.0

        self.memory.add(
            text=content,
            x=px,
            y=py,
            z=pz,
            layer_name=layer_name,
            timestamp=float(self.get_ros_time().sec),
        )
        return True

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "start_episode",
                "description": (
                    "Open a named episode. All observations stored between now "
                    "and the matching end_episode will be grouped under this "
                    "name and later consolidated into a searchable summary. "
                    "Use this when the robot begins a discrete task or activity "
                    "the operator might ask about later (e.g. 'kitchen patrol', "
                    "'charging')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "Human-readable episode name, e.g. 'kitchen "
                                "patrol', 'delivery to Alice'."
                            ),
                        },
                    },
                    "required": ["name"],
                },
            },
        }
    )
    def start_episode(self, name: str) -> str:
        """Start a named episode."""
        return self.memory.start_episode(name)

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "end_episode",
                "description": (
                    "Close the currently active episode. This triggers memory "
                    "consolidation: the episode's observations are clustered, "
                    "summarized into gists by the LLM, and entities (objects, "
                    "people, landmarks) are extracted and deduplicated. Call "
                    "this when the corresponding task is finished."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    )
    def end_episode(self) -> str:
        """End the active episode and trigger consolidation."""
        return self.memory.end_episode() or "No active episode"

    ### Retrieval actions ###

    @component_action(description=_tool("semantic_search"), phase=ActionPhase.PLANNING)
    def semantic_search(self, **kwargs) -> str:
        """Search memory by meaning."""
        return self.memory.dispatch_tool_call("semantic_search", kwargs)

    @component_action(description=_tool("spatial_query"), phase=ActionPhase.PLANNING)
    def spatial_query(self, **kwargs) -> str:
        """Find observations within a radius of a point."""
        return self.memory.dispatch_tool_call("spatial_query", kwargs)

    @component_action(description=_tool("temporal_query"), phase=ActionPhase.PLANNING)
    def temporal_query(self, **kwargs) -> str:
        """Find observations in a time range."""
        return self.memory.dispatch_tool_call("temporal_query", kwargs)

    @component_action(description=_tool("episode_summary"), phase=ActionPhase.PLANNING)
    def episode_summary(self, **kwargs) -> str:
        """Get summary of one or more episodes."""
        return self.memory.dispatch_tool_call("episode_summary", kwargs)

    @component_action(
        description=_tool("get_current_context"), phase=ActionPhase.PLANNING
    )
    def get_current_context(self, **kwargs) -> str:
        """Get situational awareness."""
        return self.memory.dispatch_tool_call("get_current_context", kwargs)

    @component_action(description=_tool("search_gists"), phase=ActionPhase.PLANNING)
    def search_gists(self, **kwargs) -> str:
        """Search consolidated memory summaries."""
        return self.memory.dispatch_tool_call("search_gists", kwargs)

    @component_action(description=_tool("entity_query"), phase=ActionPhase.PLANNING)
    def entity_query(self, **kwargs) -> str:
        """Find known entities."""
        return self.memory.dispatch_tool_call("entity_query", kwargs)

    @component_action(description=_tool("locate"), phase=ActionPhase.PLANNING)
    def locate(self, **kwargs) -> str:
        """Find the spatial location of a concept."""
        return self.memory.dispatch_tool_call("locate", kwargs)

    @component_action(description=_tool("recall"), phase=ActionPhase.PLANNING)
    def recall(self, **kwargs) -> str:
        """Recall everything known about a concept."""
        return self.memory.dispatch_tool_call("recall", kwargs)

    @component_action(description=_tool("body_status"), phase=ActionPhase.BOTH)
    def body_status(self, **kwargs) -> str:
        """Get latest body/internal state readings."""
        return self.memory.dispatch_tool_call("body_status", kwargs)

    ###  LLM tool registration ###

    def register_tools_on(
        self,
        llm,
        tools: Optional[List[str]] = None,
        send_tool_response_to_model: bool = True,
    ) -> None:
        """Register eMEM retrieval tools on an LLM component for tool calling.

        :param llm: The LLM or Cortex component to register tools on.
        :type llm: LLM
        :param tools: Optional subset of tool names to register (default: all 10).
        :type tools: Optional[list[str]]
        :param send_tool_response_to_model: Whether tool results are sent back to the model for a follow-up response.
        :type send_tool_response_to_model: bool

        Example usage:
        ```python
        memory.register_tools_on(llm, send_tool_response_to_model=True)
        # Or register a subset:
        memory.register_tools_on(llm, tools=["semantic_search", "locate", "get_current_context"])
        ```
        """
        for fn, desc in self.memory.get_tools_for_registration():
            name = desc["function"]["name"]
            if tools is None or name in tools:
                llm.register_tool(fn, desc, send_tool_response_to_model)

    def _update_cmd_args_list(self):
        """
        Update launch command arguments
        """
        super()._update_cmd_args_list()

        self.launch_cmd_args = [
            "--layers",
            self._get_layers_json(),
        ]

        self.launch_cmd_args = [
            "--model_client",
            self._get_client_json(self.model_client),
        ]

        self.launch_cmd_args = [
            "--embedding_client",
            self._get_client_json(self.embedding_client),
        ]

    def _get_layers_json(self) -> Union[str, bytes, bytearray]:
        """
        Serialize component layers to json

        :return: Serialized inputs
        :rtype:  str | bytes | bytearray
        """
        if not hasattr(self, "layers_dict"):
            return "[]"
        return json.dumps([layer.to_json() for layer in self.layers_dict.values()])

    def _get_client_json(
        self, client: Optional[ModelClient]
    ) -> Union[str, bytes, bytearray]:
        """
        Serialize component client to json

        :return: Serialized inputs
        :rtype:  str | bytes | bytearray
        """
        if not client:
            return ""
        return json.dumps(client.serialize())

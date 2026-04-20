"""Spatio-temporal memory wired into a Cortex planner.

This example showcases the use of the Memory component with Cortex. Memory
exposes two distinct retrieval surfaces:

- **Perception** (scene descriptions, detections) — retrieved via semantic,
  spatial, temporal, and episodic queries.
- **Interoception** (internal body state such as battery level) — layers
  marked ``is_internal_state=True``, retreived via body state tool.

Cortex auto-discovers both and can proactively check body state before
acting.

Send Cortex a natural-language goal like:

    # Perception queries
    "Where did you last see the fridge?"
    "Summarize what happened in the last 10 minutes."
    "What is currently around you?"

    # Body / interoception queries
    "What is your current battery level?"
    "How are you feeling?"

Usage:
    python3 examples/memory_with_cortex.py

    # Open localhost:5001 in the browser and send a goal through the UI.

    # This example includes a ``battery_layer`` subscribed to /battery_level.
    # If no real battery publisher exists, fake one from another terminal:
    #
    #     ros2 topic pub /battery_level std_msgs/msg/Float32 "{data: 42.0}"
    #
    # (use ``-r 1`` to publish at 1 Hz, or re-run with a different value to
    # simulate battery drain)

Requirements:
    - Ollama running locally with:
        - a VLM (e.g. ``gemma4:latest``)
        - a planner LLM (e.g. the same ``gemma4:latest``)
        - an embedding model (e.g. ``nomic-embed-text-v2-moe:latest``)
    - ``emem`` installed in the Python env the launcher uses
"""

from agents.clients import OllamaClient
from agents.components import Cortex, Memory, VLM, TextToSpeech, Vision
from agents.config import (
    CortexConfig,
    MemoryConfig,
    TextToSpeechConfig,
    VisionConfig,
)
from agents.models import OllamaModel
from agents.ros import FixedInput, Launcher, MemLayer, Topic


# -- Shared perception topics --
image_in = Topic(name="/image_raw", msg_type="Image")
position = Topic(name="/odom", msg_type="Odometry")


# -- Vision component: object detection --
detections_out = Topic(name="detections", msg_type="Detections")

vision = Vision(
    inputs=[image_in],
    outputs=[detections_out],
    config=VisionConfig(threshold=0.5, enable_local_classifier=True),
    trigger=1.0,
    component_name="vision",
)


# -- VLM component: scene captioning --
# Output goes into memory as the 'scene_description' layer AND into TTS
# as spoken feedback (same topic, two subscribers).
scene_query = FixedInput(
    name="scene_query",
    msg_type="String",
    fixed=(
        "Describe the scene in one concise sentence. Focus on the room type, "
        "notable objects, and any people present."
    ),
)
scene_description = Topic(name="scene_description", msg_type="String")

vlm_model = OllamaModel(name="gemma4", checkpoint="gemma4:latest")
vlm_client = OllamaClient(vlm_model)

captioner = VLM(
    inputs=[scene_query, image_in],
    outputs=[scene_description],
    model_client=vlm_client,
    trigger=10.0,
    component_name="captioner",
)


# -- Memory component --
embedding_model = OllamaModel(
    name="embeddings", checkpoint="nomic-embed-text-v2-moe:latest"
)
embedding_client = OllamaClient(embedding_model)

detections_layer = MemLayer(subscribes_to=detections_out, temporal_change=True)
scene_layer = MemLayer(subscribes_to=scene_description)

# Add an interoception layer
battery_layer = MemLayer(
    subscribes_to=Topic(name="/battery_level", msg_type="Float32"),
    is_internal_state=True,
)

memory = Memory(
    layers=[detections_layer, scene_layer, battery_layer],
    position=position,
    model_client=vlm_client,
    embedding_client=embedding_client,
    config=MemoryConfig(
        db_path="/tmp/memory_with_cortex.db",
        consolidation_window=300.0,
        archive_after_seconds=1800.0,
    ),
    trigger=10.0,
    component_name="memory",
)


# -- Text-to-Speech: speak Cortex's final responses --
tts_in = Topic(name="cortex_output", msg_type="StreamingString")
tts = TextToSpeech(
    inputs=[tts_in],
    config=TextToSpeechConfig(enable_local_model=True, play_on_device=True),
    trigger=tts_in,
    component_name="tts",
)


# -- Cortex --
cortex = Cortex(
    output=tts_in,  # Cortex streams its replies to the TTS input
    model_client=vlm_client,
    config=CortexConfig(max_execution_steps=10),
    component_name="cortex",
)


# -- Launch --
launcher = Launcher()
launcher.enable_ui(inputs=[cortex.ui_main_action_input], outputs=[tts_in])
launcher.add_pkg(
    components=[vision, captioner, memory, tts, cortex],
    multiprocessing=True,
    package_name="automatika_embodied_agents",
)
launcher.on_process_fail()
launcher.bringup()

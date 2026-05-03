"""Semantic map built with the Memory component."""

from typing import Optional

from agents.clients import OllamaClient, RoboMLRESPClient
from agents.components import VLM, Memory, Vision
from agents.config import MemoryConfig, VisionConfig
from agents.models import OllamaModel, VisionModel
from agents.ros import FixedInput, Launcher, MemLayer, Topic


# -- Vision component: object detection --
image0 = Topic(name="image_raw", msg_type="Image")
detections_topic = Topic(name="detections", msg_type="Detections")

object_detection = VisionModel(
    name="object_detection", checkpoint="PekingU/rtdetr_r50vd_coco_o365"
)
roboml_detection = RoboMLRESPClient(object_detection)

vision = Vision(
    inputs=[image0],
    outputs=[detections_topic],
    trigger=image0,
    config=VisionConfig(threshold=0.5),
    model_client=roboml_detection,
    component_name="detection_component",
)


# -- VLM component: introspection (room classification) --
qwen_vl = OllamaModel(name="qwen_vl", checkpoint="qwen2.5vl:latest")
qwen_client = OllamaClient(qwen_vl)

introspection_query = FixedInput(
    name="introspection_query",
    msg_type="String",
    fixed=(
        "What kind of a room is this? Is it an office, a bedroom or a "
        "kitchen? Give a one word answer, out of the given choices"
    ),
)
introspection_answer = Topic(name="introspection_answer", msg_type="String")

introspector = VLM(
    inputs=[introspection_query, image0],
    outputs=[introspection_answer],
    model_client=qwen_client,
    trigger=15.0,
    component_name="introspector",
)


def introspection_validation(output: str) -> Optional[str]:
    for option in ["office", "bedroom", "kitchen"]:
        if option in output.lower():
            return option


introspector.add_publisher_preprocessor(introspection_answer, introspection_validation)


# -- Memory component --
# Two perception layers: detections (high temporal change) and scene labels.
detections_layer = MemLayer(subscribes_to=detections_topic, temporal_change=True)
scene_layer = MemLayer(subscribes_to=introspection_answer)

# Memory uses real-world coordinates from Odometry directly.
# For embeddings we use a small Ollama embedding model; if no
# embedding_client is provided, Memory falls back to sentence-transformers.
embedding_client = OllamaClient(
    OllamaModel(name="embeddings", checkpoint="nomic-embed-text-v2-moe:latest")
)

position = Topic(name="odom", msg_type="Odometry")

memory = Memory(
    layers=[detections_layer, scene_layer],
    position=position,
    model_client=qwen_client,  # used for episodic consolidation / summarization
    embedding_client=embedding_client,
    config=MemoryConfig(db_path="/tmp/semantic_map.db"),
    trigger=15.0,
    component_name="memory",
)


# -- Launch --
launcher = Launcher()
launcher.add_pkg(components=[vision, introspector, memory])
launcher.bringup()

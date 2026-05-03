"""Tool calling on an LLM, backed by Memory."""

import re
from typing import Optional

import numpy as np

from agents.clients import OllamaClient, RoboMLRESPClient
from agents.components import LLM, Memory, Vision
from agents.config import LLMConfig, MemoryConfig, VisionConfig
from agents.models import OllamaModel, VisionModel
from agents.ros import Launcher, MemLayer, Topic


# -- Perception side: vision + memory --
image0 = Topic(name="image_raw", msg_type="Image")
detections_topic = Topic(name="detections", msg_type="Detections")
position = Topic(name="odom", msg_type="Odometry")

vision = Vision(
    inputs=[image0],
    outputs=[detections_topic],
    trigger=image0,
    config=VisionConfig(threshold=0.5),
    model_client=RoboMLRESPClient(
        VisionModel(name="rtdetr", checkpoint="PekingU/rtdetr_r50vd_coco_o365")
    ),
    component_name="vision",
)

embedding_client = OllamaClient(
    OllamaModel(name="embeddings", checkpoint="nomic-embed-text-v2-moe:latest")
)

memory = Memory(
    layers=[MemLayer(subscribes_to=detections_topic, temporal_change=True)],
    position=position,
    embedding_client=embedding_client,
    config=MemoryConfig(db_path="/tmp/tool_calling.db"),
    trigger=10.0,
    component_name="memory",
)


# -- Go-to-X LLM --
qwen = OllamaModel(name="qwen", checkpoint="qwen3.5:latest")
qwen_client = OllamaClient(qwen)

goto_in = Topic(name="goto_in", msg_type="String")
goal_point = Topic(name="goal_point", msg_type="PoseStamped")

goto = LLM(
    inputs=[goto_in],
    outputs=[goal_point],
    model_client=qwen_client,
    trigger=goto_in,
    config=LLMConfig(),
    component_name="go_to_x",
)

goto.set_component_prompt(
    template=(
        "The user asks you to go to a place. Call the "
        "``find_location_coordinates`` tool with the place's name."
    )
)


# -- Custom tool that wraps Memory's locate call --
_LOCATION_RE = re.compile(r"Location:\s*\(([^)]+)\)")


def find_location_coordinates(place: str) -> Optional[np.ndarray]:
    """Look up *place* in spatio-temporal memory and return its coordinates.

    Returns a 3-vector ``np.ndarray`` ([x, y, z]) which the framework will
    convert into the outgoing ``PoseStamped`` message. Returns ``None`` if
    the place can't be located, in which case nothing is published.
    """
    locate_output = memory.locate(concept=place)
    match = _LOCATION_RE.search(locate_output)
    if not match:
        return
    try:
        coords = np.fromstring(match.group(1), sep=",", dtype=np.float64)
    except ValueError:
        return
    if coords.shape[0] == 2:
        coords = np.append(coords, 0.0)
    if coords.shape[0] != 3:
        return
    return coords


find_location_description = {
    "type": "function",
    "function": {
        "name": "find_location_coordinates",
        "description": (
            "Look up the coordinates of a known place in spatio-temporal "
            "memory and dispatch the robot there."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "place": {
                    "type": "string",
                    "description": (
                        "The name or short description of the place to "
                        "navigate to (e.g. 'kitchen', 'the door')."
                    ),
                },
            },
            "required": ["place"],
        },
    },
}

# send_tool_response_to_model=False: the tool's return (np.ndarray) is the
# component's output and gets published as PoseStamped.
goto.register_tool(
    tool=find_location_coordinates,
    tool_description=find_location_description,
    send_tool_response_to_model=False,
)


# -- Launch (single process so the closure can call memory directly) --
launcher = Launcher()
launcher.add_pkg(components=[vision, memory, goto])
launcher.bringup()

"""Go-to-X using Memory tool calling on an LLM."""

import re
from typing import Optional

import numpy as np

from agents.clients import OllamaClient
from agents.components import LLM, Memory, Vision
from agents.config import LLMConfig, MemoryConfig, VisionConfig
from agents.models import OllamaModel, VisionModel
from agents.clients import RoboMLRESPClient
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
    config=MemoryConfig(db_path="/tmp/go_to_x.db"),
    trigger=10.0,
    component_name="memory",
)


# -- Go-to-X LLM with Memory's locate tool --
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
        "The user asks you to go to a place. Use the available tools to "
        "look up the place's location in memory. Pass the place name to "
        "the locate tool as the ``concept`` argument."
    )
)

# Register Memory's locate tool on the LLM. send_tool_response_to_model=False
# means the tool's textual return value is what gets published (after the
# preprocessor below converts it).
memory.register_tools_on(goto, tools=["locate"], send_tool_response_to_model=False)


# -- Publisher preprocessor: parse "Location: (x, y, z)" → np.ndarray --
_LOCATION_RE = re.compile(r"Location:\s*\(([^)]+)\)")


def locate_text_to_goal_point(output: str) -> Optional[np.ndarray]:
    """Pull the centroid coordinates out of Memory.locate's text output."""
    match = _LOCATION_RE.search(output)
    if not match:
        return  # no match means no publish
    try:
        coords = np.fromstring(match.group(1), sep=",", dtype=np.float64)
    except ValueError:
        return
    if coords.shape[0] == 2:
        coords = np.append(coords, 0.0)
    if coords.shape[0] != 3:
        return
    return coords


goto.add_publisher_preprocessor(goal_point, locate_text_to_goal_point)


# -- Launch (single process so the LLM can call Memory in-process) --
launcher = Launcher()
launcher.add_pkg(components=[vision, memory, goto])
launcher.bringup()

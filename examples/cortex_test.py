"""Minimal Cortex test recipe using the extended Launcher."""

from agents.components import Vision, Cortex
from agents.config import VisionConfig, CortexConfig
from agents.models import OllamaModel
from agents.clients import OllamaClient
from agents.ros import Topic, Action, Event
from agents.ros import Launcher


# -- Model client for Cortex planner --
qwen = OllamaModel(name="qwen", checkpoint="llava:latest")
qwen_client = OllamaClient(qwen)

# -- Vision component (local classifier, no remote model needed) --
image_in = Topic(name="camera/image_raw", msg_type="Image")
detections_out = Topic(name="detections", msg_type="Detections")

vision = Vision(
    inputs=[image_in],
    outputs=[detections_out],
    config=VisionConfig(enable_local_classifier=True),
    trigger=1.0,
    component_name="vision",
)

# -- Cortex: the planner / monitor --
status_topic = Topic(name="cortex_status", msg_type="String")

cortex = Cortex(
    inputs=[],
    outputs=[status_topic],
    actions=[
        Action(
            method=vision.take_picture,
            description="Take a picture from the camera and save it to disk",
        ),
    ],
    model_client=qwen_client,
    config=CortexConfig(max_iterations=10),
    component_name="cortex",
)

test_topic = Topic(name="test_topic", msg_type="String")
event_test_in = Event(event_condition=test_topic.msg.data == "trigger")


def log_something():
    print(
        "Event triggered! The worker should have been called by the Cortex to process the input and produce an output."
    )


events_actions = {event_test_in: Action(method=log_something)}

# -- Launch with the extended Launcher (replaces Monitor with Cortex) --
launcher = Launcher()
launcher.add_pkg(components=[vision, cortex], events_actions=events_actions)
launcher.bringup()

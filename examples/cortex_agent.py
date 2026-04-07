"""Cortex agent with vision, visual Q&A, speech, and a custom action.

This example demonstrates a Cortex-based agent that can:
- Take pictures using the Vision component's ``take_picture`` action
- Describe what it sees using the VLM component's ``describe`` action
- Speak using the TextToSpeech component's ``say`` action,
  or automatically speak VLM output routed via the text_in topic
- Toggle an LED via a custom action defined in this script

The Cortex component acts as the system monitor and task planner. Send it a
goal like "describe what you see" and it will inspect the available
components, plan the steps, and execute them in sequence. The VLM's text
output is routed to the TTS component so descriptions are spoken aloud.

Usage:
    python3 examples/cortex_agent.py

    # In another terminal, send a goal:
    ros2 action send_goal /cortex_<process_id>/vision_language_action automatika_embodied_agents/action/VisionLanguageAction "{task: 'describe what you see'}"
"""

from agents.components import Vision, VLM, TextToSpeech, Cortex
from agents.config import VisionConfig, TextToSpeechConfig, CortexConfig
from agents.models import OllamaModel
from agents.clients import OllamaClient
from agents.ros import Topic, Action, Launcher


# -- Model clients --
planner_model = OllamaModel(name="qwen", checkpoint="qwen3.5:latest")
planner_client = OllamaClient(planner_model)

vlm_model = OllamaModel(name="qwen_vl", checkpoint="qwen2.5vl:latest")
vlm_client = OllamaClient(vlm_model)

# -- Vision component --
image_in = Topic(name="/image_raw", msg_type="Image")
detections_out = Topic(name="detections", msg_type="Detections")

vision = Vision(
    inputs=[image_in],
    outputs=[detections_out],
    config=VisionConfig(enable_local_classifier=True),
    trigger=0.5,
    component_name="vision",
)

# -- VLM component (describe action, output routed to TTS) --
vlm_query = Topic(name="vlm_query", msg_type="String")
vlm_output = Topic(name="text_in", msg_type="String")  # same topic as TTS input

vlm = VLM(
    inputs=[vlm_query, image_in],
    outputs=[vlm_output],
    model_client=vlm_client,
    trigger=vlm_query,
    component_name="vlm",
)

# -- Text-to-Speech component (local model, triggered by VLM output) --
tts = TextToSpeech(
    inputs=[vlm_output],
    outputs=[Topic(name="audio_out", msg_type="Audio")],
    config=TextToSpeechConfig(enable_local_model=True, play_on_device=True),
    trigger=vlm_output,
    component_name="tts",
)


# -- Custom action: toggle an LED --
led_on = False


def toggle_led():
    """Toggle an LED on the robot."""
    global led_on
    led_on = not led_on
    state = "ON" if led_on else "OFF"
    print(f"LED toggled {state}")


# -- Cortex: the planner / monitor --
cortex = Cortex(
    actions=[
        Action(method=toggle_led, description="Toggle the robot's LED on or off."),
    ],
    model_client=planner_client,
    config=CortexConfig(max_planning_steps=5, max_execution_steps=10),
    component_name="cortex",
)


# -- Launch --
launcher = Launcher()
launcher.add_pkg(
    components=[vision, vlm, tts, cortex],
    multiprocessing=True,
    package_name="automatika_embodied_agents",
)
launcher.bringup()

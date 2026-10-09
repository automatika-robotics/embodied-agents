"""Cortex writing its own actions and event conditions.

With ``enable_scratch_functions`` on, the planner can write a Python function
when nothing available does what a task needs, and use it at once: a written
action becomes a tool it can plan with and give to an event, and a written
condition becomes a condition it can install an event on. The functions run in
the launcher process with its privileges, also when the other components run
in their own processes as here, since Cortex is the system monitor. That is why
this is off by default and should only be turned on for a planning model and a
deployment you trust to that degree. ``scratch_notes`` tells the planner what the environment
offers, such as the names of the variables holding credentials.

This recipe has a Vision component publishing detections, and a TextToSpeech
component with a ``say`` action. Send Cortex standing instructions that need
more than those, for example:

- "Whenever a person is detected, email me about it." There is no email
  action, so the planner writes one, with the SMTP settings read from the
  environment variables named in ``scratch_notes``, and installs an event on
  the detections topic that runs it.
- "Whenever the CPU temperature goes above 80 degrees, say so out loud."
  No topic publishes the CPU temperature, so the planner writes a condition
  that reads it, and installs a polled event on that condition that runs
  ``say``.

Usage:
    export SMTP_HOST=mail.example.org SMTP_PORT=587 SMTP_USER=robot@example.org
    export SMTP_PASSWORD=... ALERT_EMAIL=you@example.org
    python3 examples/cortex_out_of_graph_actions.py

    # In another terminal, send a standing instruction:
    ros2 action send_goal /cortex_input_command automatika_embodied_agents/action/VisionLanguageAction "{task: 'Whenever a person is detected, email me about it.'}"
"""

from agents.components import Vision, TextToSpeech, Cortex
from agents.config import VisionConfig, TextToSpeechConfig, CortexConfig
from agents.models import OllamaModel, VisionModel, TransformersTTS
from agents.clients import OllamaClient, RoboMLRESPClient
from agents.ros import Topic, Launcher


# -- Model clients --
# A thinking model left to think plans slowly and can use up its token budget
# before it answers
planner_client = OllamaClient(
    OllamaModel(name="qwen", checkpoint="qwen3.5:latest", think=False),
    inference_timeout=120,
)
detection_client = RoboMLRESPClient(
    VisionModel(name="rtdetr", checkpoint="PekingU/rtdetr_r50vd_coco_o365")
)
tts_client = RoboMLRESPClient(TransformersTTS(name="speecht5"))

# -- Vision component: detections the planner can install events on --
image_in = Topic(name="/image_raw", msg_type="Image")
detections_out = Topic(name="detections", msg_type="Detections")

vision = Vision(
    inputs=[image_in],
    outputs=[detections_out],
    model_client=detection_client,
    config=VisionConfig(threshold=0.5),
    trigger=0.5,
    component_name="vision",
)

# -- TextToSpeech component: its `say` action is a tool for the planner --
text_in = Topic(name="text_in", msg_type="String")

tts = TextToSpeech(
    inputs=[text_in],
    model_client=tts_client,
    config=TextToSpeechConfig(play_on_device=True),
    trigger=text_in,
    component_name="tts",
)

# -- Cortex: may write functions, and knows what the environment offers --
cortex = Cortex(
    model_client=planner_client,
    config=CortexConfig(
        enable_events=True,  # standing instructions need runtime events
        enable_scratch_functions=True,  # read the warning in CortexConfig first
        scratch_notes=(
            "The process environment holds SMTP_HOST, SMTP_PORT, SMTP_USER and "
            "SMTP_PASSWORD, an SMTP server and account for sending mail, and "
            "ALERT_EMAIL, the operator's address. The CPU temperature in "
            "millidegrees Celsius can be read from "
            "/sys/class/thermal/thermal_zone0/temp."
        ),
        max_new_tokens=2000,  # a written function's source travels in one reply
    ),
    component_name="cortex",
)

launcher = Launcher()
launcher.add_pkg(
    components=[vision, tts, cortex],
    package_name="automatika_embodied_agents",
    multiprocessing=True,
)
launcher.bringup()

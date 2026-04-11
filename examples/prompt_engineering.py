from agents.components import Vision, MLLM
from agents.models import VisionModel, OllamaModel
from agents.clients import RoboMLRESPClient, OllamaClient
from agents.ros import Topic, Launcher
from agents.config import VisionConfig

image0 = Topic(name="image_raw", msg_type="Image")
detections_topic = Topic(name="detections", msg_type="Detections")

object_detection = VisionModel(
    name="object_detection", checkpoint="PekingU/rtdetr_r50vd_coco_o365"
)
roboml_detection = RoboMLRESPClient(object_detection)

detection_config = VisionConfig(threshold=0.5)
vision = Vision(
    inputs=[image0],
    outputs=[detections_topic],
    trigger=image0,
    config=detection_config,
    model_client=roboml_detection,
    component_name="detection_component",
)

text_query = Topic(name="text0", msg_type="String")
text_answer = Topic(name="text1", msg_type="String")

mllm_model = OllamaModel(name="mllm_model", checkpoint="qwen2.5vl:latest")
mllm_client = OllamaClient(mllm_model)

mllm = MLLM(
    inputs=[text_query, image0, detections_topic],
    outputs=[text_answer],
    model_client=mllm_client,
    trigger=text_query,
    component_name="mllm_component",
)

mllm.set_component_prompt(
    template="""Imagine you are a robot.
    This image has following items: {{ detections }}.
    Answer the following about this image: {{ text0 }}"""
)
launcher = Launcher()
launcher.enable_ui(inputs=[text_query], outputs=[text_answer, detections_topic])
launcher.add_pkg(components=[vision, mllm])
launcher.bringup()

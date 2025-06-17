from agents.components import MLLM, SpeechToText, TextToSpeech
from agents.config import SpeechToTextConfig, TextToSpeechConfig
from agents.clients import OllamaClient, RoboMLWSClient
from agents.models import Whisper, SpeechT5, OllamaModel
from agents.ros import Topic, Launcher

audio_in = Topic(name="audio0", msg_type="Audio")
text_query = Topic(name="text0", msg_type="String")

whisper = Whisper(name="whisper")  # Custom model init params can be provided here
roboml_whisper = RoboMLWSClient(whisper)

s2t_config = SpeechToTextConfig(
    enable_vad=True,  # option to listen for speech through the microphone
    enable_wakeword=True,  # option to invoke the component with a wakeword like 'hey jarvis'
)
speech_to_text = SpeechToText(
    inputs=[audio_in],
    outputs=[text_query],
    model_client=roboml_whisper,
    trigger=audio_in,
    config=s2t_config,
    component_name="speech_to_text",
)

image0 = Topic(name="image_raw", msg_type="Image")
text_answer = Topic(name="text1", msg_type="String")

llava = OllamaModel(name="llava", checkpoint="llava:latest")
llava_client = OllamaClient(llava)

mllm = MLLM(
    inputs=[text_query, image0],
    outputs=[text_answer],
    model_client=llava_client,
    trigger=text_query,
    component_name="vqa",
)

# config for playing audio on device
t2s_config = TextToSpeechConfig(play_on_device=True)

speecht5 = SpeechT5(name="speecht5")
roboml_speecht5 = RoboMLWSClient(speecht5)
text_to_speech = TextToSpeech(
    inputs=[text_answer],
    trigger=text_answer,
    model_client=roboml_speecht5,
    config=t2s_config,
    component_name="text_to_speech",
)

launcher = Launcher()
launcher.add_pkg(
    components=[speech_to_text, mllm, text_to_speech],
)
launcher.bringup()

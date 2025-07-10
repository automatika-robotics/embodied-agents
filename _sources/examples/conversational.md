# Create a conversational agent with audio

Often times robots are equipped with a speaker system and a microphone. Once these peripherals have been exposed through ROS, we can use _EmbodiedAgents_ to trivially create a conversational interface on the robot. Our conversational agent will use a multimodal LLM for contextual question/answering utilizing the camera onboard the robot. Furthermore, it will use speech-to-text and text-to-speech models for converting audio to text and vice versa. We will start by importing the relavent components that we want to string together.

```python
from agents.components import MLLM, SpeechToText, TextToSpeech
```

 [Components](../basics/components) are basic functional units in _EmbodiedAgents_. Their inputs and outputs are defined using ROS [Topics](../basics/components.md#topic). And their function can be any input transformation, for example the inference of an ML model. Lets setup these components one by one. Since our input to the robot would be speech, we will setup the speech-to-text component first.

 ## SpeechToText Component
 This component listens to input an audio input topic, that takes in a multibyte array of audio (captured in a ROS std_msgs message, which maps to Audio msg_type in Sugarcoat🍬) and can publish output to a text topic. It can also be configured to get the audio stream from microphones on board our robot. By default the component is configured to use a small Voice Activity Detection (VAD) model, [Silero-VAD](https://github.com/snakers4/silero-vad) to filter out any audio that is not speech.

 However, merely utilizing speech can be problamatic in robots, due to the hands free nature of the audio system. Therefore its useful to add wakeword detection, so that speech-to-text is only activated when the robot is called with a specific phrase (e.g. 'Hey Jarvis').

 We will be using this configuration in our example. First we will setup our input and output topics and then create a config object which we can later pass to our component.


```{note}
With **enable_vad** set to **True**, the component automatically downloads and deploys [Silero-VAD](https://github.com/snakers4/silero-vad) by default in ONNX format. This model has a small footprint and can be easily deployed on the edge. However we need to install a couple of dependencies for this to work. These can be installed with: `pip install pyaudio onnxruntime`
```
```{note}
With **enable_wakeword** set to **True**, the component automatically downloads and deploys a pre-trained model from [openWakeWord](https://github.com/dscripka/openWakeWord) by default in ONNX format, that can be invoked with **'Hey Jarvis'**. Other pre-trained models from openWakeWord are available [here](https://github.com/dscripka/openWakeWord). However it is recommended that you deploy own wakeword model, which can be easily trained by following [this amazing tutorial](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb). The tutorial notebook can be run in [Google Colab](https://colab.research.google.com/drive/1yyFH-fpguX2BTAW8wSQxTrJnJTM-0QAd?usp=sharing).
```
```python
from agents.ros import Topic
from agents.config import SpeechToTextConfig

# Define input and output topics (pay attention to msg_type)
audio_in = Topic(name="audio0", msg_type="Audio")
text_query = Topic(name="text0", msg_type="String")

s2t_config = SpeechToTextConfig(enable_vad=True,     # option to listen for speech through the microphone
                                enable_wakeword=True) # option to invoke the component with a wakeword like 'hey jarvis'
```
```{warning}
The _enable_wakeword_ option cannot be enabled without the _enable_vad_ option.
```
```{seealso}
Check the available defaults and options for the SpeechToTextConfig [here](../apidocs/agents/agents.config)
```

To initialize the component we also need a model client for a speech to text model. We will be using the WebSocket client for RoboML for this purpose.

```{note}
RoboML is a aggregator library that provides a model serving aparatus for locally serving opensource ML models useful in robotics. Learn about setting up RoboML [here](https://www.github.com/automatika-robotics/roboml).
```

Additionally, we will use the client with a model called, Whisper, a popular opensource speech to text model from OpenAI. Lets see what the looks like in code.

```python
from agents.clients import RoboMLWSClient
from agents.models import Whisper

# Setup the model client
whisper = Whisper(name="whisper")  # Custom model init params can be provided here
roboml_whisper = RoboMLWSClient(whisper)

# Initialize the component
speech_to_text = SpeechToText(
    inputs=[audio_in],  # the input topic we setup
    outputs=[text_query], # the output topic we setup
    model_client=roboml_whisper,
    trigger=audio_in,
    config=s2t_config,  # pass in the config object
    component_name="speech_to_text"
)
```
The trigger parameter lets the component know that it has to perform its function (in this case model inference) when an input is received on this particular topic. In our configuration, the component will be triggered using voice activity detection on the continuous stream of audio being received on the microphone. Next we will setup our MLLM component.

## MLLM Component
The MLLM component takes as input a text topic (the output of the SpeechToText component) and an image topic, assuming we have a camera device onboard the robot publishing this topic. And just like before we need to provide a model client, this time with an MLLM model. This time we will use the OllamaClient along with _llava:latest_ model, a popular opensource multimodal LLM available on Ollama. Furthermore, we will configure our MLLM component using `MLLMConfig`. We will set `stream=True` to make the MLLM output text, be published as a stream for downstream components that consume this output. In _EmbodiedAgents_, streaming can output can be chunked using a `break_character` in the config (Default: '.').This way the downstream TextToSpeech component can start generating audio as soon as the first sentence is produced by the LLM.


```{note}
Ollama is one of the most popular local LLM serving projects. Learn about setting up Ollama [here](https://ollama.com).
```
Here is the code for our MLLM setup.

```python
from agents.clients.ollama import OllamaClient
from agents.models import OllamaModel
from agents.config import MLLMConfig

# Define the image input topic and a new text output topic
image0 = Topic(name="image_raw", msg_type="Image")
text_answer = Topic(name="text1", msg_type="String")

# Define a model client (working with Ollama in this case)
# OllamaModel is a generic wrapper for all ollama models
llava = OllamaModel(name="llava", checkpoint="llava:latest")
llava_client = OllamaClient(llava)

mllm_config = MLLMConfig(stream=True)  # Other inference specific paramters can be provided here

# Define an MLLM component
mllm = MLLM(
    inputs=[text_query, image0],  # Notice the text input is the same as the output of the previous component
    outputs=[text_answer],
    model_client=llava_client,
    trigger=text_query,
    component_name="vqa" # We have also given our component an optional name
)
```

We can further customize the our MLLM component by attaching a context prompt template. This can be done at the component level or at the level of a particular input topic. In this case we will attach a prompt template to the input topic **text_query**.

```python
# Attach a prompt template
mllm.set_topic_prompt(text_query, template="""You are an amazing and funny robot.
Answer the following about this image: {{ text0 }}"""
)
```
Notice that the template is a jinja2 template string, where the actual name of the topic is set as a variable. For longer templates you can also write them to a file and provide its path when calling this function. After this we move on to setting up our last component.

## TextToSpeech Component
The TextToSpeech component setup will be very similar to the SpeechToText component. We will once again use a RoboML client, this time with the SpeechT5 model (opensource model from Microsoft). Furthermore, this component can be configured to play audio on a playback device available onboard the robot. We will utilize this option through our config. An output topic is optional for this component as we will be playing the audio directly on device.

```{note}
In order to utilize _play_on_device_ you need to install a couple of dependencies as follows: `pip install soundfile sounddevice`
```

```python
from agents.config import TextToSpeechConfig
from agents.models import SpeechT5

# config for asynchronously playing audio on device
t2s_config = TextToSpeechConfig(play_on_device=True, stream=True)

speecht5 = SpeechT5(name="speecht5")
roboml_speecht5 = RoboMLWSClient(speecht5)
text_to_speech = TextToSpeech(
    inputs=[text_answer],
    trigger=text_answer,
    model_client=roboml_speecht5,
    config=t2s_config,
    component_name="text_to_speech"
)
```
## Launching the Components
The final step in this example is to launch the components. This is done by passing the defined components to the launcher and calling the **bringup** method.

```python
from agents.ros import Launcher

# Launch the components
launcher = Launcher()
launcher.add_pkg(
    components=[speech_to_text, mllm, text_to_speech]
    )
launcher.bringup()
```

Et voila! we have setup a graph of three components in less than 50 lines of well formatted code. The complete example is as follows:

```{code-block} python
:caption: Multimodal Audio Conversational Agent
:linenos:
from agents.components import MLLM, SpeechToText, TextToSpeech
from agents.config import SpeechToTextConfig, TextToSpeechConfig, MLLMConfig
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
mllm_config = MLLMConfig(stream=True)  # Other inference specific paramters can be provided here

mllm = MLLM(
    inputs=[text_query, image0],
    outputs=[text_answer],
    model_client=llava_client,
    trigger=text_query,
    config=mllm_config,
    component_name="vqa",
)

# config for asynchronously playing audio on device
t2s_config = TextToSpeechConfig(play_on_device=True, stream=True)

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
```

## Web Based Client for Interacting with the Robot

To interact with text and audio based topics on the robot, _EmbodiedAgents_ includes a tiny browser based client made with [chainlit](https://chainlit.io/). This is useful if the robot does not have a microphone/speaker interface or if one wants to communicate with it remotely. The client can be launched as follows:

```shell
ros2 run automatika_embodied_agents tiny_web_client
```

The client displays a web UI on http://localhost:8000. Open this address from browser. ROS input and output topic settings for text and audio topics can be configured from the web UI by pressing the settings icon.

```{seealso}
To customize the webapp, checkout how to set the configuration options for the [app](https://docs.chainlit.io/backend/config/overview) and the [UI](https://docs.chainlit.io/customisation/overview)
```

"""
A Component is the main execution unit in EmbodiedAgents and in essence each component is synctactic sugar over a ROS2 Lifecycle Node. EmbodiedAgents provides the following components. These components can be arbitrarily combined to form an embodied agent graph.

```{list-table}
:widths: 20 80
:header-rows: 1
* - Component Name
  - Description

* - **[LLM](agents.components.llm.md)**
  - This component utilizes large language models (e.g LLama) that can be used to process text data.

* - **[MLLM](agents.components.mllm.md)**
  - This component utilizes multi-modal large language models (e.g. Llava) that can be used to process text and image data.

* - **[SpeechToText](agents.components.speechtotext.md)**
  - This component takes in audio input and outputs a text representation of the audio using Speech-to-Text models (e.g. Whisper).

* - **[TextToSpeech](agents.components.texttospeech.md)**
  - This component takes in text input and outputs an audio representation of the text using TTS models (e.g. SpeechT5). The generated audio can be played using any audio playback device available on the agent.

* - **[MapEncoding](agents.components.map_encoding.md)**
  - Map encoding component that encodes text information as a semantic map based on the robots localization. It takes in map layers, position topic, map meta data topic, and a vector database client. Map layers can be arbitrary text based outputs from other components such as MLLMs or Vision.

* - **[SemanticRouter](agents.components.semantic_router.md)**
  - A component that routes semantic information from input topics to output topics based on pre-defined routes. The Semantic Router takes in a list of input topics, a list of routes, an optional default route, and a configuration object. It uses the database client to store and retrieve routing information.

* - **[Vision](agents.components.vision.md)**
  - This component performs object detection and tracking on input images and outputs a list of detected objects, along with their bounding boxes and confidence scores.

* - **[VideoMessageMaker](agents.components.imagestovideo.md)**
  - This component generates ROS video messages from input image messages. A video message is a collection of image messages that have a perceivable motion. I.e. the primary task of this component is to make intentionality decisions about what sequence of consecutive images should be treated as one coherent temporal sequence. The motion estimation method used for selecting images for a video can be configured in component config.
```
"""

from .component_base import Component
from .imagestovideo import VideoMessageMaker
from .llm import LLM
from .map_encoding import MapEncoding
from .mllm import MLLM
from .model_component import ModelComponent
from .semantic_router import SemanticRouter
from .speechtotext import SpeechToText
from .texttospeech import TextToSpeech
from .vision import Vision

__all__ = [
    "Component",
    "ModelComponent",
    "MapEncoding",
    "MLLM",
    "LLM",
    "SpeechToText",
    "TextToSpeech",
    "Vision",
    "VideoMessageMaker",
    "SemanticRouter",
]

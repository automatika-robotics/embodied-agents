# Components 🧩

A **Component** is the primary execution unit in _EmbodiedAgents_. Conceptually, each component is syntactic sugar over a ROS 2 Lifecycle Node, inheriting all its lifecycle behaviors while also offering additional abstraction to simplify development. Components receive one or more ROS topics as inputs and produce outputs on designated topics. The specific types and formats of these topics depend on the component's function.

```{note}
To learn more about the internal structure and lifecycle behavior of components, check out the documentation of [Sugarcoat🍬](https://automatika-robotics.github.io/sugarcoat/).
```

## Components Available in _EmbodiedAgents_

_EmbodiedAgents_ provides a suite of ready-to-use components. These can be composed into flexible execution graphs for building autonomous, perceptive, and interactive robot behavior. Each component focuses on a particular modality or functionality, from vision and speech to map reasoning and LLM-based inference.

```{list-table}
:widths: 20 80
:header-rows: 1
* - Component Name
  - Description

* - **[LLM](../apidocs/agents/agents.components.llm)**
  - Uses large language models (e.g., LLaMA) to process text input. Can be used for reasoning, tool calling, instruction following, or dialogue. It can also utilize vector DBs for storing and retreiving contextual information.

* - **[VLM](../apidocs/agents/agents.components.mllm)**
  - Leverages multimodal LLMs (e.g., Llava) for understanding and processing both text and image data. Inherits all functionalities of the LLM component. It can also utilize multimodal LLM based planning models for task specific outputs (e.g. pointing, grounding, affordance etc.)

* - **[SpeechToText](../apidocs/agents/agents.components.speechtotext)**
  - Converts spoken audio into text using speech-to-text models (e.g., Whisper). Suitable for voice command recognition. It also implements small on-board models for Voice Activity Detection (VAD) and Wakeword recognition, using audio capture devices onboard the robot.

* - **[TextToSpeech](../apidocs/agents/agents.components.texttospeech)**
  - Synthesizes audio from text using TTS models (e.g., SpeechT5, Bark). Output audio can be played using the robot's speakers or published to a topic. Implements `say(text)` and `stop_playback` functions to play/stop audio based on events from other components or the environment.

* - **[MapEncoding](../apidocs/agents/agents.components.map_encoding)**
  - Provides a spatio-temporal working memory by converting semantic outputs (e.g., from MLLMs or Vision) into a structured map representation. Uses robot localization data and output topics from other components to store information in a vector DB.

* - **[SemanticRouter](../apidocs/agents/agents.components.semantic_router)**
  - Routes information between topics based on semantic content and predefined routing rules. Uses a vector DB for semantic matching and decision-making. This allows for creating complex graphs of components where a single input source can trigger different information processing pathways.

* - **[Vision](../apidocs/agents/agents.components.vision)**
  - An essential component in all vision powered robots. Performs object detection and tracking on incoming images. Outputs object classes, bounding boxes, and confidence scores. It implements a low-latency small on-board classification model as well.

* - **[VideoMessageMaker](../apidocs/agents/agents.components.imagestovideo)**
  - This component generates ROS video messages from input image messages. A video message is a collection of image messages that have a perceivable motion. I.e. the primary task of this component is to make intentionality decisions about what sequence of consecutive images should be treated as one coherent temporal sequence. The chunking method used for selecting images for a video can be configured in component config. It can be useful in sending videos to ML models that take image sequences.
```

## Topic

A [topic](../apidocs/agents/agents.ros) is an idomatic wrapper for a ROS2 topic. Topics can be given as inputs or outputs to components. When given as inputs, components automatically create listeners for the topics upon their activation. And when given as outputs, components create publishers for publishing to the topic. Each topic has a name (duh?) and a data type, defining its listening callback and publishing behavior. The data type can be provided to the topic as a string. Checkout the list of supported data types [here](https://automatika-robotics.github.io/sugarcoat/advanced/types.html).

```{note}
Learn more about Topics in [Sugarcoat🍬](https://automatika-robotics.github.io/sugarcoat/).
```

## Component Config

Each component can optionally be configured using a `config` object. Configs are generally built using [`attrs`](https://www.attrs.org/en/stable/) and include parameters controlling model inference, thresholds, topic remapping, and other component-specific behavior. Components involving ML models define their inference options here.

To see the default configuration options for each component, refer to the respective config classes in [the API reference](../apidocs/agents/agents.config).

## Component RunType

In _EmbodiedAgents_, components can operate in one of two modes:

```{list-table}
:widths: 10 80
* - **Timed**
  - Executes its main function at regular time intervals (e.g., every N milliseconds).
* - **Event**
  - Executes in response to specific incoming messages or events on one or more trigger topics.
```

## Health Check and Fallback

Each component maintains an internal health state. This is used to support fallback behaviors and graceful degradation in case of errors or resource unavailability. Health monitoring is essential for building reliable and resilient autonomous agents, especially in real-world environments.

Fallback behaviors can include retry mechanisms, switching to alternate inputs, or deactivating the component safely. For deeper understanding, refer to [Sugarcoat🍬](https://automatika-robotics.github.io/sugarcoat/), which underpins the lifecycle and health management logic.

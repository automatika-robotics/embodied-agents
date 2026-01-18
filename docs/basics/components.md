# Components

A **Component** is the primary execution unit in _EmbodiedAgents_. They can represent anything that can be termed as functional behaviour. For example the ability to understand the process text. Components can be combined arbitrarily to create more complex systems such as multi-modal agents with perception-action loops. Conceptually, each component is a lot of syntactic sugar over a ROS2 Lifecycle Node, inheriting all its lifecycle behaviors while also offering allied functionality to manage inputs and outputs to simplify development. Components receive one or more ROS topics as inputs and produce outputs on designated topics. The specific types and formats of these topics depend on the component's function.

```{note}
To learn more about the internal structure and lifecycle behavior of components, check out the documentation of [Sugarcoat🍬](https://automatika-robotics.github.io/sugarcoat/design/component.html).
```

## Components Available in _EmbodiedAgents_

_EmbodiedAgents_ provides a suite of ready-to-use components. These can be composed into flexible execution graphs for building autonomous, perceptive, and interactive robot behavior. Each component focuses on a particular modality or functionality, from vision and speech to map reasoning and VLA based manipulation.

```{list-table}
:widths: 20 80
:header-rows: 1

* - Component Name
  - Description

* - **[LLM](../apidocs/agents/agents.components.llm)**
  - Uses large language models (e.g., LLaMA) to process text input. Can be used for reasoning, tool calling, instruction following, or dialogue. It can also utilize vector DBs for storing and retreiving contextual information.

* - **[VLM](../apidocs/agents/agents.components.mllm)**
  - Leverages multimodal LLMs (e.g., Llava) for understanding and processing both text and image data. Inherits all functionalities of the LLM component. It can also utilize multimodal LLM based planning models for task specific outputs (e.g. pointing, grounding, affordance etc.). This component is also called MLLM.

* - **[VLA](../apidocs/agents/agents.components.vla.md)**
  - Provides an interface to utilize Vision Language Action (VLA) models for manipulation and control tasks. It can use VLA Policies (such as SmolVLA, Pi0 etc.) served with HuggingFace LeRobot Async Policy Server and publish them to common topic formats in MoveIt Servo and ROS2 Control.

* - **[SpeechToText](../apidocs/agents/agents.components.speechtotext)**
  - Converts spoken audio into text using speech-to-text models (e.g., Whisper). Suitable for voice command recognition. It also implements small on-board models for Voice Activity Detection (VAD) and Wakeword recognition, using audio capture devices onboard the robot.

* - **[TextToSpeech](../apidocs/agents/agents.components.texttospeech)**
  - Synthesizes audio from text using TTS models (e.g., SpeechT5, Bark). Output audio can be played using the robot's speakers or published to a topic. Implements `say(text)` and `stop_playback` functions to play/stop audio based on events from other components or the environment.

* - **[MapEncoding](../apidocs/agents/agents.components.map_encoding)**
  - Provides a spatio-temporal working memory by converting semantic outputs (e.g., from MLLMs or Vision) into a structured map representation. Uses robot localization data and output topics from other components to store information in a vector DB.

* - **[SemanticRouter](../apidocs/agents/agents.components.semantic_router)**
  - Routes information between topics based on semantic content and predefined routing rules. Uses a vector DB for semantic matching or an LLM for decision-making. This allows for creating complex graphs of components where a single input source can trigger different information processing pathways.

* - **[Vision](../apidocs/agents/agents.components.vision)**
  - An essential component in all vision powered robots. Performs object detection and tracking on incoming images. Outputs object classes, bounding boxes, and confidence scores. It implements a low-latency small on-board classification model as well.

* - **[VideoMessageMaker](../apidocs/agents/agents.components.imagestovideo)**
  - This component generates ROS video messages from input image messages. A video message is a collection of image messages that have a perceivable motion. I.e. the primary task of this component is to make intentionality decisions about what sequence of consecutive images should be treated as one coherent temporal sequence. The chunking method used for selecting images for a video can be configured in component config. It can be useful in sending videos to ML models that take image sequences.
```

## Topic

A [topic](../apidocs/agents/agents.ros) is an idomatic wrapper for a ROS2 topic. Topics can be given as inputs or outputs to components. When given as inputs, components automatically create listeners for the topics upon their activation. And when given as outputs, components create publishers for publishing to the topic. Each topic has a name (duh?) and a data type, defining its listening callback and publishing behavior. The data type can be provided to the topic as a string. The list of supported data types below.

```{note}
Learn more about Topics in [Sugarcoat🍬](https://automatika-robotics.github.io/sugarcoat/).
```

```{list-table}
:widths: 20 40 40
:header-rows: 1

* - Message
  - ROS2 package
  - Description

* - **[String](https://automatika-robotics.github.io/sugarcoat/apidocs/ros_sugar/ros_sugar.io.supported_types.md/#classes)**
  - [std_msgs](https://docs.ros2.org/foxy/api/std_msgs/msg/String.html)
  - Standard text message.

* - **[Bool](https://automatika-robotics.github.io/sugarcoat/apidocs/ros_sugar/ros_sugar.io.supported_types.md/#classes)**
  - [std_msgs](https://docs.ros2.org/foxy/api/std_msgs/msg/Bool.html)
  - Boolean value (True/False).

* - **[Float32](https://automatika-robotics.github.io/sugarcoat/apidocs/ros_sugar/ros_sugar.io.supported_types.md/#classes)**
  - [std_msgs](https://docs.ros2.org/foxy/api/std_msgs/msg/Float32.html)
  - Single-precision floating point number.

* - **[Float32MultiArray](https://automatika-robotics.github.io/sugarcoat/apidocs/ros_sugar/ros_sugar.io.supported_types.md/#classes)**
  - [std_msgs](https://docs.ros2.org/foxy/api/std_msgs/msg/Float32MultiArray.html)
  - Array of single-precision floating point numbers.

* - **[Float64](https://automatika-robotics.github.io/sugarcoat/apidocs/ros_sugar/ros_sugar.io.supported_types.md/#classes)**
  - [std_msgs](https://docs.ros2.org/foxy/api/std_msgs/msg/Float64.html)
  - Double-precision floating point number.

* - **[Float64MultiArray](https://automatika-robotics.github.io/sugarcoat/apidocs/ros_sugar/ros_sugar.io.supported_types.md/#classes)**
  - [std_msgs](https://docs.ros2.org/foxy/api/std_msgs/msg/Float64MultiArray.html)
  - Array of double-precision floating point numbers.

* - **[Twist](https://automatika-robotics.github.io/sugarcoat/apidocs/ros_sugar/ros_sugar.io.supported_types.html)**
  - [geometry_msgs](https://docs.ros2.org/foxy/api/geometry_msgs/msg/Twist.html)
  - Velocity expressed as linear and angular components.

* - **[Image](https://automatika-robotics.github.io/sugarcoat/apidocs/ros_sugar/ros_sugar.io.supported_types.html)**
  - [sensor_msgs](https://docs.ros2.org/foxy/api/sensor_msgs/msg/Image.html)
  - Raw image data.

* - **[CompressedImage](https://automatika-robotics.github.io/sugarcoat/apidocs/ros_sugar/ros_sugar.io.supported_types.html)**
  - [sensor_msgs](https://docs.ros2.org/foxy/api/sensor_msgs/msg/CompressedImage.html)
  - Compressed image data (e.g., JPEG, PNG).

* - **[Audio](https://automatika-robotics.github.io/sugarcoat/apidocs/ros_sugar/ros_sugar.io.supported_types.html)**
  - [sensor_msgs](https://docs.ros2.org/foxy/api/sensor_msgs/msg/Audio.html)
  - Audio stream data.

* - **[Path](https://automatika-robotics.github.io/sugarcoat/apidocs/ros_sugar/ros_sugar.io.supported_types.html)**
  - [nav_msgs](https://docs.ros2.org/foxy/api/nav_msgs/msg/Path.html)
  - An array of poses representing a navigation path.

* - **[OccupancyGrid](https://automatika-robotics.github.io/sugarcoat/apidocs/ros_sugar/ros_sugar.io.supported_types.html)**
  - [nav_msgs](https://docs.ros2.org/foxy/api/nav_msgs/msg/OccupancyGrid.html)
  - 2D grid map where each cell represents occupancy probability.

* - **[ComponentStatus](https://automatika-robotics.github.io/sugarcoat/apidocs/ros_sugar/ros_sugar.io.supported_types.html)**
  - [automatika_ros_sugar](https://github.com/automatika-robotics/sugarcoat/blob/main/msg/ComponentStatus.msg)
  - Lifecycle status and health information of a component.

* - **[StreamingString](../apidocs/agents/agents.ros.md#classes)**
  - [automatika_embodied_agents](https://github.com/automatika-robotics/ros-agents/tree/main/msg/StreamingString.msg)
  - String chunk for streaming applications (e.g., LLM tokens).

* - **[Video](../apidocs/agents/agents.ros.md#classes)**
  - [automatika_embodied_agents](https://github.com/automatika-robotics/ros-agents/tree/main/msg/Video.msg)
  - A sequence of image frames.

* - **[Detections](../apidocs/agents/agents.ros.md#classes)**
  - [automatika_embodied_agents](https://github.com/automatika-robotics/ros-agents/blob/main/msg/Detections2D.msg)
  - 2D bounding boxes with labels and confidence scores.

* - **[DetectionsMultiSource](../apidocs/agents/agents.ros.md#classes)**
  - [automatika_embodied_agents](https://github.com/automatika-robotics/ros-agents/tree/main/msg/Detections2DMultiSource.msg)
  - List of 2D detections from multiple input sources.

* - **[PointsOfInterest](../apidocs/agents/agents.ros.md#classes)**
  - [automatika_embodied_agents](https://github.com/automatika-robotics/ros-agents/tree/main/msg/PointsOfInterest.msg)
  - Specific 2D coordinates of interest within an image.

* - **[Trackings](../apidocs/agents/agents.ros.md#classes)**
  - [automatika_embodied_agents](https://github.com/automatika-robotics/ros-agents/blob/main/msg/Trackings.msg)
  - Object tracking data including IDs, labels, and trajectories.

* - **[TrackingsMultiSource](../apidocs/agents/agents.ros.md#classes)**
  - [automatika_embodied_agents](https://github.com/automatika-robotics/ros-agents/tree/main/msg/TrackingsMultiSource.msg)
  - Object tracking data from multiple sources.

* - **[RGBD](../apidocs/agents/agents.ros.md#classes)**
  - [realsense2_camera_msgs](https://github.com/IntelRealSense/realsense-ros)
  - Synchronized RGB and Depth image pair.

* - **[JointTrajectoryPoint](../apidocs/agents/agents.ros.md#classes)**
  - [trajectory_msgs](https://docs.ros2.org/foxy/api/trajectory_msgs/msg/JointTrajectoryPoint.html)
  - Position, velocity, and acceleration for joints at a specific time.

* - **[JointTrajectory](../apidocs/agents/agents.ros.md#classes)**
  - [trajectory_msgs](https://docs.ros2.org/foxy/api/trajectory_msgs/msg/JointTrajectory.html)
  - A sequence of waypoints for joint control.

* - **[JointJog](../apidocs/agents/agents.ros.md#classes)**
  - [control_msgs](https://github.com/ros-controls/control_msgs)
  - Immediate displacement or velocity commands for joints.

* - **[JointState](../apidocs/agents/agents.ros.md#classes)**
  - [sensor_msgs](https://docs.ros2.org/foxy/api/sensor_msgs/msg/JointState.html)
  - Instantaneous position, velocity, and effort of joints.
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
* - **Reactive**
  - Executes in response to trigger. A trigger can be either incoming messages on one or more trigger topics, OR an `Event`.
* - **Action Server**
  - Executes in response to an action request. Components of this type execute a long running task (action) and can return feedback while the execution is ongoing.
```

## Health Check and Fallback

Each component maintains an internal health state. This is used to support fallback behaviors and graceful degradation in case of errors or resource unavailability. Health monitoring is essential for building reliable and resilient autonomous agents, especially in real-world environments.

Fallback behaviors can include retry mechanisms, switching to alternate inputs, or deactivating the component safely. For deeper understanding, refer to [Sugarcoat🍬](https://automatika-robotics.github.io/sugarcoat/design/fallbacks.html), which underpins the lifecycle and health management logic.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package automatika_embodied_agents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.3.3 (2025-01-28)
------------------
* (fix) Removes python dependencies from package manifest until package names merged in rosdistro
* Contributors: ahr

0.3.2 (2025-01-28)
------------------
* (docs) Updates docs for conversational agent and SpeechToTextConfig
* (feature) Adds vad, audio feautres and wakeword classification classes based local onnx models
* (feature) Adds utility function for downloading models and status classes for speech processing
* (feature) Adds configuration for wakeword detections in speechtotext component
* (fix) Fixes error in ollama client where tool calls are received without output content
* (fix) Adds a fix to map encoding where it can start with a single detections layer
* (refactor) Makes component name non-optional in components to avoid name conflicts
* (fix) Fixes error for long prompts when checking if prompt is a filename
* (refactor) Removes pytorch as a dependency and runs VAD model with onnxruntime
* (refactor) Makes warmup a property of model components that defaults to false
* (feature) Adds utility method to download onnx model files
* (refactor) Replaces info with debug to reduce logging spam
* (fix) Fixes getting logging severity level for jazzy onwards
* (fix) Adds minor improvements to branching for llm and mllm components
* (chore) Cleansup dependencies for packaging
* (chore) Adds dependency for sugar and removes unnecessary python dependencies from packaging
* (fix) Corrects import of Topic class
* (docs) Removes redefinition of Topic and corrects links to ROS Sugar
* (fix) Changes topic in base component to be directly inherited from ROS Sugar for consistency accross packages
* (feature) Adds warmup functions to all model based components
* (refactor) Removes pillow as a dependancy
* (refactor) Removes overrrides from components and adds custom meathods instead
* (feature) Adds warmup to vision component for displaying stats on init
* (fix) Adds fix for correct colors in cv2 visualization
* (fix) Adds node name as window name for visualization in vision component
* (feature) Adds cv2 based visualization option to vision component
* (refactor) Reduces branching in execution step for components
* (chore) Combines agents and agents_interfaces to one package
* (chore) Changes deb package name
* (fix) Fixes raising error in model initialization for roboml clients
* (refactor) Adds passing additional agent types to ros sugar
* (fix) Fixes error messages when wrong component inputs/outputs are passed
* (feature) Adds support for CompressedImage msg type in components
* (feature) Adds option to deploy vision models using tensorrt
  Works with roboml
* (fix) Fixes check on sufficient topics in component validation
* (fix) Fixes a bug in topic validation
* (fix) Fixes validation of topics in components
* (refactor) Changes handling of image messages for publication
  - Adds support for CompressedImage messages
  - Gathers image messages directly in vision component instead of getting them back from clients
* (feature) Adds frame_id to trackings publisher and updates msg and callback
* (feature) Adds boxes to vision tracking message
* Contributors: ahr, mkabtoul

0.3.1 (2024-10-29)
------------------
* (chore) bump version 0.3.0 -> 0.3.1
* (feature) Adds support for using tool calling in LLM components in multiprocess execution
* Contributors: ahr

0.3.0 (2024-10-28)
------------------
* (chore) bump version 0.2.0 -> 0.3.0
* (chore) Adds bumpver config
* Merge pull request `#14 <https://github.com/automatika-robotics/ros-agents/issues/14>`_ from automatika-robotics/feature/external_processors
  Adds support for running components as separate processes
* (docs) Updates docs based on ROS Sugar version update
* (fix) Fixes bug in registering triggers with components
* (refactor) Simplifies by adding direct serialization of clients and triggers
* (refactor) Removes gratuitous logging from utils
* (fix) Minor bug fixes for components to run in multiprocessing
  - Fixes trigger assignment for components
  - Handles private attributes of attrs classes
  - Fixes component and config init in common executable
* (fix) Fixes serializing log level in clients
* (fix) Fixes minor bugs in utils, components, configs and models
* (feature) Adds support for running components in multiple processes
  - Adds common executable to the package for ROS Sugar launcher
  - Refactors components to be serializable
  - Adds serialization to clients
  - Minor type hint changes for compatibility with older versions of ROS
* (fix) Adds the correct check for external processors given new ros-sugar implementation
* Contributors: ahr

0.2.0 (2024-09-28)
------------------
* (chore) Bump up the version
* Merge pull request `#13 <https://github.com/automatika-robotics/ros-agents/issues/13>`_ from automatika-robotics/feature/better_clients
  Adds enhanced functionality in clients specifically for LLM and MLLM components
* (feature) Adds tool calling for LLM component using the OllamaClient
* (fix) Fixes rag results in templated inputs to LLMs which do not contain input
* (refactor) Makes named models subclasses of TransformersLLM and TransformersMLLM for easier handling in roboml client
* (fix) Fixes key error in ollama client response retreival
* (fix) Adds flag for chat history for chat history reset and fixes logging
* (feature) Adds TransformersLLM and TransformersMLLM models for roboml clients
* (fix) Removes history reset phrase from model definitions and add system prompt for LLMs and derivates
* (refactor) Changes model component to have execution step as an abstract method implemented by child components
* (fix) Changes ollama client inference call to use chat endpoint
* (feature) Adds chat history management to llm and mllm components
* (docs) Clarifies handling of RAG results for llm component
* (fix) Fixes bug in rag result handling for llm component
* (fix) Removes default init_timeout from models
* (refactor) Moves roboml resp client dependancies inside the client initialization
* (fix) Explicity exposes QoSConfig in ros module
* (refactor) Replaces map_meta_data parameter with map_topic for MapEncoding component
* (refactor) Removes direct dependancy on pypdf
* (fix) Changes map meta data topic to type OccupancyGrid
* (feature) Adds audio options to chainlit client
* (fix) Removes unused imports
* (fix) Fixes the initialization of map encoding and semantic router components
* (refactor) Fixes imports and refactors code according to latest version of ROS sugar
* (fix) Fixes passing the config in components to parent base component
* (fix) Fixes ROS sugar import for BaseTopic
* (refactor) Removes auto_ros as a dependency
* (feature) Adds init_on_activation flag to all implemented clientsc
* (feature) Seperates abstract methods from callable methods in db client base
* (feature) Seperates callable methods, from abstract methods in client base class
* Contributors: ahr

0.1.1 (2024-09-05)
------------------
* (feature) Adds component action for adding points to map collection (`#12 <https://github.com/automatika-robotics/ros-agents/issues/12>`_)
  * Makes version compliant with ROS convention
* (chore) Adds license declaration in setup.py
* Bumps version number and adds license information
* Initial release 0.1.1a
* Contributors: ahr, mkabtoul

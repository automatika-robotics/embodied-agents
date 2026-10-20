^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package automatika_embodied_agents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.8.0 (2026-10-20)
------------------
* (chore) Bumps required min sugarcoat version
* (feature) Adds support for decision models in ollama client and shows model pull progress
* (feature) Adds model download in thread in model base along with model progress
* (fix) Fixes stamping messages correctly in vision processing
* (chore) Removes deprecated map encoding component
* (fix) Adds correct handling of additional types in recipes which include other sugarcoated packages
* (docs) Updates examples with chat history set to true
* (feature) Adds getting and resetting chat history as actions for llm
* (feature) Adds handling cortex planning calls as chat history
* (fix) Serialize pydantic tool calls returned in ollama client
* (feature) Adds summarization and use of chat history in cortex planning loop
* (feature) Adds calculation of history budget from models reported context for supporting clients
* (feature) Adds turn by turn chat history with turn and token budget based trimming
* (fix) Fixes resetting chat history with the reset phrase preserving system prompt
* (fix) Fixes sending images correctly as part of the last message to generic and ollama clients
* (docs) Updates docs for scratch functions
* (feature) Adds a scratch pad for the planner to write custom functions for actions and internal events when none available
* (fix) Fixes passing think option to ollama models
* (chore) Adds decision model example
* (feature) Adds calling component actions for state gathering in decision component
* (refactor) Adds overriding base component method for creating service clients in model components
* (refactor) Simplifies external component tool calling in llm
* (refactor) Lifts calling component actions on other components to model component
* (feature) Adds a component action to get current task
* (feature) Makes cortex monitoring loop use tool calls instead of text response parsing and handled failures and skipped steps correctly
* (feature) Adds DECISION mode in semantic router for using decision models
* (feature) Adds a decision component for calling decision models
* (feature) Adds a decision msg and its sender/callback for one answer from a decision component
* (feature) Adds calling the systemone decision api in the generic http client
* (feature) Adds generic decision model wrapper
* (fix) Fixes deinit in clients to keep the connection open for safe restart
* (fix) Fixes bug in chroma db deinit and creating a collection without a distance function
* (chore) Fixes integration tests and remove the ignore in the CI
* (docs) Adds minor edit in cortex docs
* (fix) Adds a small gil release sleep in vla busy loop when inputs not received
* (fix) Fixes tool call results being sent back to model in llm component
* (fix) Fixes ollama model init for models already on disk are not pulled
* (refactor) Standardizes tool call parsing for OpenAI compatible servers
* (refactor) Adds calling external processors based on the new contract in llm and vla components
* (fix) Fixes using events as triggers in components by returning the correct action return type
* (refator) Replaces dots with hyphens in tool names for cortex
* (feature) Adds plugin tools to be compilable into routines and removes duplicates when sending tools to planner
* (feature) Makes planning actions compilable into a routine
* (refactor) Registers and calls plugins actions via the monitor
* (feature) Adds dynamically adding events/actions during planning step in cortex
* (feature) Adds compilation of plan steps into monitored routines in cortex execution
* (feature) Adds adding headless routines to cortex for being called and monitored as action servers
* (feature) Adds a wait tool in cortex to dwell on actions after calling them so they can take effect
* (fix) Fixes how cortex parses action responses
* (feature) Adds checksum verification to local model downloads that we manage ourselves
* (fix) Strips the scheme if added to the lerobot client host with a warning
* (fix) Fixes correct handling of wss endpoints in roboml websocket client
* (feature) Adds a ca cert option to provide a self signed certificate for trusted self hosted model and db servers
* (feature) Adds proper serialization of key env variable and warning when sending over http
* (feature) Adds issuing a warning for non encrypted external connections when not to loopback addresses
* (feature) Adds registration of sensor plugins in planning prompt and adds proper execution of plugin actions as launcher process methods with kwargs
* (feature) Adds ui callback for trackings
* (feature) Adds ui callback for jointstate
* (fix) Fixes output of streaming string and adds ui element for video
* (refactor) Makes goal sending to action servers by cortex conditional on previous goal being cancelled
* (feature) Updates action return types based on requirement from new sugarcoat version
* (fix) Fixes warnings and adds a debug log in case no trusted 3d lifts are available
* (refactor) Adds depth factor when creating depth detector so conversion and filtering happens in core
* (refactor) Reads depth validity directly from lifted boxes and removes calculation perception3d utils
* (refactor) Uses source index from depth detector for lifted boxes and give depth images in the right format for zero copy
* (fix) Adds configuring transform providers when creating subscribers in base component overload
* (refactor) Uses log_once from base component instead of rclpy once=True
* (feature) Adds 3d lifitng using pointcloud as depth for vision and mllm components
* (feature) Enables pointcloud type to be used in 3d utilities for creating a depth detector
* (feature) Adds image resizing config option to reduce downstream computation of motion detection
* (fix) Stops cloud points being accumulated if odom is wired but hasnt arrived to avoid frame mismatch
* (feature) Adds a ring buffer to publish before and after frames when motion is detected from video
* (feature) Adds turn threshold so that motion detection is paused on robots rotating in please similar to robots with above threshold speed
* (fix) Fixes handling of single channel (IR) image streams for motion detection
* (feature) Adds per pixel flow magnitude thresolding for optical flow based motion detection
* (feature) Takes out globals shifts when calculating frame difference to make motion detection more robust
* (fix) Fixes motion detection using images, gives a default method and adds blurring before frame difference calculation
* (refactor) Simplifies detections2d callbacks for string creation and removes unnecessary utils
* (feature) Allows detections3d to be used in llm and mllm context
* (feature) Adds storing 3d lifted object detections in memory
* (fix) Fixes interop vs intraop param setting for ncpu in voice models as well (irrelevant for perf though)
* (feature) Gets local vision model width and height from its input spec to resize incoming input and sets correct param for ncpu
* (feature) Adds plugin config serialization to executable
* (fix) Fixes the check for trigger topics when input topic is replaced before activation
* (feature) Adds a run task action to the mllm component for cortex triggered grounding
* (feature) Adds labels to 2d boxes from grounding in mllm component
* (feature) Adds grasp orientation and tolerance params for both pick and pose goals
* (refactor) Removes explicit named targets from the config as params (to be supplied only in the srdf)
* (feature) Adds side approach in the pick action as a configurable maneuver for grippers that need to approach horizontally
* (feature) Adds adding acm for grasping contact to plan around and freezing refreshes during pick and place
* (feature) Removes planning padding for attaching the objects post grasp
* (feature) Returns camera position from 3d lifting in order to push bbox centers to appoximate middle of the object
* (feature) Adds label filtering param to ignore unrelated background objects
* (feature) Adds a forward kinematics client to get end effector orientation for poses where orientation is not given (identity), specially in cartesian motion
* (feature) Adds a seperate cartesian group config param for use with orientation-tracking IK for underactuated arms
* (feature) Freezes scene when an object is held to avoid mixing labels between similar objects
* (feature) Adds lifting to 3d and publishing detections3d in vlm component
* (feature) Adds 3d contract inputs and contract checking to vlm component
* (feature) Adds 3d lift config to vlm config
* (feature) Adds place routine and updates docstring in the component
* (feature) Adds pick and place routine for action server execution
* (feature) Adds a target match radius around a target for pick and place given by pose
* (feature) Adds PICK and PLACE move modes
* (feature) Caches object geometry for objects in the scene
* (feature) Adds config knob for pick and place clearance
* (feature) Enhance moveit action with pick and place modes
* (feature) Adds action and functionality for refreshing planning scene from objects received from detections on_goal, continuously or manually via action
* (feature) Adds attaching and detaching links to planning scene
* (feature) Adds scene planning component actions
* (feature) Adds scene planning service clients for the moveit component
* (feature) Adds detections3d as input to moveit component
* (feature) Adds multiprocessing launch for the new features in vision component
* (feature) Adds planning related config params for moveit component, adds validators to all literals
* (feature) Adds moveit utils for scene planning
* (feature) Adds 3d lifting using depth detector in vision component for detections3d output
* (feature) Add Detection3d as input to the vision component along with vision contract
* (feature) Adds config options for lifting 2d detections to 3d in vision component
* (chore) Adds tests for vision component
* (feature) Adds a simplified contract in vision component for inputs/outputs
* (fix) Fixes checking for truthiness of ndarray
* (fix) Removes guard that would have raised with an ndarray
* (fix) Fixes publishing grounding outputs from mllm with multiple camera inputs
* (refactor) Makes Trackings and Detections2d raise if given detections/trackings from multiple cameras
* (fix) Fixes creating the depth detector with the correct rotation convention in ROS
* (refactor) Adds a helper for creating 2d bboxes
* (feature) Adds utilities for lifting 2d bboxes to 3d metric space via kompass-core
* (feature) Adds detection3d msg callback and sender
* (feature) Adds 3d bbox and detection messages
* (feature) Adds frame id to all vision messages
* (chore) Updates CI for testing moveit component
* (feature) Adds moveit action server component for calling moveit services
* (feature) Adds configuration for new moveit component
* (chore) Adds tests for moveit utils
* (feature) Adds utils for building moveit msgs and service requests and parsing them
* (feature) Adds an action msg for MoveIt action server component
* (fix) Fixes creating renampe_map (and _features) in lerebot policy
* (fix) Fixes the action branches for cleanup where inputs not received on time or cancel is requested
* (fix) Adds depth camera feature check and appropriate warning along with using the observation prefix
* (feature) Adds utility function for getting channels from dataset spec and exposes observation prefix as a constant
* (fix) Adds iterator finalization for llm stream handling so that local model iterators dont wait for garbage collection if an exception occurs while publishing
* (fix) Adds within think block termination warning to streaming output path in llms
* (fix) Fixes monotonic duration for joints trajectory and joint jog message publishing (one action tick per command)
* (chore) Updates local llm, local vlm and router tests
* (fix) Fixes setting stream to true for local vlm in the config
* (feature) Adds guard for consecutive requests to local vlm and adds streaming option for local vlm
* (feature) Adds handling concurrent calls to local llm and tool call extraction
* (fix) Fixes stripping thinking tokens when thinking block is cut off due to max new tokens
* (fix) Fixes semantic router init of local models
* (feature) Adds new defaults for local llm and vlm, use of model options and correct resolution of different model families
* (chore) Adds tests for modified local stt
* (feature) Adds wakeword detection with sherpa's key word spotter, customized wake words via config
* (feature) Adds model archive download utility
* (feature) Adds model options for loading all tts models available in sherpa-onnx
* (feature) Adds model options for local stt and updates default to parakeet
* (chore) Adds tests for local tts
* (fix) Fixes loading local models from disk
* (feature) Adds streaming generation to local tts model use
* (feature) Handles all sherpa model families and streaming generation
* (feature) Adds additional sherpa generation options and updates default model to pocket
* (fix) Handles aborted goals and event based termination reset correctly
* (fix) Fixes reading images for inference input when reading tick and image callback tick have a mismatch
* (feature) Adds urdf to policy unit conversion along with a heuristic warning path
* (fix) Fixes logging of stop event
* (feature) Adds policy action units config option for auto conversion
* (feature) Adds utility to convert joint limits to units utilized by the policy
* (refactor) Changes default max_new_tokens to 512
* (fix) Adds missing abstract method to vla to avoid abc enforcement in newer rclpy
* (feature) Updates the VLA component to set preset agg functions and take depth images as input
* (feature) Makes dtype degradation explicit based on lerobot behavior for feature checking
* (feature) Adds preset aggregator function as config param
* (feature) Adds getting depth only images from rgbd callback
* (feature) Updates lerobot client with a policy type error (if policy is not supported) and a rename map
* (feature) Exposes latest lerobot policies for usage in available model types
* (fix) Fixes bugs in JointState and JointJog senders
  - Correctly copies accelerations
  - time_from_start converted to proper Duration
  - Fixes duration in JointJog
* (feature) Adds the notion of coherent clusters > min cluster size, for use as motion evidence to reduce false positives
* (chore) Updates tests for motion detection
* (feature) Adds new params to config for the accumulation window and adds some rational defaults
* (feature) Adds differencing against union of a time-window of scans to detect voxels with motion so that non standard scan patterns like livox can be covered
* (chore) Adds motion detection example and updates tests
* (chore) Deprecates images to video component in favour of the new motion detection component
* (feature) Adds a new motion detector component to detect motion from images or pointclouds and publish videos or centers of motion
* (feature) Adds prior memory addition to memlayers and removes redundant attributes
* (feature) Adds rate sampled outputs for custom types for api/UI and modified streaming string output to give commulated string for mid stream consumers
* (fix) Fixes parsing tool call args from openAI compatible clients in cortex
* (fix) Fixes a serialization bug in generic client affecting initialization setup
* (fix) Disables launch_ros from testing
* (fix) Disables incompatible launch_testing pytest plugin in CI
* (fix) Fixes multiprocessing launch check for generic client (`#34 <https://github.com/automatika-robotics/embodied-agents/issues/34>`_)
* Contributors: ahr

0.7.4 (2026-06-05)
------------------
* (chore) Bumps up sugarcoat version
* (refactor) Adds build url helper for building url schema if none provided
* (fix) Allows say method to raise exceptions to be handled by parent 'execute_method' and fixes attribute check
* (feature) Adds robot plugin serialization/deserialization to the executable
* (fix) Adds using pre-init logger in methods executed before activation
* (fix) Fixes an error where TLS endpoints couldnt be called by clients
* (feature) Adds plugin actions to actions available to cortex and augments planning prompt with robot description from the plugin
* Contributors: ahr, mkabtoul

0.7.3 (2026-05-08)
------------------
* (fix) Fixes the helper method typo
* (fix) Fixes validate topics check to include comparing the associated ROS2 message type, resolves `#30 <https://github.com/automatika-robotics/embodied-agents/issues/30>`_
* (fix) Runs callback disambiguation on layers topics using Sugarcoat reparse_inputs_callbacks, resolves `#31 <https://github.com/automatika-robotics/embodied-agents/issues/31>`_
* Contributors: ahr, mkabtoul

0.7.2 (2026-05-06)
------------------
* (chore) Updates complete agent recipes
* (chore) Updates tool calling and go to x recipes to use memory
* (feature) Enables passing component actions as tools to the llm component
  - Executed through service calls
* (refactor) Makes response handling from component action execution a utility
* (chore) Updates go to x recipe to use the memory component
* (fix) Fixes registering memory tools with llms before lifecycle configuration
* (chore) Updates tool calling recipe to use the memory component
* (chore) Updates semantic map recipes to use the memory component
* Contributors: ahr

0.7.1 (2026-05-02)
------------------
* (chore) Bumps minimum sugarcoat version
* (chore) Adds test for memory component
* (chore) Adds an example recipe for using cortex with memory
* (chore) Adds both component level and launcher level failure recovery in multiprocessing example
* (chore) Updates docstrings. Adds deprecation for MapEncoding component in favour of memory component
* (feature) Augments memory prompt for interoception tools
* (feature) Adds mem layer alias for map layers with internal state flag for interoception topics
* (feature) Adds memory specific planning prompt augmentation
* (feature) Adds better failure handling in cortex for task summary
* (feature) Adds serialization, deserialization of clients for memory component
* (fix) Fixes init/deinit of ollama based embedding models
* (feature) Adds classification in memory component actions for planning and execution actions
* (feature) Updates cortex to utilize executed component action outputs and replan if the plan has not fully executed
* (feature) Overrides component action decorator to differentiate between planning and execution actions in agents
* (feautre) Changes component actions and fallbacks to raise errors and return meaningful responses on happy path
* (feature) Adds a store specific memory method for cortex to write arbitrary runtime memories
* (feautre) Adds information about perception layers in the memory components inspect method
* (feature) Adds memory component using emem that exposes basic memory management tools as component actions
* (chore) Removes safe_restart from base component as it has been upstreamed in sugarcoat
* Contributors: ahr

0.7.0 (2026-04-11)
------------------
* (feature) Adds an optional output topic in cortex for capturing outputs when an action is not needed
* (docs) Updates docs for component actions setup
* (chore) Bumps up required sugarcoat version
* (fix) Fixes model class type check in roboml client
* (feature) Adds Action goal final status to Cortex feedback lines
* (fix) Fixes converter for detections message to handle empty detections
* (feature) Adds tracking action to vision component
* (fix) Fixes publishing to multiple output topics in model component
* (feature) Adds a safe restart context manager to base component
* (fix) Fixes sender for tracking based on new roboml api
* (chore) Adds CI workflow to run tests
* (feature) Adds a describe action to the vlm component
* (fix) Standardizes param name in POI publishing
* (fix) Increases default max token limits
* (fix) Fixes monitoring ongoing actions and feedback status update for step decision
* (fix) Fixes return value when sending action goal to component
* (feature) Adds async execution of action clients and monitoring using the main action loop
* (feature) Adds helper methods to monitor ongoing action clients and cancel their goals
* (fix) Adds fixes for python3.8 compatibility
* (refactor) Updates examples for new roboml api
* (feature) Enables registering component additional ROS entrypoints as system tools
* (feature) Updates model definitions and default checkpoints based on new version of RoboML
* (fix) Removes unnecessary config validator for stt
* (fix) Fixes empty buffer inference call in stt
* (refactor) Simplifies text to speech playback pipeline for less jitter
* (feature) Adds internal events setup to cortex launch
* (fix) Fixes say action for abrupt stopping of play on device thread
* (feature) Adds multistep planning loop
  - Separates planning and execution tools
  - inpection is used only in planning loop
  - execution handles failures when missed by llm
* (chore) Locks methods not implemented in VLA component
* (fix) Fixes tool calling for string args in llm component
* (fix) Fixes local model download cache paths
* (feature) Adds component config params inspection to _inspect_component
* (feature) Registers tools in Cortex for sending Goals to any available component Action Server
* (refactor) Adds a meaninful action name to the VLA component
* (feature) Adds running component actions as service call and captures their output for subsequent steps
* (feature) Adds tool registration from component tools and adds inspect component tool to cortex
* (feature) Adds tool descriptions to available component actions
* (feature) Adds planning and execution phases to cortex
* (feature) Extends the base Launcher to use the Cortex component as a monitor
* (feature) Adds Cortex, the master planning and execution node
* (refactor) Moves strip think token to utilities
* (feature) Adds feedback to VisionLanguageAction feedback msg
* Contributors: ahr, mkabtoul

0.6.0 (2026-03-21)
------------------
* (docs) Adds advanced guides
* (docs) Adds developer docs
* (refactor) Adds sherpa-onnx for stt and tts local models
* (refactor) Adds ggml based local llm and vlm models with llama cpp
* (feature) Adds config option to strip think tokens
* (fix) Fixes execution provider setup when executor is set to be cpu
* (refactor) Standardizes local model handling in vision model
* (docs) Updates fallback example and adds new local model fallback recipe
* (feature) Adds fallback_to_local as a model component action for switching to local model on the fly
* (chore) Adds markers for running tests that require model serving
* (chore) Adds comprehensive test coverage for components
* (fix) Fixes bug in model components
* (fix) Fixes init of local LLM in semantic router for multiprocessing setup
* (feature) Adds local models for STT and TTS
* (feature) Enables local llm in semantic router component
* (feature) Adds moondream as local vlm for VLM component
* (feature) Adds local llm model for LLM component
* (fix) Adds extra check in rgbd callback
* (refactor) Adds warning for timed components when calling take_picture or record_video
* (docs) Updates vlm based planning recipe
* Contributors: ahr, mkabtoul

0.5.1 (2026-02-16)
------------------
* (feature) Adds a record_video action to the Vision component
* Takes specific input topic name and duration to record video
* (feature) Adds a take_picture action to the vision component
* (feature) Adds arbitrary action execution to router
* (feature) Adds handling of topic and action lists in semantic router
* (refactor) Makes action methods in Vision component use extra_callback to avoid starvation
* (refactor) Increases default max new tokens param for llm/vlm configs
* (refactor) Makes changes for new api of events/actions in sugar
* (fix) Fixes semantic router handling of action routes based on new api
* (fix) Fixes warmup for model components
* (chore) Adds think option for ollama models
* (chore) Adds proper docstrings to exposed methods and classes
* (docs) Updates model definitions in examples
* (docs) Adds new prompt for llms.txt and new tutorial intros
* (docs) Fix incorrect import in docs
* (docs) Updates installation instructions
* Contributors: ahr, mkabtoul

0.5.0 (2026-01-19)
------------------
* (docs) Adds supported types table and component runtype in basics
* (docs) Adds VLA tutorial to docs and refactors them
* (docs) Adds tutorials and examples for using events/fallbacks
* (refactor) Enables health broadcast by default
* (feautre) Adds feature for passing additional model clients to components
  Component can switch model clients based on events
* (feature) Adds external events as component triggers for certain components that can be run in a timed way
* (feature) Adds stt and tts endpoints and generic model wrappers for the generic http client (openai compatible)
* (feature) Enables support for tool calls in generic http client
* (feature) Adds llms.txt in docs
* (docs) Adds explanation of both routing modes to semantic router docs
* (feature) Adds LLM based agentic routing to semantic router
* (feature) Adds timeout for event based termination if the event doesnt happen within max_timesteps
* (feature) Create observation spec form mapping dicts if dataset info.json is not provided
* (feature) Adds publisher for joint state msg
* (feature) Adds parameter to regulate observation sending rate
* (fix) Adds various fixes for running in multiprocessing (vla)
* (feature) Adds dynamically adjusting loop rate to action server (vla)
* (feature) Adds termination logic to the VLA action server
* (feature) Adds capping of actions based on joint limits before publishing
* (feature) Adds action sending in VLA for multiple datatypes
* (feature) Adds a setter for passing an external aggregation function
* (feature) Adds mapping between LeRobot dataset keys and robot urdf and topics
  - Adds warning and errors for mapped keys, topics
  - Adds creation of input for VLA
  - Adds action receiving and aggregation
* (feature) Adds utility for reading joint limits from URDF files/urls
  - Adds mapping utility function in internal joint state class
* (fix) Adds destruction of stream timer if its created for async websocket clients
* (docs) Adds documentaion for lerobot client
* (feature) Adds an async grpc client for lerobot
* (docs) Removes onnxruntime as a mandatory dependancy, its only required for local models
* (feature) Adds LeRobotPolicy wrapper in models
  Adds utility for extracting features and actions from info.json
* (feature) Implements replacing input topics to handle trigger callbacks
* (feature) Enables None triggers for Server components
* (fix) Fixes unnecessary logging causing error before component activation
* (fix) Fixes deserialization of map layers and routes in multiprocess run
* Contributors: ahr, mkabtoul

0.4.3 (2025-11-07)
------------------
* (docs) Adds instructions for using the dynamic web UI
* (chore) Removes tiny web client
* (fix) Removes StreamingString as input option for LLM/MLLM
* (feature) Adds logging output from StreamingString in the UI
* (chore) Adds warning for using templates with StreamingString
* (chore) Adds alias VLM for MLLM component
* (fix) Adds detections as handled output in mllm component
* (feature) Adds callback for points of interest msg
* (feature) Adds ui callback for rgbd images
* (docs) Updates docs for planning model example
* (fix) Fixes publishing images as part of detection msgs
* (refactor) Updates callbacks for video and rgbd type messages
* (feature) Adds handling of additional types from other sugar derived packages in agent's components
* (feature) Adds UI element definitions for custom types
* (feature) Adds ui callbacks for Detections and DetectionsMultiSource
* (feature) Adds utility for drawing images with bounding boxes
* (fix) Fixes passing topic from sugar derived packages to agents components
* (feature) Adds Detection2D as allowed input in map encoding component
* (feature) Adds callback for Detections2D and their use in llm/mllm components
  - Streamlines names of detection msgs
  - Streamlines names of tracking msgs
* (fix) Gets raw msg data in execution step to avoid calling get_output twice
* (chore) Adds websockets as an explicit dependency
* Contributors: ahr, mkabtoul

0.4.2 (2025-09-03)
------------------
* (feature) Adds udp streaming to IP:PORT as an option to TextToStream component when play_on_device is enabled
* (docs) Updates docs to use new web based client
* (feature) Adds processing of audio messages in web client
* (chore) Removes chainlit based client
* (feature) Adds a custom webclient to replace chainlit
* (feature) Adds persistent ros node in web client for async stream handling
* (feature) Adds warning when not using streaming string msg_type with streaming enabled in components
* (feature) Adds streaming string msg for managing streams in external clients
* (docs) Adds recipe for vision guided point navigation
* (fix) Fixes empty image input for Detection2D msg publication
* (fix) Fixes websocket receiving in text to speech
* (fix) Fixes keyword argument in detection and tracking publishing
* (feature) Adds publishing a singular detection or tracking message from the vision component
* Contributors: ahr, mkabtoul

0.4.1 (2025-07-10)
------------------
* (docs) Updates docs for using planning based MLLMs
* (feature) Adds options to get RGBD array from rgbd message callback
* (refactor) Breaks complex functions and fixes warmup result logging
* (feature) Adds support for planning mllm models, starting with robobrain2.0
* (docs) Adds streaming to conversational agent example
* Contributors: ahr, mkabtoul

0.4.0 (2025-06-18)
------------------
* (docs) Adds international readme files
* (feature) Adds better connection error messages in clients, adds installation instructions
* (chore) Adds debian packaging workflow
* (docs) Updates installation instructions
* (chore) Updates package names .. ROS Agents -> EmbodiedAgents
* (feature) Adds a GenericHTTPClient for using llm and mllm models served on any OpenAI compatible API
* (feature) Adds ollama specific inference options to OllamaModel and client
* (feature) Adds MeloTTS model to model definitions
* (feature) Adds say text method to text to speech for invoking with events
* (feature) Adds streaming playback for streaming input in speeech to text component
* (fix) Fixes clearing old output in the vision component when getting subscription data in a timed manner
* (feature) Adds tensorrt as an onnx provider option for local models
* (refactor) Removes sounddevice as a dependancy for text to speech component
* (feature) Adds local classification model for Vision component
  Default model: DEIM: DETR with Improved Matching for Fast Convergence by Huang et al.
* (feature) Adds warnings if device for local models is set to GPU and runtime is not available
* (feature) Adds hypothesis buffer for publishing confirmed transcripts when using streaming
* (feature) Adds asynchronous receiving for streaming websockets client in speech to text component
* (refactor) Adds getting inference params just once during node configuration
* (fix) Fixes handling of model init params and sending np arrays during inference
* (feature) Adds asynchronous publishing of response in LLM component when streaming with websocket client
* (feature) Adds local embeddings option using sentence-transformers to ChromaDB client
* (feature) Adds ChromaDB http client with ollama embeddigs
* (feature) Adds streaming with websocket client in llm component
* (fix) Fixes error message for required topics when they can be either/or
* (feature) Adds support for RGBD messages (in realsense style)
* (feature) Adds async websocket client for roboml
* (refactor) Marks child threads as daemons for smoother termination
* (feature) Adds break_character to llm component config to handle breaking streaming output into chunks for publishing
* (feature) Adds streaming to roboml http client for text data
* (feature) Adds streaming output handling to ollama client
* (refactor) Adds set_system_prompt to components and removes it from model config
  The same model can be called with various system prompts by different components
* (fix) Fixes typing bugs for for python 3.8 compatibility
* Contributors: ahr, aleph-ra, mkabtoul

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

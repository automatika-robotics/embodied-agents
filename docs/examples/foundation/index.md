# Foundation Recipes Overview

Welcome to the foundation of **EmbodiedAgents**.

Before building complex, self-evolving systems, the recipes in this section introduce you to the core **Components**, the primary execution units that drive your physical agents.

## The Power of Modularity

_EmbodiedAgents_ treats every capability, whether it's hearing (**SpeechToText**), speaking (**TextToSpeech**), seeing (**Vision** / **VLM**), or thinking (**LLM**), as a modular, production-ready component. These are not just wrappers; they are robust ROS2 Lifecycle Nodes with all the allied functionality required for utilizing the ML models in a simple, Pythonic abstraction.

In these foundational recipes, you will see how the framework's "separation of concerns" works in practice:

- **Pythonic Graphs**: See how to describe your agent's architecture in pure Python, avoiding the complexity of traditional ROS development.
- **Multi-Modal Interaction**: Combine text, images, and audio seamlessly. You will learn to route data between components, feeding the output of a Vision model into an LLM, or turning an LLM's text responses into a spatio-temporal map that the robot can use.
- **Clients & Models**: Learn how to utilize models and vector DBs, swapping and reusing them across various functional components. Connect your components to local inference engines (like **Ollama** or **RoboML**) or cloud APIs just by changing the Client configuration.

These recipes cover the journey from a basic multimodal conversational agent to fully embodied interactions involving semantic mapping, response routing and tool usage.

## Recipes

- **[A Simple Conversational Agent](conversational.md)**
  Build your first "Hello World" agent that uses **STT**, **VLM** and **TTS** components to hold a simple dialogue multimodal dialogue, introducing the basics of component configuration and clients.

- **[Prompt Engineering](prompt_engineering.md)**
  Learn how to use **templates** at the topic or component level to create dynamic, context-aware system prompts that guide your agent's behavior.

- **[Semantic Map](semantic_map.md)**
  Utilize the **MapEncoding** component to give your robot a spatio-temporal working memory, allowing it to store and retrieve semantic information about its environment using a Vector DB.

- **[GoTo X](goto.md)**
  A navigation recipe that demonstrates how to connect language understanding with physical actuation, enabling the robot to move to locations based on natural language commands.

- **[Tool Calling](tool_calling.md)**
  Empower your agent to act on the world by giving the **LLM** access to executable functions (tools), enabling it to perform tasks beyond simple text generation.

- **[Semantic Routing](semantic_router.md)**
  Implement intelligent control flow using the **SemanticRouter**, which directs messages to different graph branches based on their meaning rather than hard-coded topic connections.

- **[A Complete Agent](complete.md)**
  An end-to-end example that combines perception, memory, and reasoning components into a cohesive, fully embodied system.

- **[Utilizing Multimodal Planning](planning_model.md)**
  Configure a specific **VLM** to act as a high-level planner, decomposing complex user instructions into a sequence of executable low-level actions.

- **[Robot Manipulation](vla.md)**
  Control physical actuators using end-to-end Vision Language Action (VLA) models. This recipe demonstrates how to use the VLA component and LeRobot policies to map visual inputs directly to robot joint commands.

- **[Event Driven Robot Manipulation](vla_with_event.md)**
  Build a closed-loop agent where a VLM acts as a referee for a VLA. This recipe demonstrates how to use Events to automatically stop physical actions based on visual verification of task completion.

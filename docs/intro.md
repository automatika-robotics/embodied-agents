<div>
  <img src="_static/EMBODIED_AGENTS_LIGHT.png" class="only-light" />
  <img src="_static/EMBODIED_AGENTS_DARK.png" class="only-dark" />
</div>
<br/>

# EmbodiedAgents 🤖

_EmbodiedAgents_ is a fully-loaded framework, written in pure ROS2, for creating interactive physical agents that can understand, remember, and act upon contextual information from their environment.

- **Production Ready Physical Agents:** Designed to be used with autonomous robot systems that operate in real world dynamic environments. _EmbodiedAgents_ makes it simple to create systems that make use of Physical AI.
- **Intuitive API**: Simple pythonic API to utilize local or cloud based ML models (specifically **Multimodal LLMs** and other **transformer based architectures**) on robots, with all the benefits of component lifecycle management, health monitoring and fallback mechanisms to make your agents robust.
- **Self-referential and Event Driven**: An agent created with _EmbodiedAgents_ can start, stop or reconfigure its own components based on internal and external events. For example, an agent can change the ML model for planning based on its location on the map or input from the vision model. _EmbodiedAgents_ makes it simple to create agents that are self-referential [Gödel machines](https://en.wikipedia.org/wiki/G%C3%B6del_machine).
- **Semantic Memory**: Integrates vector databases, semantic routing and other supporting components to quickly build arbitrarily complex graphs for agentic information flow. No need to utilize bloated "GenAI" frameworks on your robot.
- **Made in ROS2**: Utilizes [ROS2](https://docs.ros.org/en/kilted/index.html) as the underlying distributed communications backbone. Theoretically, all devices that provide a ROS2 package can be utilized to send data to ML models, with callbacks implemented for most commonly used data types and infinite extensibility.

Checkout [Installation Instructions](installation.md) 🛠️

Get started with the [Quickstart Guide](quickstart.md) 🚀

Get familiar with [Basic Concepts](basics/index.md) 📚

Dive right in with [Examples](examples/index.md) ✨

## Contributions

_EmbodiedAgents_ has been developed in collaboration between [Automatika Robotics](https://automatikarobotics.com/) and [Inria](https://inria.fr/). Contributions from the community are most welcome.

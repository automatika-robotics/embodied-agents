<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/EMBODIED_AGENTS_DARK.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/_static/EMBODIED_AGENTS_LIGHT.png">
  <img alt="EmbodiedAgents Logo." src="docs/_static/EMBODIED_AGENTS_DARK.png">
</picture>
<br/>

🇨🇳 [简体中文](docs/README.zh.md) | 🇯🇵 [日本語](docs/README.ja.md)

**_EmbodiedAgents_** is a fully-loaded framework, written in pure ROS2, for creating interactive physical agents that can understand, remember, and act upon contextual information from their environment.

- **Production Ready Physical Agents:** Designed to be used with autonomous robot systems that operate in real world dynamic environments. _EmbodiedAgents_ makes it simple to create systems that make use of Physical AI.
- **Intuitive API**: Simple pythonic API to utilize local or cloud based ML models (specifically **Multimodal LLMs** and other **transformer based architectures**) on robots, with all the benefits of component lifecycle management, health monitoring and fallback mechanisms to make your agents robust.
- **Self-referential and Event Driven**: An agent created with _EmbodiedAgents_ can start, stop or reconfigure its own components based on internal and external events. For example, an agent can change the ML model for planning based on its location on the map or input from the vision model. _EmbodiedAgents_ makes it simple to create agents that are self-referential [Gödel machines](https://en.wikipedia.org/wiki/G%C3%B6del_machine).
- **Semantic Memory**: Integrates vector databases, semantic routing and other supporting components to quickly build arbitrarily complex graphs for agentic information flow. No need to utilize bloated "GenAI" frameworks on your robot.
- **Made in ROS2**: Utilizes ROS2 as the underlying distributed communications backbone. Theoretically, all devices that provide a ROS2 package can be utilized to send data to ML models, with callbacks implemented for most commonly used data types and infinite extensibility.

Checkout [Installation Instructions](https://automatika-robotics.github.io/embodied-agents/installation.html) 🛠️

Get started with the [Quickstart Guide](https://automatika-robotics.github.io/embodied-agents/quickstart.html) 🚀

Get familiar with [Basic Concepts](https://automatika-robotics.github.io/embodied-agents/basics.html) 📚

Dive right in with [Examples](https://automatika-robotics.github.io/embodied-agents/examples/index.html) ✨

## Installation 🛠️

### Install a model serving platform

The core of _EmbodiedAgents_ is agnostic to model serving platforms. It currently supports [Ollama](https://ollama.com), [RoboML](https://github.com/automatika-robotics/robo-ml) and any platform or cloud provider with an OpenAI compatible API (e.g. [vLLM](https://github.com/vllm-project/vllm), [lmdeploy](https://github.com/InternLM/lmdeploy) etc.). Please install either of these by following the instructions provided by respective projects. Support for new platforms is being continuously added. If you would like to support a particular platform, please open an issue/PR.

### Install _EmbodiedAgents_ (Ubuntu)

For ROS versions >= _humble_, you can install _EmbodiedAgents_ with your package manager. For example on Ubuntu:

`sudo apt install ros-$ROS_DISTRO-automatika-embodied-agents`

Alternatively, grab your favorite deb package from the [release page](https://github.com/automatika-robotics/embodied-agents/releases) and install it as follows:

`sudo dpkg -i ros-$ROS_DISTRO-automatica-embodied-agents_$version$DISTRO_$ARCHITECTURE.deb`

If the attrs version from your package manager is < 23.2, install it using pip as follows:

`pip install 'attrs>=23.2.0'`

### Install _EmbodiedAgents_ from source

#### Get Dependencies

Install python dependencies

```shell
pip install numpy opencv-python-headless 'attrs>=23.2.0' jinja2 httpx setproctitle msgpack msgpack-numpy platformdirs tqdm
```

Download Sugarcoat🍬

```shell
git clone https://github.com/automatika-robotics/sugarcoat
```

#### Install _EmbodiedAgents_

```shell
git clone https://github.com/automatika-robotics/embodied-agents.git
cd ..
colcon build
source install/setup.bash
python your_script.py
```

## Quick Start 🚀

Unlike other ROS package, _EmbodiedAgents_ provides a pure pythonic way of describing the node graph using [Sugarcoat🍬](https://www.github.com/automatika-robotics/sugarcoat). Copy the following code in a python script and run it.

```python
from agents.clients.ollama import OllamaClient
from agents.components import MLLM
from agents.models import OllamaModel
from agents.ros import Topic, Launcher

# Define input and output topics (pay attention to msg_type)
text0 = Topic(name="text0", msg_type="String")
image0 = Topic(name="image_raw", msg_type="Image")
text1 = Topic(name="text1", msg_type="String")

# Define a model client (working with Ollama in this case)
llava = OllamaModel(name="llava", checkpoint="llava:latest")
llava_client = OllamaClient(llava)

# Define an MLLM component (A component represents a node with a particular functionality)
mllm = MLLM(
    inputs=[text0, image0],
    outputs=[text1],
    model_client=llava_client,
    trigger=[text0],
    component_name="vqa"
)
# Additional prompt settings
mllm.set_topic_prompt(text0, template="""You are an amazing and funny robot.
    Answer the following about this image: {{ text0 }}"""
)
# Launch the component
launcher = Launcher()
launcher.add_pkg(components=[mllm])
launcher.bringup()
```

And just like that we have an agent that can answer questions like **'What do you see?'**. To interact with this agent, _EmbodiedAgents_ includes a tiny web client. Checkout the [Quick Start Guide](https://automatika-robotics.github.io/embodied-agents/quickstart.html) to learn more about how components and models work together.

## Complex Physical Agents

The quickstart example above is just an amuse-bouche of what is possible with _EmbodiedAgents_. In _EmbodiedAgents_ we can create arbitrarily sophisticated component graphs. And furthermore our system can be configured to even change or reconfigure itself based on events internal or external to the system. Check out the code for the following agent [here](https://automatika-robotics.github.io/embodied-agents/examples/complete.html).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/complete_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/_static/complete_light.png">
  <img alt="Elaborate Agent" src="docs/_static/complete_dark.png">
</picture>

## Copyright

The code in this distribution is Copyright (c) 2024 [Automatika Robotics](https://automatikarobotics.com/) unless explicitly indicated otherwise.

_EmbodiedAgents_ is made available under the MIT license. Details can be found in the [LICENSE](LICENSE) file.

## Contributions

_EmbodiedAgents_ has been developed in collaboration between [Automatika Robotics](https://automatikarobotics.com/) and [Inria](https://inria.fr/). Contributions from the community are most welcome.

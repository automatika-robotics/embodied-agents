<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/EMBODIED_AGENTS_DARK.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/_static/EMBODIED_AGENTS_LIGHT.png">
  <img alt="EmbodiedAgents Logo" src="docs/_static/EMBODIED_AGENTS_DARK.png" width="600">
</picture>

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble%2B-green)](https://docs.ros.org/en/humble/index.html)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/B9ZU6qjzND)
[![简体中文](https://img.shields.io/badge/简体中文-gray.svg)](docs/README.zh.md)
[![日本語](https://img.shields.io/badge/日本語-gray.svg)](docs/README.ja.md)

**The production-grade framework for deploying Physical AI**

[**Installation**](#installation) | [**Quick Start**](#quick-start) | [**Documentation**](https://automatika-robotics.github.io/embodied-agents/) | [**Discord**](https://discord.gg/B9ZU6qjzND)

</div>

---

## Overview

**EmbodiedAgents** enables you to create interactive, **physical agents** that do not just chat, but **understand**, **move**, **manipulate**, and **adapt** to their environment.

Unlike standard chatbots, this framework provides an orchestration layer for **Adaptive Intelligence** designed specifically for autonomous systems in dynamic environments.

### Core Features

- **Production Ready**
  Designed for real-world deployment. Provides a robust orchestration layer that makes deploying Physical AI simple, scalable, and reliable.

- **Self-Referential Logic**
  Create agents that are self-aware. Agents can start, stop, or reconfigure their components based on internal or external events. Trivially switch planners based on location, or toggle between cloud and local ML (See: [Gödel machines](https://en.wikipedia.org/wiki/G%C3%B6del_machine)).

- **Spatio-Temporal Memory**
  Utilize embodiment primitives like hierarchical spatio-temporal memory and semantic routing. Build arbitrarily complex graphs for agentic information flow without the overhead of standard Generative AI web frameworks.

- **Pure Python, Native ROS2**
  Define complex asynchronous graphs in standard Python without touching XML launch files. Under the hood, it is pure ROS2—fully compatible with the entire ecosystem of hardware drivers, simulation tools, and visualization suites.

---

## Quick Start

_EmbodiedAgents_ provides a pythonic way to describe node graphs using [Sugarcoat](https://www.github.com/automatika-robotics/sugarcoat).

Copy the following recipe into a python script (e.g., `agent.py`) to create a VLM-powered agent that can answer questions like "What do you see?".

```python
from agents.clients.ollama import OllamaClient
from agents.components import VLM
from agents.models import OllamaModel
from agents.ros import Topic, Launcher

# 1. Define input and output topics
text0 = Topic(name="text0", msg_type="String")
image0 = Topic(name="image_raw", msg_type="Image")
text1 = Topic(name="text1", msg_type="String")

# 2. Define a model client (e.g., Qwen via Ollama)
qwen_vl = OllamaModel(name="qwen_vl", checkpoint="qwen2.5vl:latest")
qwen_client = OllamaClient(qwen_vl)

# 3. Define a VLM component
# A component represents a node with specific functionality
vlm = VLM(
    inputs=[text0, image0],
    outputs=[text1],
    model_client=qwen_client,
    trigger=text0,
    component_name="vqa"
)

# 4. Set the prompt template
vlm.set_topic_prompt(text0, template="""You are an amazing and funny robot.
    Answer the following about this image: {{ text0 }}"""
)

# 5. Launch the agent
launcher = Launcher()
launcher.add_pkg(components=[vlm])
launcher.bringup()
```

> **Note:** Check out the [Quick Start Guide](https://automatika-robotics.github.io/embodied-agents/quickstart.html) or dive into [Example Recipes](https://automatika-robotics.github.io/embodied-agents/examples/foundation/index.html) for more details.

---

## Complex Component Graphs

The quickstart example above is just an amuse-bouche of what is possible with _EmbodiedAgents_. We can create arbitrarily sophisticated component graphs and configure the system to change or reconfigure itself based on events, both internal or external to the system. Check out the code for the following agent [here](https://automatika-robotics.github.io/embodied-agents/examples/foundation/complete.html).

<div align="center">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="docs/_static/complete_dark.png">
<source media="(prefers-color-scheme: light)" srcset="docs/_static/complete_light.png">
<img alt="Elaborate Agent" src="docs/_static/complete_dark.png" width="80%">
</picture>
</div>

## Dynamic Web UI

Every agent recipe generates a **fully dynamic Web UI** automatically. Built with FastHTML, it provides instant control and visualization without writing a single line of frontend code.

<div align="center">
<picture>
<img alt="EmbodiedAgents UI Example GIF" src="docs/_static/ui_agents.gif" width="70%">
</picture>
</div>

---

## Installation

To get **EmbodiedAgents** up and running, follow these steps in order.

### 1. Pre-Requisite: Model Serving Platform

*EmbodiedAgents* is agnostic to model serving platforms. You must have one of the following installed:

* **[Ollama](https://ollama.com)** (Recommended for local inference)
* **[RoboML](https://github.com/automatika-robotics/robo-ml)**
* **OpenAI Compatible APIs** (e.g., [llama.cpp](https://github.com/ggml-org/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang))
* **[LeRobot](https://github.com/huggingface/lerobot)** (For VLA models)

> **Note:** You can skip this if using a cloud service like HuggingFace inference endpoints.

---

### 2. Standard Installation (Ubuntu/Debian)

For ROS versions **Humble** or newer.

**Option A: Using `apt` (Recommended)**

```bash
sudo apt install ros-$ROS_DISTRO-automatika-embodied-agents
```

**Option B: Using `.deb` package**

1. Download from the [Release Page](https://github.com/automatika-robotics/embodied-agents/releases).
2. Install the package:

```bash
sudo dpkg -i ros-$ROS_DISTRO-automatica-embodied-agents_$version$DISTRO_$ARCHITECTURE.deb
```

**Requirement:** Ensure your `attrs` version is up to date:

```bash
pip install 'attrs>=23.2.0'
```

---

### 3. Advanced Installation (From Source)

Use this method if you want to get the nightly version or plan to contribute to the project.

**Step 1: Install Dependencies**

```bash
pip install numpy opencv-python-headless 'attrs>=23.2.0' jinja2 \
            httpx setproctitle msgpack msgpack-numpy \
            platformdirs tqdm websockets
```

**Step 2: Clone and Build**

```bash
# Clone Sugarcoat dependency
git clone https://github.com/automatika-robotics/sugarcoat

# Clone and build EmbodiedAgents
git clone https://github.com/automatika-robotics/embodied-agents.git
cd ..
colcon build
source install/setup.bash
```

---

## Resources

- [Installation Instructions](https://automatika-robotics.github.io/embodied-agents/installation.html)
- [Quickstart Guide](https://automatika-robotics.github.io/embodied-agents/quickstart.html)
- [Basic Concepts](https://automatika-robotics.github.io/embodied-agents/basics/components.html)
- [Example Recipes](https://automatika-robotics.github.io/embodied-agents/examples/foundation/index.html)

## Copyright & Contributions

**EmbodiedAgents** is a collaboration between [Automatika Robotics](https://automatikarobotics.com/) and [Inria](https://inria.fr/).

The code is available under the **MIT License**. See [LICENSE](https://www.google.com/search?q=LICENSE) for details.
Copyright (c) 2024 Automatika Robotics unless explicitly indicated otherwise.

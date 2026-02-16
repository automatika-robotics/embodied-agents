# Installation

## Prerequisites

:::{admonition} *ROS2* Required
:class: note
EmbodiedAgents supports all *ROS2* distributions from **Humble** up to **Rolling**.
Please ensure you have a working [ROS2 installation](https://docs.ros.org/) before proceeding.
:::

<span class="sd-text-primary" style="font-weight: bold; font-size: 1.1em;">Install a Model Inference Platform</span>

*EmbodiedAgents* is agnostic to model serving platforms. You must have one of the following installed:

* **[Ollama](https://ollama.com)** (Recommended for local inference)
* **[RoboML](https://github.com/automatika-robotics/robo-ml)**
* **OpenAI Compatible APIs** (e.g., [llama.cpp](https://github.com/ggml-org/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang))
* **[LeRobot](https://github.com/huggingface/lerobot)** (For VLA models)

> **Note:** You can skip this if using a cloud service like HuggingFace inference endpoints.

```{tip}
For utilizing larger models, it is recommended that model serving platforms are not installed directly on the robot (or the edge device) but on a GPU powered machine on the local network (or use one of the cloud providers).
```


## Install _EmbodiedAgents_

::::{tab-set}

:::{tab-item} {material-regular}`widgets;1.5em;sd-text-primary` Binary
:sync: binary

**Best for users who want to get started quickly**

For ROS versions >= _humble_, you can install _EmbodiedAgents_ with your package manager. For example on Ubuntu:

```bash
sudo apt install ros-$ROS_DISTRO-automatika-embodied-agents
```

Alternatively, grab your favorite deb package from the [release page](https://github.com/automatika-robotics/embodied-agents/releases) and install it as follows:

```bash
sudo dpkg -i ros-$ROS_DISTRO-automatica-embodied-agents_$version$DISTRO_$ARCHITECTURE.deb
```

If the attrs version from your package manager is < 23.2, install it using pip as follows:

`pip install 'attrs>=23.2.0'`

:::

:::{tab-item} {material-regular}`build;1.5em;sd-text-primary` Source
:sync: source

**Best for contributors or users needing the absolute latest features**

1. Create your ROS workspace.

```shell
mkdir -p agents_ws/src
cd agents_ws/src
```

2. Install python dependencies


```shell
pip install numpy opencv-python-headless 'attrs>=23.2.0' jinja2 httpx setproctitle msgpack msgpack-numpy platformdirs tqdm pyyaml toml websockets
```

3. Install Sugarcoat🍬

```shell
git clone https://github.com/automatika-robotics/sugarcoat
```

4. Install _EmbodiedAgents_

```shell
# Clone repository
git clone https://github.com/automatika-robotics/embodied-agents.git
cd ..

# Build and source
colcon build
source install/setup.bash

# Run your recipe!
python your_script.py
```

:::

::::

<!--
## Install _EmbodiedAgents_ (Ubuntu)

For ROS versions >= _humble_, you can install _EmbodiedAgents_ with your package manager. For example on Ubuntu:

`sudo apt install ros-$ROS_DISTRO-automatika-embodied-agents`

Alternatively, grab your favorite deb package from the [release page](https://github.com/automatika-robotics/embodied-agents/releases) and install it as follows:

`sudo dpkg -i ros-$ROS_DISTRO-automatica-embodied-agents_$version$DISTRO_$ARCHITECTURE.deb`

**Requirement:** If the attrs version from your package manager is < 23.2, install it using pip as follows:

`pip install 'attrs>=23.2.0'`

## Install _EmbodiedAgents_ from source

Create your ROS workspace.

```shell
mkdir -p agents_ws/src
cd agents_ws/src
```

### Get Dependencies

Install python dependencies

```shell
pip install numpy opencv-python-headless 'attrs>=23.2.0' jinja2 httpx setproctitle msgpack msgpack-numpy platformdirs tqdm pyyaml toml websockets
```

Download Sugarcoat🍬.

```shell
git clone https://github.com/automatika-robotics/sugarcoat
```

### Install _EmbodiedAgents_

```shell
git clone https://github.com/automatika-robotics/embodied-agents.git
cd ..
colcon build
source install/setup.bash
python your_script.py
``` -->

# Installation 🛠️

## Pre-Requisits

### Install ROS

_EmbodiedAgents_ is built to be used with ROS2. All ROS distributions starting from _Iron_ are supported. Install ROS2 by following the instructions on the [official site](https://docs.ros.org/en/iron/Installation.html).

### Install a model serving platform

The core of _EmbodiedAgents_ is agnostic to model serving platforms. It currently supports [Ollama](https://ollama.com), [RoboML](https://github.com/automatika-robotics/robo-ml) and any platform or cloud provider with an OpenAI compatible API (e.g. [vLLM](https://github.com/vllm-project/vllm), [lmdeploy](https://github.com/InternLM/lmdeploy) etc.). Please install either of these by following the instructions provided by respective projects. Support for new platforms is being continuously added. If you would like to support a particular platform, please open an issue/PR.


```{tip}
For utilizing larger models, it is recommended that model serving platforms are not installed directly on the robot (or the edge device) but on a GPU powered machine on the local network (or one of the cloud providers).
```

## Install _EmbodiedAgents_ (Ubuntu)

For ROS versions >= _humble_, you can install _EmbodiedAgents_ with your package manager. For example on Ubuntu:

`sudo apt install ros-$ROS_DISTRO-automatika-embodied-agents`

Alternatively, grab your favorite deb package from the [release page](https://github.com/automatika-robotics/embodied-agents/releases) and install it as follows:

`sudo dpkg -i ros-$ROS_DISTRO-automatica-embodied-agents_$version$DISTRO_$ARCHITECTURE.deb`

If the attrs version from your package manager is < 23.2, install it using pip as follows:

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
pip install numpy opencv-python-headless 'attrs>=23.2.0' jinja2 httpx setproctitle msgpack msgpack-numpy platformdirs tqdm pyyaml toml
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
```

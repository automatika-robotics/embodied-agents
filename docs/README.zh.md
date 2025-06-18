<picture>
  <source media="(prefers-color-scheme: dark)" srcset="_static/EMBODIED_AGENTS_DARK.png">
  <source media="(prefers-color-scheme: light)" srcset="_static/EMBODIED_AGENTS_LIGHT.png">
  <img alt="EmbodiedAgents Logo." src="_static/EMBODIED_AGENTS_DARK.png">
</picture>
<br/>

> 🌐 [English Version](../README.md) | 🇯🇵 [日本語版](README.ja.md)

EmbodiedAgents 是一个功能齐全的框架，完全使用 ROS2 编写，用于创建能够理解、记忆并基于环境上下文信息采取行动的交互式物理智能体。

- **可投入生产的物理智能体**：设计用于在现实世界动态环境中运行的自主机器人系统。EmbodiedAgents 使构建基于物理 AI 的系统变得简单。
- **直观的 API**：使用简单的 Python 风格 API，在机器人上调用本地或云端的机器学习模型（特别是 **多模态大语言模型（Multimodal LLMs）** 和其他 **transformer 架构**）。
- **语义记忆**：集成向量数据库、语义路由和其他支持组件，快速构建复杂的代理信息流图。无需在机器人上部署臃肿的 "GenAI" 框架。
- **基于 ROS2 构建**：以 ROS2 作为分布式通信的核心。理论上所有提供 ROS2 包的设备都可用于将数据发送给 ML 模型，已实现常用数据类型的回调，并具备无限扩展性。

查看 [安装说明](https://automatika-robotics.github.io/embodied-agents/installation.html) 🛠️
立即开始 [快速上手](https://automatika-robotics.github.io/embodied-agents/quickstart.html) 🚀
了解 [基本概念](https://automatika-robotics.github.io/embodied-agents/basics.html) 📚
参考 [示例代码](https://automatika-robotics.github.io/embodied-agents/examples/index.html) ✨

## 安装 🛠️

### 先决条件

#### 安装 ROS

EmbodiedAgents 构建于 ROS2 之上。支持从 _Humble_ 开始的所有 ROS 版本。请参阅 [官方安装指南](https://docs.ros.org/en/iron/Installation.html)。

#### 安装模型服务平台

EmbodiedAgents 与模型服务平台无关。当前支持 [Ollama](https://ollama.com) 和 [RoboML](https://github.com/automatika-robotics/robo-ml)。请按照各自项目提供的说明进行安装。我们会不断添加对新平台的支持。如有建议，请提交 issue 或 PR。

### 安装 EmbodiedAgents（Ubuntu）

使用 pip 安装 Python 依赖：

```bash
pip install 'attrs>=23.2.0' numpy-quaternion
```

ROS 版本 >= 'humble' 时，可通过包管理器安装 EmbodiedAgents，例如：

```bash
sudo apt install ros-$ROS_DISTRO-automatika-embodied-agents
```

### 从源码安装 EmbodiedAgents

#### 安装依赖项

```bash
pip install numpy opencv-python-headless 'attrs>=23.2.0' jinja2 httpx setproctitle msgpack msgpack-numpy numpy-quaternion platformdirs tqdm
```

克隆 Sugarcoat：

```bash
git clone https://github.com/automatika-robotics/sugarcoat
```

#### 安装 EmbodiedAgents

```bash
git clone https://github.com/automatika-robotics/embodied-agents.git
cd ..
colcon build
source install/setup.bash
python your_script.py
```

## 快速开始 🚀

与其他 ROS 包不同，EmbodiedAgents 使用纯 Python 的方式，通过 [Sugarcoat](https://www.github.com/automatika-robotics/sugarcoat) 描述节点图。复制以下代码并运行：

```python
from agents.clients.ollama import OllamaClient
from agents.components import MLLM
from agents.models import OllamaModel
from agents.ros import Topic, Launcher

text0 = Topic(name="text0", msg_type="String")
image0 = Topic(name="image_raw", msg_type="Image")
text1 = Topic(name="text1", msg_type="String")

llava = OllamaModel(name="llava", checkpoint="llava:latest")
llava_client = OllamaClient(llava)

mllm = MLLM(
    inputs=[text0, image0],
    outputs=[text1],
    model_client=llava_client,
    trigger=[text0],
    component_name="vqa"
)
mllm.set_topic_prompt(text0, template="""You are an amazing and funny robot.
    Answer the following about this image: {{ text0 }}"""
)
launcher = Launcher()
launcher.add_pkg(components=[mllm])
launcher.bringup()
```

这样就创建了一个可以回答如 **“你看到了什么？”** 的智能体。EmbodiedAgents 还包括一个轻量级的网页客户端。查看 [快速上手指南](https://automatika-robotics.github.io/embodied-agents/quickstart.html) 了解组件与模型如何协作。

## 复杂物理智能体

上述示例只是 EmbodiedAgents 的冰山一角。你可以使用 EmbodiedAgents 构建任意复杂的组件图，并使系统根据内部或外部事件动态重构。查看该复杂代理的代码：[点击这里](https://automatika-robotics.github.io/embodied-agents/examples/complete.html)。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="_static/complete_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="_static/complete_light.png">
  <img alt="Elaborate Agent" src="_static/complete_dark.png">
</picture>

## 版权信息

除非另有明确说明，本发行版中的代码版权归 2024 Automatika Robotics 所有。

EmbodiedAgents 采用 MIT 许可证发布。详细信息请参见 [LICENSE](LICENSE) 文件。

## 贡献

EmbodiedAgents 由 [Automatika Robotics](https://automatikarobotics.com/) 与 [Inria](https://inria.fr/) 共同开发。欢迎社区贡献。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="_static/EMBODIED_AGENTS_DARK.png">
  <source media="(prefers-color-scheme: light)" srcset="_static/EMBODIED_AGENTS_LIGHT.png">
  <img alt="EmbodiedAgents Logo." src="_static/EMBODIED_AGENTS_DARK.png">
</picture>
<br/>

> 🌐 [English Version](../README.md) | 🇯🇵 [日本語版](README.ja.md)

_EmbodiedAgents_ 是一个构建于 **ROS2** 之上的生产级框架，旨在弥合生成式 AI 与实体机器人之间的鸿沟。它使您能够创建交互式的实体智能体，这些智能体不仅能聊天，还能真正**理解**、**移动**、**操作**并**适应**其环境。

- **生产就绪的实体智能体 (Production Ready Physical Agents)：** 专为在现实世界动态环境中运行的自主机器人系统而设计。_EmbodiedAgents_ 简化了利用实体人工智能 (Physical AI) 构建系统的过程。它为**自适应智能 (Adaptive Intelligence)** 提供了一个编排层。
- **自指与事件驱动 (Self-referential and Event Driven)：** 使用 _EmbodiedAgents_ 创建的智能体可以根据内部和外部事件启动、停止或重新配置其自身的组件。例如，智能体可以根据其在地图上的位置或视觉模型的输入来更改用于规划的机器学习模型。_EmbodiedAgents_ 让创建自指的 [哥德尔机 (Gödel machines)](https://en.wikipedia.org/wiki/G%C3%B6del_machine) 变得简单。
- **语义记忆 (Semantic Memory)：** 集成了向量数据库、语义路由和其他支持组件，可以快速构建任意复杂的图结构以实现智能体的信息流。无需在您的机器人上使用臃肿的“GenAI”框架。
- **纯 Python，原生 ROS2：** 使用标准 Python 定义复杂的异步图，无需接触 XML 启动文件。然而，在其底层，它是纯正的 ROS2，兼容整个硬件驱动程序、仿真工具和可视化套件的生态系统。

加入我们的 [Discord](https://discord.gg/B9ZU6qjzND) 👾

查看 [安装说明](https://automatika-robotics.github.io/embodied-agents/installation.html) 🛠️

通过 [快速入门指南](https://automatika-robotics.github.io/embodied-agents/quickstart.html) 开始使用 🚀

熟悉 [基本概念](https://automatika-robotics.github.io/embodied-agents/basics/components.html) 📚

直接深入 [示例食谱](https://automatika-robotics.github.io/embodied-agents/examples/foundation/index.html) ✨

## 安装 🛠️

### 安装模型部署平台

*EmbodiedAgents* 的核心不依赖于特定的模型服务平台。它支持 [Ollama](https://ollama.com)、[RoboML](https://github.com/automatika-robotics/robo-ml) 以及所有具有 OpenAI 兼容 API 的平台或云服务提供商（例如 [vLLM](https://github.com/vllm-project/vllm)、[lmdeploy](https://github.com/InternLM/lmdeploy) 等）。对于 VLA 模型，*EmbodiedAgents* 支持部署在 [LeRobot](https://github.com/huggingface/lerobot) 的异步推理服务器 (Async Inference server) 上的策略 (policies)。请遵循各项目的官方说明进行安装。我们正在持续增加对新平台的支持。如果您希望支持特定的平台，请提交 issue 或 PR。

### 安装 _EmbodiedAgents_（Ubuntu）

对于 ROS 版本 >= _humble_，你可以通过软件包管理器安装 _EmbodiedAgents_。例如在 Ubuntu 上执行以下命令：

`sudo apt install ros-$ROS_DISTRO-automatika-embodied-agents`

或者，你也可以从 [发布页面](https://github.com/automatika-robotics/embodied-agents/releases) 下载你喜欢的 `.deb` 安装包，并通过以下方式进行安装：

`sudo dpkg -i ros-$ROS_DISTRO-automatica-embodied-agents_$version$DISTRO_$ARCHITECTURE.deb`

如果你从软件包管理器安装的 attrs 版本低于 23.2，请使用 pip 安装如下版本：

`pip install 'attrs>=23.2.0'`

### 从源码安装 _EmbodiedAgents_

#### 安装依赖项

```bash
pip install numpy opencv-python-headless 'attrs>=23.2.0' jinja2 httpx setproctitle msgpack msgpack-numpy platformdirs tqdm websockets
```

克隆 Sugarcoat🍬：

```bash
git clone https://github.com/automatika-robotics/sugarcoat
```

#### 安装 _EmbodiedAgents_

```bash
git clone https://github.com/automatika-robotics/embodied-agents.git
cd ..
colcon build
source install/setup.bash
python your_script.py
```

## 快速开始 🚀

与其他 ROS 包不同，_EmbodiedAgents_ 使用纯 Python 的方式，通过 [Sugarcoat🍬](https://www.github.com/automatika-robotics/sugarcoat) 描述节点图。复制以下代码并运行：

```python
from agents.clients.ollama import OllamaClient
from agents.components import VLM
from agents.models import OllamaModel
from agents.ros import Topic, Launcher

# Define input and output topics (pay attention to msg_type)
text0 = Topic(name="text0", msg_type="String")
image0 = Topic(name="image_raw", msg_type="Image")
text1 = Topic(name="text1", msg_type="String")

# Define a model client (working with Ollama in this case)
# OllamaModel is a generic wrapper for all Ollama models
qwen_vl = OllamaModel(name="qwen_vl", checkpoint="qwen2.5vl:latest")
qwen_client = OllamaClient(qwen_vl)

# Define a VLM component (A component represents a node with a particular functionality)
vlm = VLM(
    inputs=[text0, image0],
    outputs=[text1],
    model_client=qwen_client,
    trigger=text0,
    component_name="vqa"
)
# Additional prompt settings
vlm.set_topic_prompt(text0, template="""You are an amazing and funny robot.
    Answer the following about this image: {{ text0 }}"""
)

# Launch the component
launcher = Launcher()
launcher.add_pkg(components=[vlm])
launcher.bringup()
```

这样就创建了一个可以回答如 **“你看到了什么？”** 的智能体。_EmbodiedAgents_ 还包括一个轻量级的网页客户端。查看 [快速上手指南](https://automatika-robotics.github.io/embodied-agents/quickstart.html) 了解组件与模型如何协作。

## 复杂物理智能体

上述示例只是 _EmbodiedAgents_ 的冰山一角。你可以使用 EmbodiedAgents 构建任意复杂的组件图，并使系统根据内部或外部事件动态重构。查看该复杂代理的代码：[点击这里](https://automatika-robotics.github.io/embodied-agents/examples/foundation/complete.html)。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="_static/complete_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="_static/complete_light.png">
  <img alt="Elaborate Agent" src="_static/complete_dark.png">
</picture>

## EmbodiedAgent 配方的动态网页界面

借助底层 [**Sugarcoat**](https://github.com/automatika-robotics/sugarcoat) 框架的强大功能，**_EmbodiedAgents_** 为每个配方提供了一个**完全动态、自动生成的网页界面（Web UI）**。
该功能基于 **FastHTML** 构建，彻底消除了手动开发图形界面的需求，并能即时提供一个用于控制和可视化的响应式界面。

该界面会自动生成以下内容：

- 配方中所使用的所有组件的设置界面
- 组件输入/输出的实时数据可视化与控制
- 针对所有支持的消息类型的基于 WebSocket 的数据流传输

### 示例：VLM Agent 界面

系统会为 VLM 问答（Q&A）代理（类似快速入门示例）自动生成完整的界面，提供简洁的设置控制，并显示实时的文本输入/输出。

<p align="center">
<picture align="center">
  <img alt="EmbodiedAgents UI Example GIF" src="docs/_static/ui_agents.gif" width="60%">
</picture>
</p>

## 版权信息

除非另有明确说明，本发行版中的代码版权归 2024 [Automatika Robotics](https://automatikarobotics.com/) 所有。

_EmbodiedAgents_ 采用 MIT 许可证发布。详细信息请参见 [LICENSE](LICENSE) 文件。

## 贡献

_EmbodiedAgents_ 由 [Automatika Robotics](https://automatikarobotics.com/) 与 [Inria](https://inria.fr/) 共同开发。欢迎社区贡献。

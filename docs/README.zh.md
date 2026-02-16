<div align="center">

<picture>
<source media="(prefers-color-scheme: dark)" srcset="_static/EMBODIED_AGENTS_DARK.png">
<source media="(prefers-color-scheme: light)" srcset="_static/EMBODIED_AGENTS_LIGHT.png">
<img alt="EmbodiedAgents Logo" src="_static/EMBODIED_AGENTS_DARK.png" width="600">
</picture>

**用于部署具身智能 (Physical AI) 的生产级框架**

**[安装](#安装)** | **[快速开始](#快速开始)** | **[文档](https://agents.automatikarobotics.com/)** | **[Discord](https://discord.gg/B9ZU6qjzND)**

</div>

---

## 概览

**EmbodiedAgents** 让您可以创建交互式的**具身智能体 (Physical Agents)**，它们不仅会聊天，还能**理解**、**移动**、**操作**并**适应**环境。

与标准的聊天机器人不同，本框架提供了一个专为动态环境中的自主系统设计的**自适应智能 (Adaptive Intelligence)** 编排层。

### 核心特性

- **生产就绪 (Production Ready)**
  专为现实世界的部署而设计。提供了一个强大的编排层，使具身智能的部署变得简单、可扩展且可靠。
- **自指逻辑 (Self-Referential Logic)**
  创建具有自我意识的智能体。智能体可以根据内部或外部事件启动、停止或重新配置其组件。例如，根据位置轻松切换规划器，或在云端和本地机器学习模型之间切换（参见：[哥德尔机](https://en.wikipedia.org/wiki/G%C3%B6del_machine)）。
- **时空记忆 (Spatio-Temporal Memory)**
  利用具身原语，如分层时空记忆和语义路由。构建任意复杂的智能体信息流图。无需在机器人上使用臃肿的通用 "GenAI" 框架。
- **纯 Python，原生 ROS2**
  使用标准 Python 定义复杂的异步图，无需接触 XML 启动文件。在底层，它是纯 ROS2 的——完全兼容硬件驱动、仿真工具和可视化套件的整个生态系统。

---

## 快速开始

_EmbodiedAgents_ 提供了一种 Python 风格的方式，使用 [Sugarcoat](https://www.github.com/automatika-robotics/sugarcoat) 来描述节点图。

将以下配方复制到 Python 脚本中（例如 `agent.py`），即可创建一个由 VLM（视觉语言模型）驱动的智能体，它可以回答诸如“你看到了什么？”之类的问题。

```python
from agents.clients.ollama import OllamaClient
from agents.components import VLM
from agents.models import OllamaModel
from agents.ros import Topic, Launcher

# 1. 定义输入和输出话题 (Topics)
text0 = Topic(name="text0", msg_type="String")
image0 = Topic(name="image_raw", msg_type="Image")
text1 = Topic(name="text1", msg_type="String")

# 2. 定义模型客户端 (例如：通过 Ollama 调用 Qwen)
qwen_vl = OllamaModel(name="qwen_vl", checkpoint="qwen2.5vl:latest")
qwen_client = OllamaClient(qwen_vl)

# 3. 定义 VLM 组件
# 组件代表具有特定功能的节点
vlm = VLM(
    inputs=[text0, image0],
    outputs=[text1],
    model_client=qwen_client,
    trigger=text0,
    component_name="vqa"
)

# 4. 设置提示词模板
vlm.set_topic_prompt(text0, template="""你是一个很棒且有趣的机器人。
    请回答关于这张图片的问题：{{ text0 }}"""
)

# 5. 启动智能体
launcher = Launcher()
launcher.add_pkg(components=[vlm])
launcher.bringup()
```

> **注意：** 查看 [快速开始指南](https://automatika-robotics.github.io/embodied-agents/quickstart.html) 或深入研究 [示例配方](https://automatika-robotics.github.io/embodied-agents/examples/foundation/index.html) 以了解更多详情。

---

## 复杂的组件图

上面的快速入门示例仅仅是 _EmbodiedAgents_ 功能的冰山一角。我们可以创建任意复杂的组件图，并配置系统根据内部或外部事件进行更改或重新配置。点击 [此处](https://automatika-robotics.github.io/embodied-agents/examples/foundation/complete.html) 查看以下智能体的代码。

<div align="center">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="_static/complete_dark.png">
<source media="(prefers-color-scheme: light)" srcset="_static/complete_light.png">
<img alt="Elaborate Agent" src="_static/complete_dark.png" width="80%">
</picture>
</div>

## 动态 Web UI

每个智能体配方都会自动生成一个**全动态 Web UI**。它使用 FastHTML 构建，无需编写一行前端代码即可提供即时控制和可视化。

<div align="center">
<picture>
<img alt="EmbodiedAgents UI Example GIF" src="_static/ui_agents.gif" width="70%">
</picture>
</div>

---

## 安装

要运行 **EmbodiedAgents**，请按顺序执行以下步骤。

### 1. 先决条件：模型服务平台

_EmbodiedAgents_ 与模型服务平台无关。您必须安装以下其中之一：

- **[Ollama](https://ollama.com)** (推荐用于本地推理)
- **[RoboML](https://github.com/automatika-robotics/robo-ml)**
- **兼容 OpenAI API 的推理服务器** (例如：[llama.cpp](https://github.com/ggml-org/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang))
- **[LeRobot](https://github.com/huggingface/lerobot)** (用于 VLA 模型)

> **注意：** 如果使用像 HuggingFace Inference Endpoints 这样的云服务，可以跳过此步骤。

---

### 2. 标准安装 (Ubuntu/Debian)

适用于 **Humble** 或更高版本的 ROS。

**选项 A: 使用 `apt` (推荐)**

```bash
sudo apt install ros-$ROS_DISTRO-automatika-embodied-agents
```

**选项 B: 使用 `.deb` 包**

1. 从 [发布页面](https://github.com/automatika-robotics/embodied-agents/releases) 下载。
2. 安装软件包：

```bash
sudo dpkg -i ros-$ROS_DISTRO-automatica-embodied-agents_$version$DISTRO_$ARCHITECTURE.deb
```

**要求：** 确保您的 `attrs` 版本是最新的：

```bash
pip install 'attrs>=23.2.0'
```

---

### 3. 高级安装 (源码安装)

如果您想使用每夜构建版 (nightly version) 或计划为项目做贡献，请使用此方法。

**步骤 1: 安装依赖**

```bash
pip install numpy opencv-python-headless 'attrs>=23.2.0' jinja2 \
            httpx setproctitle msgpack msgpack-numpy \
            platformdirs tqdm websockets
```

**步骤 2: 克隆并构建**

```bash
# 克隆 Sugarcoat 依赖
git clone https://github.com/automatika-robotics/sugarcoat

# 克隆并构建 EmbodiedAgents
git clone https://github.com/automatika-robotics/embodied-agents.git
cd ..
colcon build
source install/setup.bash
```

---

## 资源

- [安装说明](https://automatika-robotics.github.io/embodied-agents/installation.html)
- [快速开始指南](https://automatika-robotics.github.io/embodied-agents/quickstart.html)
- [基本概念](https://automatika-robotics.github.io/embodied-agents/basics/components.html)
- [示例配方](https://automatika-robotics.github.io/embodied-agents/examples/foundation/index.html)

## 版权与贡献

**EmbodiedAgents** 是 [Automatika Robotics](https://automatikarobotics.com/) 和 [Inria](https://inria.fr/) 之间的合作项目。

代码在 **MIT 许可** 下提供。详情请参阅 [LICENSE](../LICENSE)。
除非另有明确说明，版权所有 (c) 2024 Automatika Robotics。

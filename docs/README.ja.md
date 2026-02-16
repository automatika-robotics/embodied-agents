<div align="center">

<picture>
<source media="(prefers-color-scheme: dark)" srcset="_static/EMBODIED_AGENTS_DARK.png">
<source media="(prefers-color-scheme: light)" srcset="_static/EMBODIED_AGENTS_LIGHT.png">
<img alt="EmbodiedAgents Logo" src="_static/EMBODIED_AGENTS_DARK.png" width="600">
</picture>

**フィジカルAI (Physical AI) 導入のための実運用向けフレームワーク**

**[インストール]()** | **[クイックスタート]()** | **[ドキュメント](https://automatika-robotics.github.io/embodied-agents/)** | **[Discord](https://discord.gg/B9ZU6qjzND)**

</div>

---

## 概要

**EmbodiedAgents** は、単にチャットをするだけでなく、環境を**理解**し、**移動**し、**操作**し、そして**適応**するインタラクティブな**物理エージェント (Physical Agents)** の作成を可能にします。

標準的なチャットボットとは異なり、本フレームワークは動的な環境における自律システムのために特別に設計された、**適応型インテリジェンス (Adaptive Intelligence)** のためのオーケストレーション層を提供します。

### 主な特徴

- **実運用に対応 (Production Ready)**
  実世界での展開を想定して設計されています。フィジカルAIの展開をシンプルかつスケーラブルで信頼性の高いものにする、堅牢なオーケストレーション層を提供します。
- **自己参照ロジック (Self-Referential Logic)**
  自己認識を持つエージェントを作成できます。エージェントは内部または外部のイベントに基づいて、自身のコンポーネントを開始、停止、または再構成できます。場所に基づいてプランナーを簡単に切り替えたり、クラウドとローカルのMLモデルを切り替えたりすることが可能です（参照：[ゲーデルマシン](https://en.wikipedia.org/wiki/G%C3%B6del_machine)）。
- **時空間メモリ (Spatio-Temporal Memory)**
  階層的な時空間メモリやセマンティックルーティングなどの身体性プリミティブ (embodiment primitives) を活用します。エージェントの情報フローのために、任意に複雑なグラフを構築できます。ロボット上で肥大化した汎用的な「生成AI」フレームワークを使用する必要はありません。
- **純粋なPython、ネイティブROS2**
  XMLのローンチファイルに触れることなく、標準的なPythonで複雑な非同期グラフを定義できます。内部的には純粋なROS2であり、ハードウェアドライバ、シミュレーションツール、可視化スイートなどのエコシステム全体と完全に互換性があります。

---

## クイックスタート

_EmbodiedAgents_ は、[Sugarcoat](https://www.github.com/automatika-robotics/sugarcoat) を使用してノードグラフを記述するための Pythonic な方法を提供します。

以下のレシピをPythonスクリプト（例：`agent.py`）にコピーして、「何が見えますか？」のような質問に答えることができるVLM（視覚言語モデル）搭載エージェントを作成してみましょう。

```python
from agents.clients.ollama import OllamaClient
from agents.components import VLM
from agents.models import OllamaModel
from agents.ros import Topic, Launcher

# 1. 入出力トピックの定義
text0 = Topic(name="text0", msg_type="String")
image0 = Topic(name="image_raw", msg_type="Image")
text1 = Topic(name="text1", msg_type="String")

# 2. モデルクライアントの定義 (例: Ollama経由のQwen)
qwen_vl = OllamaModel(name="qwen_vl", checkpoint="qwen2.5vl:latest")
qwen_client = OllamaClient(qwen_vl)

# 3. VLMコンポーネントの定義
# コンポーネントは特定の機能を持つノードを表します
vlm = VLM(
    inputs=[text0, image0],
    outputs=[text1],
    model_client=qwen_client,
    trigger=text0,
    component_name="vqa"
)

# 4. プロンプトテンプレートの設定
vlm.set_topic_prompt(text0, template="""あなたは素晴らしくて面白いロボットです。
    この画像について次の質問に答えてください: {{ text0 }}"""
)

# 5. エージェントの起動
launcher = Launcher()
launcher.add_pkg(components=[vlm])
launcher.bringup()
```

> **注意:** 詳細は [クイックスタートガイド](https://automatika-robotics.github.io/embodied-agents/quickstart.html) を確認するか、[サンプルレシピ](https://automatika-robotics.github.io/embodied-agents/examples/foundation/index.html) をご覧ください。

---

## 複雑なコンポーネントグラフ

上記のクイックスタートの例は、_EmbodiedAgents_ で可能なことのほんの一部に過ぎません。任意に洗練されたコンポーネントグラフを作成し、システム内部または外部のイベントに基づいて、システム自体を変更または再構成するように設定できます。以下のエージェントのコードは [こちら](https://automatika-robotics.github.io/embodied-agents/examples/foundation/complete.html) で確認できます。

<div align="center">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="_static/complete_dark.png">
<source media="(prefers-color-scheme: light)" srcset="_static/complete_light.png">
<img alt="Elaborate Agent" src="_static/complete_dark.png" width="80%">
</picture>
</div>

## ダイナミック Web UI

すべてのエージェントレシピは、**完全に動的な Web UI** を自動的に生成します。FastHTML で構築されており、フロントエンドのコードを一行も書くことなく、即座に制御と可視化が可能です。

<div align="center">
<picture>
<img alt="EmbodiedAgents UI Example GIF" src="_static/ui_agents.gif" width="70%">
</picture>
</div>

---

## インストール

**EmbodiedAgents** を稼働させるには、以下の手順を順番に実行してください。

### 1. 前提条件: モデルサービングプラットフォーム

_EmbodiedAgents_ はモデルサービングプラットフォームに依存しません。以下のいずれかがインストールされている必要があります：

- **[Ollama](https://ollama.com)** (ローカル推論に推奨)
- **[RoboML](https://github.com/automatika-robotics/robo-ml)**
- **OpenAI API 互換の推論サーバー** (例: [llama.cpp](https://github.com/ggml-org/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang))
- **[LeRobot](https://github.com/huggingface/lerobot)** (VLAモデル用)

> **注意:** HuggingFace Inference Endpoints などのクラウドサービスを使用する場合は、この手順をスキップできます。

---

### 2. 標準インストール (Ubuntu/Debian)

ROS バージョン **Humble** 以降向けです。

**オプション A: `apt` を使用 (推奨)**

```bash
sudo apt install ros-$ROS_DISTRO-automatika-embodied-agents
```

**オプション B: `.deb` パッケージを使用**

1. [リリースページ](https://github.com/automatika-robotics/embodied-agents/releases) からダウンロードします。
2. パッケージをインストールします：

```bash
sudo dpkg -i ros-$ROS_DISTRO-automatica-embodied-agents_$version$DISTRO_$ARCHITECTURE.deb
```

**要件:** `attrs` のバージョンが最新であることを確認してください：

```bash
pip install 'attrs>=23.2.0'
```

---

### 3. 高度なインストール (ソースから)

ナイトリービルドを使用したい場合や、プロジェクトへの貢献を計画している場合は、この方法を使用してください。

**ステップ 1: 依存関係のインストール**

```bash
pip install numpy opencv-python-headless 'attrs>=23.2.0' jinja2 \
            httpx setproctitle msgpack msgpack-numpy \
            platformdirs tqdm websockets
```

**ステップ 2: クローンとビルド**

```bash
# Sugarcoat (依存関係) のクローン
git clone https://github.com/automatika-robotics/sugarcoat

# EmbodiedAgents のクローンとビルド
git clone https://github.com/automatika-robotics/embodied-agents.git
cd ..
colcon build
source install/setup.bash
```

---

## リソース

- [インストール手順](https://automatika-robotics.github.io/embodied-agents/installation.html)
- [クイックスタートガイド](https://automatika-robotics.github.io/embodied-agents/quickstart.html)
- [基本コンセプト](https://automatika-robotics.github.io/embodied-agents/basics/components.html)
- [サンプルレシピ](https://automatika-robotics.github.io/embodied-agents/examples/foundation/index.html)

## 著作権と貢献

**EmbodiedAgents** は [Automatika Robotics](https://automatikarobotics.com/) と [Inria](https://inria.fr/) による共同プロジェクトです。

コードは **MIT ライセンス** の下で提供されています。詳細は [LICENSE](../LICENSE) を参照してください。
特に明記されていない限り、Copyright (c) 2024 Automatika Robotics に帰属します。

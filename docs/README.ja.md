<picture>
  <source media="(prefers-color-scheme: dark)" srcset="_static/EMBODIED_AGENTS_DARK.png">
  <source media="(prefers-color-scheme: light)" srcset="_static/EMBODIED_AGENTS_LIGHT.png">
  <img alt="_EmbodiedAgents_ ロゴ" src="_static/EMBODIED_AGENTS_DARK.png">
</picture>
<br/>

> 🌐 [English Version](../README.md) | 🇨🇳 [简体中文](README.zh.md)

_EmbodiedAgents_ は、生成 AI (Generative AI) と物理ロボット工学のギャップを埋めるために設計された、**ROS2** 上に構築された実用レベルのフレームワークです。単にチャットするだけでなく、環境を**理解**し、**移動**し、**操作**し、そして**適応**できるインタラクティブな身体性を持つエージェント（Physical Agents）を作成することができます。

- **実運用可能な身体性エージェント (Production Ready Physical Agents):** 実世界の動的な環境で動作する自律型ロボットシステムで使用するために設計されています。_EmbodiedAgents_ は、物理 AI (Physical AI) を活用したシステムの構築を簡素化し、**適応的知能 (Adaptive Intelligence)** のためのオーケストレーション層を提供します。
- **自己参照およびイベント駆動 (Self-referential and Event Driven):** _EmbodiedAgents_ で作成されたエージェントは、内部および外部のイベントに基づいて、自身のコンポーネントを開始、停止、または再構成できます。例えば、エージェントは地図上の現在位置や視覚モデルからの入力に基づいて、計画（プランニング）に使用する機械学習モデルを切り替えることが可能です。_EmbodiedAgents_ は、自己参照的な [ゲーデルマシン (Gödel machines)](https://en.wikipedia.org/wiki/G%C3%B6del_machine) のようなエージェントの作成を容易にします。
- **意味記憶 (Semantic Memory):** ベクトルデータベース、セマンティックルーティング、その他のサポートコンポーネントを統合しており、エージェント的な情報の流れを実現するための任意に複雑なグラフを迅速に構築できます。ロボット上で肥大化した「GenAI」フレームワークを使用する必要はありません。
- **Pure Python, ネイティブ ROS2:** XML の launch ファイルに触れることなく、標準的な Python で複雑な非同期グラフを定義できます。その裏側では純粋な ROS2 が動作しており、ハードウェアドライバ、シミュレーションツール、可視化スイートなどの全エコシステムと互換性があります。

[Discord](https://discord.gg/B9ZU6qjzND) に参加する 👾

[インストール手順](https://automatika-robotics.github.io/embodied-agents/installation.html) を確認する 🛠️

[クイックスタートガイド](https://automatika-robotics.github.io/embodied-agents/quickstart.html) で始める 🚀

[基本概念](https://automatika-robotics.github.io/embodied-agents/basics/components.html) に慣れ親しむ 📚

[サンプルレシピ](https://automatika-robotics.github.io/embodied-agents/examples/foundation/index.html) で実践する ✨

## インストール 🛠️

### モデルサービングプラットフォームのインストール

_EmbodiedAgents_ の中核は、モデルサービングプラットフォームに依存しません。[Ollama](https://ollama.com)、[RoboML](https://github.com/automatika-robotics/robo-ml) に加え、OpenAI 互換 API を持つすべてのプラットフォームやクラウドプロバイダー（例：[vLLM](https://github.com/vllm-project/vllm)、[lmdeploy](https://github.com/InternLM/lmdeploy) 等）をサポートしています。VLA モデルについては、_EmbodiedAgents_ は [LeRobot](https://github.com/huggingface/lerobot) の非同期推論サーバー (Async Inference server) 上で提供されるポリシー (policies) に対応しています。各プロジェクトが提供する手順に従って、いずれかをインストールしてください。新たなプラットフォームへのサポートは順次追加されています。特定のプラットフォームへの対応をご希望の場合は、Issue または PR を作成してください。

### _EmbodiedAgents_ のインストール（Ubuntu）

ROS のバージョンが _humble_ 以上であれば、パッケージマネージャーを使って _EmbodiedAgents_ をインストールできます。たとえば Ubuntu では次のように実行します：

```bash
sudo apt install ros-$ROS_DISTRO-automatika-embodied-agents
```

または、[リリースページ](https://github.com/automatika-robotics/embodied-agents/releases) からお好みの `.deb` パッケージをダウンロードして、次のようにインストールすることもできます：

```bash
sudo dpkg -i ros-$ROS_DISTRO-automatica-embodied-agents_$version$DISTRO_$ARCHITECTURE.deb
```

パッケージマネージャーからインストールされる `attrs` のバージョンが 23.2 未満の場合は、次のコマンドで pip を使ってインストールしてください：

```bash
pip install 'attrs>=23.2.0'
```

### ソースからのインストール

#### 依存関係の取得

```bash
pip install numpy opencv-python-headless 'attrs>=23.2.0' jinja2 httpx setproctitle msgpack msgpack-numpy platformdirs tqdm websockets
```

Sugarcoat🍬 をクローン：

```bash
git clone https://github.com/automatika-robotics/sugarcoat
```

#### _EmbodiedAgents_ のクローンとビルド

```bash
git clone https://github.com/automatika-robotics/embodied-agents.git
cd ..
colcon build
source install/setup.bash
python your_script.py
```

## クイックスタート 🚀

_EmbodiedAgents_ は、他の ROS パッケージと異なり、[Sugarcoat🍬](https://www.github.com/automatika-robotics/sugarcoat) を用いてノードグラフを純粋な Python コードで記述できます。以下のスクリプトをコピーして実行してください：

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
llava = OllamaModel(name="llava", checkpoint="llava:latest")
llava_client = OllamaClient(llava)

# Define a VLM component (A component represents a node with a particular functionality)
mllm = VLM(
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

このコードを実行することで、**「何が見える？」** といった質問に答えるエージェントが完成します。_EmbodiedAgents_ には軽量なウェブクライアントも付属しています。[クイックスタートガイド](https://automatika-robotics.github.io/embodied-agents/quickstart.html) で、コンポーネントとモデルの連携方法を学びましょう。

## 複雑な物理エージェント

上記のクイックスタートは、_EmbodiedAgents_ の機能のごく一部にすぎません。EmbodiedAgents では、任意に複雑なコンポーネントグラフを構築できます。さらに、システム内部または外部のイベントに応じて、構成を動的に変更・再構築することも可能です。以下のエージェントのコード例を確認してみてください：[こちらをクリック](https://automatika-robotics.github.io/embodied-agents/examples/foundation/complete.html)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="_static/complete_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="_static/complete_light.png">
  <img alt="高度なエージェント" src="_static/complete_dark.png">
</picture>

## EmbodiedAgentレシピの動的Web UI

基盤となる[**Sugarcoat**](https://github.com/automatika-robotics/sugarcoat)フレームワークの強力な機能を活用し、***EmbodiedAgents***は各レシピに対して**完全に動的で自動生成されるWeb UI**を提供します。
この機能は**FastHTML**によって構築されており、手動でのGUI開発を不要にし、制御や可視化のためのレスポンシブなインターフェースを即座に提供します。

このUIは自動的に以下を生成します：

- レシピ内で使用されるすべてのコンポーネントに対する設定インターフェース
- コンポーネントの入出力に対するリアルタイムデータの可視化と制御
- すべての対応メッセージ型に対するWebSocketベースのデータストリーミング

### 例：VLMエージェントUI

VLM Q&Aエージェント（クイックスタート例と類似）のための完全なインターフェースが自動生成され、設定用のシンプルなコントロールやリアルタイムのテキスト入出力表示を提供します。

<p align="center">
<picture align="center">
  <img alt="EmbodiedAgents UI Example GIF" src="docs/_static/agents_ui.gif" width="60%">
</picture>
</p>

## 著作権情報

本配布物に含まれるコードは、特に明記されていない限り、すべて © 2024 [Automatika Robotics](https://automatikarobotics.com/) に著作権があります。

_EmbodiedAgents_ は MIT ライセンスのもとで公開されています。詳細は [LICENSE](LICENSE) ファイルをご確認ください。

## コントリビューション（貢献）

_EmbodiedAgents_ は、[Automatika Robotics](https://automatikarobotics.com/) と [Inria](https://inria.fr/) の協力により開発されました。
コミュニティからの貢献も大歓迎です。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="_static/EMBODIED_AGENTS_DARK.png">
  <source media="(prefers-color-scheme: light)" srcset="_static/EMBODIED_AGENTS_LIGHT.png">
  <img alt="_EmbodiedAgents_ ロゴ" src="_static/EMBODIED_AGENTS_DARK.png">
</picture>
<br/>

> 🌐 [English Version](../README.md) | 🇨🇳 [简体中文](README.zh.md)

**_EmbodiedAgents_** は、環境のコンテキスト情報を理解し、記憶し、それに基づいて行動できる対話型の物理エージェントを作成するための、完全に ROS2 で構築されたフル機能のフレームワークです。

- **本番対応の物理エージェント**：実世界の動的な環境で動作する自律型ロボットシステムのために設計されています。_EmbodiedAgents_ を使えば、Physical AI を簡単に活用できます。
- **直感的な API**：ローカルまたはクラウドベースの機械学習モデル（特に **マルチモーダル LLM** やその他の **トランスフォーマーアーキテクチャ**）を使用するためのシンプルな Python スタイルの API。
- **セマンティックメモリ**：ベクトルデータベース、セマンティックルーティング、その他の補助コンポーネントを統合し、エージェント情報フローの複雑なグラフを迅速に構築できます。重い「GenAI」フレームワークは不要です。
- **ROS2 ベース**：ROS2 を分散通信の基盤として使用。ROS2 パッケージを提供するすべてのデバイスが ML モデルへのデータ送信に利用可能で、主要なデータ型に対応したコールバックが用意されており、拡張性は無限です。

[インストール手順](https://automatika-robotics.github.io/embodied-agents/installation.html) 🛠️
[クイックスタートガイド](https://automatika-robotics.github.io/embodied-agents/quickstart.html) 🚀
[基本概念](https://automatika-robotics.github.io/embodied-agents/basics.html) 📚
[サンプル集](https://automatika-robotics.github.io/embodied-agents/examples/index.html) ✨

## インストール 🛠️

### モデルサービングプラットフォームのインストール

_EmbodiedAgents_ のコアは、特定のモデルサービングプラットフォームに依存しません。現在は [Ollama](https://ollama.com)、[RoboML](https://github.com/automatika-robotics/robo-ml)、および OpenAI 互換 API を備えた任意のプラットフォームやクラウドプロバイダー（例：[vLLM](https://github.com/vllm-project/vllm)、[lmdeploy](https://github.com/InternLM/lmdeploy) など）をサポートしています。これらのいずれかを使用するには、各プロジェクトの指示に従ってインストールしてください。新しいプラットフォームへの対応は継続的に追加されています。特定のプラットフォームをサポートしたい場合は、Issue や PR を提出してください。

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
pip install numpy opencv-python-headless 'attrs>=23.2.0' jinja2 httpx setproctitle msgpack msgpack-numpy platformdirs tqdm
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

このコードを実行することで、**「何が見える？」** といった質問に答えるエージェントが完成します。_EmbodiedAgents_ には軽量なウェブクライアントも付属しています。[クイックスタートガイド](https://automatika-robotics.github.io/embodied-agents/quickstart.html) で、コンポーネントとモデルの連携方法を学びましょう。

## 複雑な物理エージェント

上記のクイックスタートは、_EmbodiedAgents_ の機能のごく一部にすぎません。EmbodiedAgents では、任意に複雑なコンポーネントグラフを構築できます。さらに、システム内部または外部のイベントに応じて、構成を動的に変更・再構築することも可能です。以下のエージェントのコード例を確認してみてください：[こちらをクリック](https://automatika-robotics.github.io/embodied-agents/examples/complete.html)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="_static/complete_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="_static/complete_light.png">
  <img alt="高度なエージェント" src="_static/complete_dark.png">
</picture>

## 著作権情報

本配布物に含まれるコードは、特に明記されていない限り、すべて © 2024 [Automatika Robotics](https://automatikarobotics.com/) に著作権があります。

_EmbodiedAgents_ は MIT ライセンスのもとで公開されています。詳細は [LICENSE](LICENSE) ファイルをご確認ください。

## コントリビューション（貢献）

_EmbodiedAgents_ は、[Automatika Robotics](https://automatikarobotics.com/) と [Inria](https://inria.fr/) の協力により開発されました。
コミュニティからの貢献も大歓迎です。

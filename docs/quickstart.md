# Quick Start

Unlike other ROS package, _EmbodiedAgents_ provides a pure pythonic way of describing the node graph using [Sugarcoat🍬](https://automatika-robotics.github.io/sugarcoat/). Copy the following code in a python script and run it.

```{important}
Depending on the components and clients you use, _EmbodiedAgents_ will prompt you for extra python packages. The script will throw an error and let you know how you can install these extra pacakges.
```

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
    trigger=text0,
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

Now let us see step-by-step what we have done in this code. First we defined inputs and outputs to our component in the form of ROS Topics. Components automatically create listeners for input topics and publishers for output topics.

```python
# Define input and output topics (pay attention to msg_type)
text0 = Topic(name="text0", msg_type="String")
image0 = Topic(name="image_raw", msg_type="Image")
text1 = Topic(name="text1", msg_type="String")
```

````{important}
If you are running _EmbodiedAgents_ on a robot, make sure you change the name of the topic to which the robot's camera is publishing the RGB images to in the following line.

```python
image0 = Topic(name="NAME_OF_THE_TOPIC", msg_type="Image")
````

```{note}
If you are running _EmbodiedAgents_ on a testing machine, and the machine has a webcam, you can install the [**ROS2 USB Cam**](https://github.com/klintan/ros2_usb_camera). Make sure you use the correct name of the image topic as above.
```

Then we will create a multimodal LLM component. Components are functional units in _EmbodiedAgents_. To learn more about them, check out [Basic Concepts](basics/components.md). Other than input/output topics, the VLM component expects a model client. So first we will create a model client that can utilize a [Llava](https://ollama.com/library/llava) model on [Ollama](https://ollama.com) as its model serving platform.

```python
# Define a model client (working with Ollama in this case)
# OllamaModel is a generic wrapper for all Ollama models
llava = OllamaModel(name="llava", checkpoint="llava:latest")
llava_client = OllamaClient(llava)
```

````{important}
If you are not running Ollama on the same machine (robot) on which you are running _EmbodiedAgents_, you can define access to the machine running Ollama using host and port in this line:
```python
llava_client = OllamaClient(llava, host="127.0.0.1", port=8000)
````

```{note}
If the use of Ollama as a model serving platform is unclear, checkout [installation instructions](installation.md).
```

Now we are ready to setup our component.

```python
# Define a VLM component (A component represents a node with a particular functionality)
mllm = VLM(
    inputs=[text0, image0],
    outputs=[text1],
    model_client=llava_client,
    trigger=text0,
    component_name="vqa"
)
# Additional prompt settings
mllm.set_topic_prompt(text0, template="""You are an amazing and funny robot.
    Answer the following about this image: {{ text0 }}"""
)
```

Note how the VLM type of component, also allows us to set a topic or component level prompt, where a jinja2 template can be used to define a template in which our input string should be embedded. Finally we will launch the component.

```python
# Launch the component
launcher = Launcher()
launcher.add_pkg(components=[mllm])
launcher.bringup()
```

Now we can check that our component is running by using familiar ROS2 commands from a new terminal. We should see our component running as a ROS node and the its input and output topics in the topic list.

```shell
ros2 node list
ros2 topic list
```

In order to interact with our component _EmbodiedAgents_ can dynamically generate a web-based UI for us. We can make the client available by adding the following line to our code that tells the launcher which topics to render:

```python
# Launch the component
launcher = Launcher()
launcher.enable_ui(inputs=[text0], outputs=[text1, image0])  # <-- specify UI
launcher.add_pkg(components=[mllm])
launcher.bringup()
```

````{note}
In order to run the client you will need to install [FastHTML](https://www.fastht.ml/) and [MonsterUI](https://github.com/AnswerDotAI/MonsterUI) with
```shell
pip install python-fasthtml monsterui
````

The client displays a web UI on **http://localhost:5001** if you have run it on your machine. Or you can access it at **http://<IP_ADDRESS_OF_THE_ROBOT>:5001** if you have run it on the robot.

Open this address from browser. Component settings can be configured from the web UI by pressing the settings button. Send a question to your ROS EmbodiedAgent and you should get a the reply generated by the Llava model.

![Demo screencast](_static/agents_ui.gif)

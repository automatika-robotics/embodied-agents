# Use a MultiModal Planning Model for Vision Guided Navigation

In a previous [recipe](goto.md) we created an agent capable of understanding and responding to go-to commands. This agent relied on a semantic map that was stored in a vector database that could be accessed by an LLM component for doing retreival augmented generation. Through the magic of tool use (or manual post-processing), we were able to extract position coordinates from our vectorized information and send it to a `Pose` topic for goal-point navigation by an autonomous navigation system. In this example, we would see how we can generate a similar navigation goal, but from the visual input coming in from the robot's sensors, i.e. we should be able to ask our physical agent to navigate to an object that is in its sight.

We will acheive this by utilizing two components in our agent. An LLM component and an MLLM component. The LLM component will act as a sentence parser, isolating the object description from the user's command. The MLLM component will use a planning Vision Language Model (VLM), which can perform visual grounding and pointing.

## Initialize the LLM component

```python
from agents.components import LLM
from agents.models import OllamaModel
from agents.clients import OllamaClient
from agents.ros import Topic

# Start a Llama3.2 based llm component using ollama client
llama = OllamaModel(name="llama", checkpoint="llama3.2:3b")
llama_client = OllamaClient(llama)

# Define LLM input and output topics including goal_point topic of type PoseStamped
goto_in = Topic(name="goto_in", msg_type="String")
llm_output = Topic(name="llm_output", msg_type="String")

# initialize the component
sentence_parser = LLM(
    inputs=[goto_in],
    outputs=[llm_output],
    model_client=llama_client,
    trigger=goto_in,
    component_name='sentence_parser'
)
```

In order to configure the component to act as a sentence parser, we will set a topic prompt on its input topic.

```python
sentence_parser.set_topic_prompt(goto_in, template="""You are a sentence parsing software.
Simply return the object description in the following command. {{ goto_in }}"""
)
```

## Initialize the MLLM component

In this step, we will set up the MLLM component, which will enable the agent to visually ground natural language object descriptions (from our command, given to the LLM component above) using live sensor data. We use **[RoboBrain 2.0](https://github.com/FlagOpen/RoboBrain2.0)** by BAAI, a state-of-the-art Vision-Language model (VLM) trained specifically for embodied agents reasoning.

RoboBrain 2.0 supports a wide range of embodied perception and planning capabilities, including interactive reasoning and spatial perception.

> 📄 **Citation**:
> BAAI RoboBrain Team. "RoboBrain 2.0 Technical Report." arXiv preprint arXiv:2507.02029 (2025).
> [https://arxiv.org/abs/2507.02029](https://arxiv.org/abs/2507.02029)

In our scenario, we use RoboBrain2.0 to perform **grounding**—that is, mapping the object description (parsed by the LLM component) to a visual detection in the agent’s camera view. This detection includes spatial coordinates that can be forwarded to the navigation system for physical movement. RoboBrain2.0 is available in RoboML, which we are using as a model serving platform here.

```{note}
RoboML is an aggregator library that provides a model serving aparatus for locally serving opensource ML models useful in robotics. Learn about setting up RoboML [here](https://www.github.com/automatika-robotics/roboml).
```

To configure this grounding behavious, we initialize an `MLLMConfig` object and set the `task` parameter to `"grounding"`:

```python
config = MLLMConfig(task="grounding")
```

```{note}
The `task` parameter specifies the type of multimodal operation the component should perform.
Supported values are:
* `"general"` – free-form multimodal reasoning, produces output of type String
* `"pointing"` – provide a list of points on the object, produces output of type PointsOfInterest
* `"affordance"` – detect object affordances, produces output of type Detections
* `"trajectory"` – predict motion path in pixel space, produces output of type PointsOfInterst
* `"grounding"` – localize an object in the scene from a description with a bounding box, produces output of type Detections

This parameter ensures the model behaves in a task-specific way, especially when using models like RoboBrain 2.0 that have been trained on multiple multimodal instruction types.
```

With this setup, the MLLM component receives parsed object descriptions from the LLM and produces structured `Detections` messages identifying the object’s location in space—enabling the agent to navigate towards a visually grounded goal. Furthermore, we will use an _RGBD_ type message and the image input to the MLLM component. This message is an aligned RGB and depth image message that is usually available in the ROS2 packages provided by stereo camera vendors (e.g. Realsense). The utility of this choice, would become apparent later in this tutorial.

```python
from agents.components import MLLM
from agents.models import RoboBrain2
from agents.clients import RoboMLHTTPClient
from agents.config import MLLMConfig

# Start a RoboBrain2 based mllm component using RoboML client
robobrain = RoboBrain2(name="robobrain")
robobrain_client = RoboMLHTTPClient(robobrain)

# Define MLLM output topic
rgbd0 = Topic(name="rgbd0", msg_type="RGBD")
grounding_output = Topic(name="grounding_output", msg_type="Detections")

# Set the task in MLLMConfig
config = MLLMConfig(task="grounding")

# initialize the component
go_to_x = MLLM(
    inputs=[llm_output],
    outputs=[grounding_output],
    model_client=robobrain_client,
    trigger=llm_output,
    config=config,
    component_name="go-to-x"
)
```

```{Warning}
When a task is specified in MLLMConfig, the MLLM component automatically produces structured output depending on the task. The downstream consumers of this input should have appropriate callbacks configured for handling these output messages.
```

## **BONUS** - Configure Autonomous Navigation with **_Kompass_**

[Kompass](https://automatika-robotics.github.io/kompass) is the most advanced, GPU powered and featue complete open-source navigation stack out there. Its built with the same underlying principles as _EmbodiedAgents_, thus it is event-driven and can be customized with a simple python script. In this section we will, show how to start _Kompass_ in the same recipe that we have been developing for a vision guided, goto agent.

```{note}
Learn about installing Kompass [here](https://automatika-robotics.github.io/kompass/install.html)
```

_Kompass_ allows for various kinds of navigation behaviour configured in the same recipe. However, we will only be using point-to-point navigation and the default configuration for its components. Since _Kompass_ is a navigation stack, as a first step, we will configure the robot and its motion model as follows:

```python
import numpy as np
from kompass.robot import (
    AngularCtrlLimits,
    LinearCtrlLimits,
    RobotGeometry,
    RobotType,
)
from kompass.config import RobotConfig

# Setup your robot configuration
my_robot = RobotConfig(
    model_type=RobotType.DIFFERENTIAL_DRIVE,
    geometry_type=RobotGeometry.Type.CYLINDER,
    geometry_params=np.array([0.1, 0.3]),
    ctrl_vx_limits=LinearCtrlLimits(max_vel=0.2, max_acc=1.5, max_decel=2.5),
    ctrl_omega_limits=AngularCtrlLimits(
        max_vel=0.4, max_acc=2.0, max_decel=2.0, max_steer=np.pi / 3
    ),
)
```

Now we can add our default components. Our component of interest is the _planning_ component, that plots a path to the goal point. We will give the output topic from our MLLM component as the goal point topic to the planning component.

```{important}
While planning components typically require goal points as `Pose` or `PoseStamped` messages in world space, Kompass also accepts `Detection` and `PointOfInterest` messages from EmbodiedAgents. These contain pixel-space coordinates identified by ML models. When generated from RGBD inputs, the associated depth images are included, enabling Kompass to automatically convert pixel-space points to averaged world-space coordinates using camera intrinsics.
```

```python
from kompass.components import (
    Controller,
    Planner,
    DriveManager,
    LocalMapper,
)

# Setup components with default config, inputs and outputs
planner = Planner(component_name="planner")

# Set our grounding output as the goal_point in the planner component
planner.inputs(goal_point=grounding_output)

# Get a default Local Mapper component
mapper = LocalMapper(component_name="mapper")

# Get a default controller component
controller = Controller(component_name="controller")
# Configure Controller to use local map instead of direct sensor information
controller.direct_sensor = False

# Setup a default drive manager
driver = DriveManager(component_name="drive_manager")
```

```{seealso}
Learn the details of point navigation in Kompass using this step-by-step [tutorial](https://automatika-robotics.github.io/kompass/tutorials/point_navigation.htm)
```

## Launching the Components

Now we will launch our Go-to-X component and Kompass components using the same launcher. We will get the Launcher from Kompass this time.

```python
from kompass.launcher import Launcher

launcher = Launcher()

# Add the components from EmbodiedAgents
launcher.add_pkg(components=[sentence_parser, go_to_x], ros_log_level="warn",
                 package_name="automatika_embodied_agents",
                 executable_entry_point="executable",
                 multiprocessing=True)

# Add the components from Kompass as follows
launcher.kompass(components=[planner, controller, mapper, driver])

# Set the robot config for all components as defined above and bring up
launcher.robot = my_robot
launcher.bringup()
```

And that is all. Our Go-to-X component is ready. The complete code for this example is given below:

```{code-block} python
:caption: Vision Guided Go-to-X Component
:linenos:
import numpy as np
from agents.components import LLM
from agents.models import OllamaModel
from agents.clients import OllamaClient
from agents.ros import Topic
from agents.components import MLLM
from agents.models import RoboBrain2
from agents.clients import RoboMLHTTPClient
from agents.config import MLLMConfig
from kompass.robot import (
    AngularCtrlLimits,
    LinearCtrlLimits,
    RobotGeometry,
    RobotType,
)
from kompass.config import RobotConfig
from kompass.components import (
    Controller,
    Planner,
    DriveManager,
    LocalMapper,
)
from kompass.launcher import Launcher


# Start a Llama3.2 based llm component using ollama client
llama = OllamaModel(name="llama", checkpoint="llama3.2:3b")
llama_client = OllamaClient(llama)

# Define LLM input and output topics including goal_point topic of type PoseStamped
goto_in = Topic(name="goto_in", msg_type="String")
llm_output = Topic(name="llm_output", msg_type="String")

# initialize the component
sentence_parser = LLM(
    inputs=[goto_in],
    outputs=[llm_output],
    model_client=llama_client,
    trigger=goto_in,
    component_name='sentence_parser'
)

# Start a RoboBrain2 based mllm component using RoboML client
robobrain = RoboBrain2(name="robobrain")
robobrain_client = RoboMLHTTPClient(robobrain)

# Define MLLM output topic
rgbd0 = Topic(name="rgbd0", msg_type="RGBD")
grounding_output = Topic(name="grounding_output", msg_type="Detections")

# Set the task in MLLMConfig
config = MLLMConfig(task="grounding")

# initialize the component
go_to_x = MLLM(
    inputs=[llm_output],
    outputs=[grounding_output],
    model_client=robobrain_client,
    trigger=llm_output,
    config=config,
    component_name="go-to-x"
)

# Setup your robot configuration
my_robot = RobotConfig(
    model_type=RobotType.DIFFERENTIAL_DRIVE,
    geometry_type=RobotGeometry.Type.CYLINDER,
    geometry_params=np.array([0.1, 0.3]),
    ctrl_vx_limits=LinearCtrlLimits(max_vel=0.2, max_acc=1.5, max_decel=2.5),
    ctrl_omega_limits=AngularCtrlLimits(
        max_vel=0.4, max_acc=2.0, max_decel=2.0, max_steer=np.pi / 3
    ),
)

# Setup components with default config, inputs and outputs
planner = Planner(component_name="planner")

# Set our grounding output as the goal_point in the planner component
planner.inputs(goal_point=grounding_output)

# Get a default Local Mapper component
mapper = LocalMapper(component_name="mapper")

# Get a default controller component
controller = Controller(component_name="controller")
# Configure Controller to use local map instead of direct sensor information
controller.direct_sensor = False

# Setup a default drive manager
driver = DriveManager(component_name="drive_manager")

launcher = Launcher()

# Add the components from EmbodiedAgents
launcher.add_pkg(components=[sentence_parser, go_to_x], ros_log_level="warn",
                 package_name="automatika_embodied_agents",
                 executable_entry_point="executable",
                 multiprocessing=True)

# Add the components from Kompass as follows
launcher.kompass(components=[planner, controller, mapper, driver])

# Set the robot config for all components as defined above and bring up
launcher.robot = my_robot
launcher.bringup()
```

# Basic Concepts 📚

The following is an overview of basic building blocks of EmbodiedAgents. You can follow the links in each subsection to dig deeper.

## Component

A Component is the main execution unit in EmbodiedAgents and in essence each component is synctactic sugar over a ROS2 Lifecycle Node. All the functionalities implemented in ROS2 nodes can be found in the component. Components take a single Topic or a list of Topics as inputs and ouputs. Depending on the components functionality, certain types of Topics might be mandatory.

```{note}
To learn more about components, checkout [Sugarcoat Documentation](https://automatika-robotics.github.io/sugarcoat/).
```

### Components Available in EmbodiedAgents

EmbodiedAgents provides various ready to use components. You can see their details [here](apidocs/agents/agents.components).

### Component Config

Each component can take in an optional config. Configs are generally [attrs](https://www.attrs.org/en/stable/) classes and for components that use ML models, configs are also the place where inference parameters are defined. You can see the default options for configs of each available component [here](apidocs/agents/agents.config).

### Component RunType

In EmbodiedAgents, components can be of the following two types:

```{list-table}
:widths: 10 80
* - **Timed**
  - Execute the main execution function in a timed loop.
* - **Event**
  - Execute the main execution function based on a trigger topic/event.
```

### Health Check and Fallback

Each component maintains a health status, based on which, one can configure various fallback options for the component allowing it to recover from failures or shutdown gracefully. This aspect can be significant in embodied autonomous agents, not just in terms of safety but for generally coherent and reliable performance. To learn more about these topics, check out the documentation of [Sugarcoat Documentation](https://automatika-robotics.github.io/sugarcoat/).

## Topic

A [topic](apidocs/agents/agents.ros) is an idomatic wrapper for a ROS2 topic. Topics can be given as inputs or outputs to components. When given as inputs, components automatically create listeners for the topics upon their activation. And when given as outputs, components create publishers for publishing to the topic. Each topic has a name (duh?) and a data type, defining its listening callback and publishing behavior. The data type can be provided to the topic as a string. Checkout the list of supported data types [here](https://automatika-robotics.github.io/sugarcoat/advanced/types.html).

```{note}
Learn more about Topics in [Sugarcoat](https://automatika-robotics.github.io/sugarcoat/).
```

## Model/DB Client

Certain components in EmbodiedAgents deal with ML models, vector DBs or both. These components take in a model or db client as one of their initialization parameters. The reason for this separate abstraction is to enforce _separation of concerns_. An ML model can be running on the edge hardware itself, or a powerful compute node in the network, or in the cloud, the components running on the robot edge can always use the model (or DB) via a client in a standardized way. This also makes the components independant of the model serving platforms, which can implement various inference optimizations which are usually model specific. Thus one can choose an ML serving platform with the best latency/accuracy tradeoff, depending on the application concerns.

All clients implement a connection check. ML clients must implement a model inference and optionally model initialization and deinitialization methods (since an embodied agent can initialize different models (or fine tuned versions of the same model) for the same component, depending on some event in the environment). Similarly vector DB clients implement standard CRUD methods for vector DBs. Checkout the list of available clients [here](apidocs/agents/agents.clients).

## Models/DBs

The clients we mentioned above take as input a model or vector database specification. These are in the form of [attrs](https://www.attrs.org/en/stable/) classes and define intialization parameters, such as quantization for ML models or choice of encoding model for vector DBs, among others. The available models and databases that can be instantiated on a particular model serving platform usually depend on the platform itself. However, with these model and vector DB specifications, we aim to standardize the model initialization specifications across platforms. Check the list of [models](apidocs/agents/agents.models) and [vector DBs](apidocs/agents/agents.vectordbs) that are available.

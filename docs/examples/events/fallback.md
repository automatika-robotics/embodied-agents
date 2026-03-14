# Runtime Robustness: Model Fallback

In the real world, connections drop, APIs time out, and servers crash. Sticking with the theme of robustness, a "Production Ready" agent cannot simply freeze when it's internet connection is lost.

In this tutorial, we will demonstrate the self-referential capabilities of **EmbodiedAgents**. We will build an agent that uses a high-intelligence model (hosted remotely) as its primary _brain_, but automatically switches to a backup model if the primary one fails.

## Approach 1: Local Model Fallback

The simplest fallback strategy uses the component's built-in local model support. When the primary (remote) model fails, the component automatically deploys a lightweight local model that runs **in-process** — no external server needed.

This approach works for all model components: `LLM`, `MLLM`/`VLM`, `Vision`, `SpeechToText`, and `TextToSpeech`.

### 1. Define the Model and Client

First, we define the primary remote model client that the component will use under normal conditions.

```python
from agents.components import LLM
from agents.models import TransformersLLM
from agents.clients import RoboMLHTTPClient
from agents.config import LLMConfig
from agents.ros import Launcher, Topic, Action

# A powerful model hosted remotely (e.g., via RoboML).
# NOTE: This is illustrative for executing on a local machine.
# For a production scenario, you might use a GenericHTTPClient pointing to
# GPT-5, Gemini, HuggingFace Inference etc.
primary_model = TransformersLLM(
    name="qwen_heavy", checkpoint="Qwen/Qwen2.5-1.5B-Instruct"
)
primary_client = RoboMLHTTPClient(model=primary_model)
```

### 2. Configure the Component

Set up the component with the primary client as usual.

```python
# Define Topics
user_query = Topic(name="user_query", msg_type="String")
llm_response = Topic(name="llm_response", msg_type="String")

# Configure the LLM Component with the primary client
llm_component = LLM(
    inputs=[user_query],
    outputs=[llm_response],
    model_client=primary_client,
    trigger=user_query,
    component_name="brain",
)
```

### 3. Wire Up the Fallback

Create an `Action` that calls `fallback_to_local()` and bind it to the component's failure events. That's it — one line to define the action, two to bind it.

```python
# Define the Fallback Action
switch_to_local = Action(method=llm_component.fallback_to_local)

# Bind Failures to the Action
llm_component.on_component_fail(action=switch_to_local, max_retries=3)
llm_component.on_algorithm_fail(action=switch_to_local, max_retries=3)
```

When triggered, `fallback_to_local()` will:

1. Auto-enable local model support in the config
2. Lazily deploy the local model (only loading it into memory on first failure)
3. Deinitialize the remote client
4. Route all subsequent inference through the local model

```{note}
The local model is deployed lazily — it only loads into memory when the first failure occurs. This avoids consuming GPU/CPU resources until actually needed.
```

### The Complete Recipe

```python
from agents.components import LLM
from agents.models import TransformersLLM
from agents.clients import RoboMLHTTPClient
from agents.config import LLMConfig
from agents.ros import Launcher, Topic, Action

# 1. Define the Model and Client
# A powerful model hosted remotely (e.g., via RoboML).
# NOTE: This is illustrative for executing on a local machine.
# For a production scenario, you might use a GenericHTTPClient pointing to
# GPT-5, Gemini, HuggingFace Inference etc.
primary_model = TransformersLLM(
    name="qwen_heavy", checkpoint="Qwen/Qwen2.5-1.5B-Instruct"
)
primary_client = RoboMLHTTPClient(model=primary_model)

# 2. Define Topics
user_query = Topic(name="user_query", msg_type="String")
llm_response = Topic(name="llm_response", msg_type="String")

# 3. Configure the LLM Component
llm_component = LLM(
    inputs=[user_query],
    outputs=[llm_response],
    model_client=primary_client,
    trigger=user_query,
    component_name="brain",
)

# 4. Define the Fallback Action
switch_to_local = Action(method=llm_component.fallback_to_local)

# 5. Bind Failures to the Action
llm_component.on_component_fail(action=switch_to_local, max_retries=3)
llm_component.on_algorithm_fail(action=switch_to_local, max_retries=3)

# 6. Launch
launcher = Launcher()
launcher.add_pkg(
    components=[llm_component],
    multiprocessing=True,
    package_name="automatika_embodied_agents",
)
launcher.bringup()
```

---

## Approach 2: Remote Client Swap

If you want to fall back to another **remote** model (e.g., swapping from a cloud API to a locally-hosted Ollama server), use the `change_model_client` method with `additional_model_clients`.

### 1. Define the Models and Clients

First, we define two distinct model clients: a primary and a backup.

```python
from agents.components import LLM
from agents.models import OllamaModel, TransformersLLM
from agents.clients import OllamaClient, RoboMLHTTPClient
from agents.config import LLMConfig
from agents.ros import Launcher, Topic, Action

# A powerful model hosted remotely (e.g., via RoboML).
# NOTE: This is illustrative for executing on a local machine.
# For a production scenario, you might use a GenericHTTPClient pointing to
# GPT-5, Gemini, HuggingFace Inference etc.
primary_model = TransformersLLM(
    name="qwen_heavy", checkpoint="Qwen/Qwen2.5-1.5B-Instruct"
)
primary_client = RoboMLHTTPClient(model=primary_model)

# A smaller model running locally (via Ollama) that works offline.
backup_model = OllamaModel(name="llama_local", checkpoint="llama3.2:3b")
backup_client = OllamaClient(model=backup_model)
```

### 2. Configure the Component

Set up the component with the primary client. The `additional_model_clients` attribute allows the component to hold references to other valid clients that are waiting in the wings.

```python
# Define Topics
user_query = Topic(name="user_query", msg_type="String")
llm_response = Topic(name="llm_response", msg_type="String")

# Configure the LLM Component with the primary client
llm_component = LLM(
    inputs=[user_query],
    outputs=[llm_response],
    model_client=primary_client,
    component_name="brain",
    config=LLMConfig(stream=True),
)

# Register the Backup Client
llm_component.additional_model_clients = {"local_backup_client": backup_client}
```

### 3. Wire Up the Fallback

Create an `Action` that calls `change_model_client()` and bind it to the component's failure events. We pass the key (`'local_backup_client'`) defined in the previous step.

```{note}
All components implement some default actions as well as component specific actions. In this case we are implementing a component specific action.
```

```{seealso}
To see a list of default actions available to all components, checkout Sugarcoat🍬 [Documentation](https://automatika-robotics.github.io/sugarcoat/design/actions.html)
```

```python
# Define the Fallback Action
switch_to_backup = Action(
    method=llm_component.change_model_client,
    args=("local_backup_client",)
)

# Bind Failures to the Action
llm_component.on_component_fail(action=switch_to_backup, max_retries=3)
llm_component.on_algorithm_fail(action=switch_to_backup, max_retries=3)
```

- **`on_component_fail`**: Triggered if the component crashes or fails to initialize (e.g., the remote server is down when the robot starts).
- **`on_algorithm_fail`**: Triggered if the component is running, but the inference fails (e.g., the WiFi drops mid-conversation).

```{note}
**Why `max_retries`?** Sometimes a fallback can temporarily fail as well. The system will attempt to restart the component or algorithm up to 3 times while applying the action (switching the client) to resolve the error. This is an _optional_ parameter.
```

### The Complete Recipe

Here is the full code. To test this, you can try shutting down your RoboML server (or disconnecting the internet) while the agent is running, and watch it seamlessly switch to the local Llama model.

```python
from agents.components import LLM
from agents.models import OllamaModel, TransformersLLM
from agents.clients import OllamaClient, RoboMLHTTPClient
from agents.config import LLMConfig
from agents.ros import Launcher, Topic, Action

# 1. Define the Models and Clients
# A powerful model hosted remotely (e.g., via RoboML).
# NOTE: This is illustrative for executing on a local machine.
# For a production scenario, you might use a GenericHTTPClient pointing to
# GPT-5, Gemini, HuggingFace Inference etc.
primary_model = TransformersLLM(
    name="qwen_heavy", checkpoint="Qwen/Qwen2.5-1.5B-Instruct"
)
primary_client = RoboMLHTTPClient(model=primary_model)

# A smaller model running locally (via Ollama) that works offline.
backup_model = OllamaModel(name="llama_local", checkpoint="llama3.2:3b")
backup_client = OllamaClient(model=backup_model)

# 2. Define Topics
user_query = Topic(name="user_query", msg_type="String")
llm_response = Topic(name="llm_response", msg_type="String")

# 3. Configure the LLM Component
llm_component = LLM(
    inputs=[user_query],
    outputs=[llm_response],
    model_client=primary_client,
    component_name="brain",
    config=LLMConfig(stream=True),
)

# 4. Register the Backup Client
llm_component.additional_model_clients = {"local_backup_client": backup_client}

# 5. Define the Fallback Action
switch_to_backup = Action(
    method=llm_component.change_model_client,
    args=("local_backup_client",)
)

# 6. Bind Failures to the Action
llm_component.on_component_fail(action=switch_to_backup, max_retries=3)
llm_component.on_algorithm_fail(action=switch_to_backup, max_retries=3)

# 7. Launch
launcher = Launcher()
launcher.add_pkg(
    components=[llm_component],
    multiprocessing=True,
    package_name="automatika_embodied_agents",
)
launcher.bringup()
```

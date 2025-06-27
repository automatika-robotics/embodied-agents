# Basic Concepts 📚

Welcome to the core concepts of the **_EmbodiedAgents_** framework. This section introduces the three fundamental building blocks you'll work with:

* 🧩 **Components**: Basic building blocks of _EmbodiedAgents_.
* 🔌 **Clients**: Execution backends that instantiate and call inference on ML models served on local or remote platforms.
* 🧠 **Models / Vector DBs**: Configurations for ML models (like LLMs, MLLMs, TTS, vision etc.) or databases used by components.

Each of these building blocks can be composed, configured, and executed flexibly, enabling powerful embodied agent-based applications across modalities.

## Components 🧩

Components are the modular building blocks that define the behavior of an agent. They can represent anything that can be termed as functional behaviour. For example the ability to understand the process text. Each component defines:

* Inputs and outputs (which are ROS topics)
* Functional logic, including runtime time configuration parameters.
* Health status and fallback mechanisms

Components can be combined arbitrarily to create more complex systems such as multi-modal agents with perception-action loops.

📘 Learn more: [Components](components.md)

## Clients 🔌

Clients are execution backends that instantiate and call inference on ML models.

* Communicate with remote model serving platforms.
* Various communication protocols, HTTP, WebSockets, RESP.

Clients abstract away the underlying model serving details, allowing you to be agnostic to the platforms that the machine learning model is served on.

📘 Learn more: [Clients](clients.md)

## Models / Vector DBs 🧠

Components often rely on underlying machine learning models or vector databases. These are defined as **specifications**, such as:

* LLMs, MLLMs, Whisper, TTS, vision models
* Vector DBs like ChromaDB for semantic retreival

These models and DBs are standardized using `attrs`-based classes, so you can easily plug them into compatible clients regardless of platform or backend.

📘 Learn more: [Models & Vector DBs](models.md)

---

```{toctree}
:maxdepth: 1

components
clients
models
```

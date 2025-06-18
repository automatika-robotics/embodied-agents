"""
Clients are standard interfaces for components to interact with ML models or vector DBs served by various platforms. Currently EmbodiedAgents provides the following clients, which cover the most popular open source model deployment platforms. Simple clients can be easily implemented for other platforms and the use of unnecessarily heavy duct-tape "AI" frameworks on the robot is discouraged 😅.

```{note}
Some clients might need additional dependacies, which are provided in the following table. If missing the user will also be prompted for them at runtime.
```

```{list-table}
:widths: 20 20 60
:header-rows: 1
* - Platform
  - Client
  - Description

* - **Generic**
  - [GenericHTTPClient](agents.clients.generic.GenericHTTPClient)
  - A generic client for interacting with OpenAI-compatible APIs, including vLLM, ms-swift, lmdeploy, Google Gemini etc. This client works with LLM multimodal LLM models and supports both standard and streaming responses. It is designed to be compatible with any API that follows the OpenAI standard.

* - **RoboML**
  - [RoboMLHTTPClient](agents.clients.roboml.RoboMLHTTPClient)
  - An HTTP client for interaction with ML models served on RoboML.

* - **RoboML**
  - [RoboMLWSClient](agents.clients.roboml.RoboMLWSClient)
  - A websocket based client for persistent interaction with ML models served on RoboML. Specially useful for low latency streaming of audio or streaming text data.

* - **RoboML**
  - [RoboMLRESPClient](agents.clients.roboml.RoboMLRESPClient)
  - A Redis Serialization Protocol (RESP) based client for interaction with ML models served on RoboML. **Note:** In order to use this client, please install dependencies with `pip install redis[hiredis]`

* - **Ollama**
  - [OllamaClient](agents.clients.ollama.OllamaClient)
  - An HTTP client for interaction with ML models served on Ollama. **Note:** In order to use this client, please install dependencies with `pip install ollama`

* - **ChromaDB**
  - [ChromaClient](agents.clients.chroma.ChromaClient)
  - An HTTP client for interaction with a ChromaDB instance running as a server. Before using this client, make sure an instance of ChromaDB is running on the given host and port by executing `chroma run --path /db_path`
"""

from .generic import GenericHTTPClient
from .ollama import OllamaClient
from .roboml import (
    RoboMLHTTPClient,
    RoboMLRESPClient,
    RoboMLWSClient,
)
from .chroma import ChromaClient


__all__ = [
    "GenericHTTPClient",
    "OllamaClient",
    "ChromaClient",
    "RoboMLHTTPClient",
    "RoboMLRESPClient",
    "RoboMLWSClient",
]

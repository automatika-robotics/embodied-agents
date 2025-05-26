"""
Clients are standard interfaces for components to interact with ML models or vector DBs served by various platforms. Currently ROS Agents provides the following clients, which cover the most popular open source model deployment platforms. Simple clients can be easily implemented for other platforms and the use of heavy duct-tape "AI" frameworks on the robot is discouraged 😅.

```{note}
Some clients might need additional dependacies, which are provided in the following table. If missing the user will also be prompted for them at runtime.
```

```{list-table}
:widths: 20 20 60
:header-rows: 1
* - Platform
  - Client
  - Description

* - **RoboML**
  - [HTTPModelClient](agents.clients.roboml.HTTPModelClient)
  - An HTTP client for interaction with ML models served on RoboML.

* - **RoboML**
  - [HTTPDBClient](agents.clients.roboml.HTTPDBClient)
  - An HTTP client for interaction with vector DBs served on RoboML.

* - **RoboML**
  - [RESPModelClient](agents.clients.roboml.RESPModelClient)
  - A Redis Serialization Protocol (RESP) based client for interaction with ML models served on RoboML. **Note:** In order to use this client, please install dependancies with `pip install redis[hiredis] msgpack msgpack-numpy`

* - **RoboML**
  - [RESPDBClient](agents.clients.roboml.RESPDBClient)
  - A Redis Serialization Protocol (RESP) based client for interaction with vector DBs served on RoboML. **Note:** In order to use this client, please install dependancies with `pip install redis[hiredis] msgpack msgpack-numpy`

* - **Ollama**
  - [OllamaClient](agents.clients.ollama.OllamaClient)
  - An HTTP client for interaction with ML models served on Ollama. **Note:** In order to use this client, please install dependancies with `pip install ollama`

"""

from .ollama import OllamaClient
from .roboml import (
    HTTPDBClient,
    HTTPModelClient,
    RESPDBClient,
    RESPModelClient,
    WebSocketClient,
)


__all__ = [
    "OllamaClient",
    "HTTPDBClient",
    "HTTPModelClient",
    "WebSocketClient",
    "RESPDBClient",
    "RESPModelClient",
]

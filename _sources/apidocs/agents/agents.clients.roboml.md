---
orphan: true
---

# {py:mod}`agents.clients.roboml`

```{py:module} agents.clients.roboml
```

```{autodoc2-docstring} agents.clients.roboml
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RoboMLHTTPClient <agents.clients.roboml.RoboMLHTTPClient>`
  - ```{autodoc2-docstring} agents.clients.roboml.RoboMLHTTPClient
    :summary:
    ```
* - {py:obj}`RoboMLRESPClient <agents.clients.roboml.RoboMLRESPClient>`
  - ```{autodoc2-docstring} agents.clients.roboml.RoboMLRESPClient
    :summary:
    ```
* - {py:obj}`RoboMLWSClient <agents.clients.roboml.RoboMLWSClient>`
  - ```{autodoc2-docstring} agents.clients.roboml.RoboMLWSClient
    :summary:
    ```
````

### API

`````{py:class} RoboMLHTTPClient(model: typing.Union[agents.models.Model, typing.Dict], host: str = '127.0.0.1', port: int = 8000, inference_timeout: int = 30, init_on_activation: bool = True, logging_level: str = 'info', **kwargs)
:canonical: agents.clients.roboml.RoboMLHTTPClient

Bases: {py:obj}`agents.clients.model_base.ModelClient`

```{autodoc2-docstring} agents.clients.roboml.RoboMLHTTPClient
```

````{py:method} serialize() -> typing.Dict
:canonical: agents.clients.roboml.RoboMLHTTPClient.serialize

```{autodoc2-docstring} agents.clients.roboml.RoboMLHTTPClient.serialize
```

````

````{py:method} check_connection() -> None
:canonical: agents.clients.roboml.RoboMLHTTPClient.check_connection

```{autodoc2-docstring} agents.clients.roboml.RoboMLHTTPClient.check_connection
```

````

````{py:method} initialize() -> None
:canonical: agents.clients.roboml.RoboMLHTTPClient.initialize

```{autodoc2-docstring} agents.clients.roboml.RoboMLHTTPClient.initialize
```

````

````{py:method} inference(inference_input: typing.Dict[str, typing.Any]) -> typing.Optional[typing.Dict]
:canonical: agents.clients.roboml.RoboMLHTTPClient.inference

```{autodoc2-docstring} agents.clients.roboml.RoboMLHTTPClient.inference
```

````

````{py:method} deinitialize()
:canonical: agents.clients.roboml.RoboMLHTTPClient.deinitialize

```{autodoc2-docstring} agents.clients.roboml.RoboMLHTTPClient.deinitialize
```

````

`````

`````{py:class} RoboMLRESPClient(model: typing.Union[agents.models.Model, typing.Dict], host: str = '127.0.0.1', port: int = 6379, inference_timeout: int = 30, init_on_activation: bool = True, logging_level: str = 'info', **kwargs)
:canonical: agents.clients.roboml.RoboMLRESPClient

Bases: {py:obj}`agents.clients.model_base.ModelClient`

```{autodoc2-docstring} agents.clients.roboml.RoboMLRESPClient
```

````{py:method} serialize() -> typing.Dict
:canonical: agents.clients.roboml.RoboMLRESPClient.serialize

```{autodoc2-docstring} agents.clients.roboml.RoboMLRESPClient.serialize
```

````

````{py:method} check_connection() -> None
:canonical: agents.clients.roboml.RoboMLRESPClient.check_connection

```{autodoc2-docstring} agents.clients.roboml.RoboMLRESPClient.check_connection
```

````

````{py:method} initialize() -> None
:canonical: agents.clients.roboml.RoboMLRESPClient.initialize

```{autodoc2-docstring} agents.clients.roboml.RoboMLRESPClient.initialize
```

````

````{py:method} inference(inference_input: typing.Dict[str, typing.Any]) -> typing.Optional[typing.Dict]
:canonical: agents.clients.roboml.RoboMLRESPClient.inference

```{autodoc2-docstring} agents.clients.roboml.RoboMLRESPClient.inference
```

````

````{py:method} deinitialize()
:canonical: agents.clients.roboml.RoboMLRESPClient.deinitialize

```{autodoc2-docstring} agents.clients.roboml.RoboMLRESPClient.deinitialize
```

````

`````

`````{py:class} RoboMLWSClient(model: typing.Union[agents.models.Model, typing.Dict], host: str = '127.0.0.1', port: int = 8000, inference_timeout: int = 30, init_on_activation: bool = True, logging_level: str = 'info', **kwargs)
:canonical: agents.clients.roboml.RoboMLWSClient

Bases: {py:obj}`agents.clients.roboml.RoboMLHTTPClient`

```{autodoc2-docstring} agents.clients.roboml.RoboMLWSClient
```

````{py:method} serialize() -> typing.Dict
:canonical: agents.clients.roboml.RoboMLWSClient.serialize

```{autodoc2-docstring} agents.clients.roboml.RoboMLWSClient.serialize
```

````

````{py:method} check_connection() -> None
:canonical: agents.clients.roboml.RoboMLWSClient.check_connection

```{autodoc2-docstring} agents.clients.roboml.RoboMLWSClient.check_connection
```

````

````{py:method} initialize() -> None
:canonical: agents.clients.roboml.RoboMLWSClient.initialize

```{autodoc2-docstring} agents.clients.roboml.RoboMLWSClient.initialize
```

````

````{py:method} inference(inference_input: typing.Dict[str, typing.Any]) -> typing.Optional[typing.Dict]
:canonical: agents.clients.roboml.RoboMLWSClient.inference

```{autodoc2-docstring} agents.clients.roboml.RoboMLWSClient.inference
```

````

````{py:method} deinitialize()
:canonical: agents.clients.roboml.RoboMLWSClient.deinitialize

```{autodoc2-docstring} agents.clients.roboml.RoboMLWSClient.deinitialize
```

````

`````

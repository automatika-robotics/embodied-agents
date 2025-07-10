---
orphan: true
---

# {py:mod}`agents.clients.ollama`

```{py:module} agents.clients.ollama
```

```{autodoc2-docstring} agents.clients.ollama
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OllamaClient <agents.clients.ollama.OllamaClient>`
  - ```{autodoc2-docstring} agents.clients.ollama.OllamaClient
    :summary:
    ```
````

### API

`````{py:class} OllamaClient(model: typing.Union[agents.models.OllamaModel, typing.Dict], host: str = '127.0.0.1', port: int = 11434, inference_timeout: int = 30, init_on_activation: bool = True, logging_level: str = 'info', **kwargs)
:canonical: agents.clients.ollama.OllamaClient

Bases: {py:obj}`agents.clients.model_base.ModelClient`

```{autodoc2-docstring} agents.clients.ollama.OllamaClient
```

````{py:method} serialize() -> typing.Dict
:canonical: agents.clients.ollama.OllamaClient.serialize

```{autodoc2-docstring} agents.clients.ollama.OllamaClient.serialize
```

````

````{py:method} check_connection() -> None
:canonical: agents.clients.ollama.OllamaClient.check_connection

```{autodoc2-docstring} agents.clients.ollama.OllamaClient.check_connection
```

````

````{py:method} initialize() -> None
:canonical: agents.clients.ollama.OllamaClient.initialize

```{autodoc2-docstring} agents.clients.ollama.OllamaClient.initialize
```

````

````{py:method} inference(inference_input: typing.Dict[str, typing.Any]) -> typing.Optional[typing.Dict]
:canonical: agents.clients.ollama.OllamaClient.inference

```{autodoc2-docstring} agents.clients.ollama.OllamaClient.inference
```

````

````{py:method} deinitialize()
:canonical: agents.clients.ollama.OllamaClient.deinitialize

```{autodoc2-docstring} agents.clients.ollama.OllamaClient.deinitialize
```

````

`````

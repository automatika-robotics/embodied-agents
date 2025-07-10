---
orphan: true
---

# {py:mod}`agents.clients.generic`

```{py:module} agents.clients.generic
```

```{autodoc2-docstring} agents.clients.generic
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GenericHTTPClient <agents.clients.generic.GenericHTTPClient>`
  - ```{autodoc2-docstring} agents.clients.generic.GenericHTTPClient
    :summary:
    ```
````

### API

`````{py:class} GenericHTTPClient(model: typing.Union[agents.models.LLM, typing.Dict], host: str = '127.0.0.1', port: typing.Optional[int] = 8000, inference_timeout: int = 30, api_key: typing.Optional[str] = None, logging_level: str = 'info', **kwargs)
:canonical: agents.clients.generic.GenericHTTPClient

Bases: {py:obj}`agents.clients.model_base.ModelClient`

```{autodoc2-docstring} agents.clients.generic.GenericHTTPClient
```

````{py:method} serialize() -> typing.Dict
:canonical: agents.clients.generic.GenericHTTPClient.serialize

```{autodoc2-docstring} agents.clients.generic.GenericHTTPClient.serialize
```

````

````{py:method} check_connection() -> None
:canonical: agents.clients.generic.GenericHTTPClient.check_connection

```{autodoc2-docstring} agents.clients.generic.GenericHTTPClient.check_connection
```

````

````{py:method} initialize() -> None
:canonical: agents.clients.generic.GenericHTTPClient.initialize

```{autodoc2-docstring} agents.clients.generic.GenericHTTPClient.initialize
```

````

````{py:method} inference(inference_input: typing.Dict[str, typing.Any]) -> typing.Optional[typing.Dict]
:canonical: agents.clients.generic.GenericHTTPClient.inference

```{autodoc2-docstring} agents.clients.generic.GenericHTTPClient.inference
```

````

````{py:method} deinitialize()
:canonical: agents.clients.generic.GenericHTTPClient.deinitialize

```{autodoc2-docstring} agents.clients.generic.GenericHTTPClient.deinitialize
```

````

`````

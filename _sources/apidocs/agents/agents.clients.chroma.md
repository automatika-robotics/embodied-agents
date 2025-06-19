---
orphan: true
---

# {py:mod}`agents.clients.chroma`

```{py:module} agents.clients.chroma
```

```{autodoc2-docstring} agents.clients.chroma
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ChromaClient <agents.clients.chroma.ChromaClient>`
  - ```{autodoc2-docstring} agents.clients.chroma.ChromaClient
    :summary:
    ```
````

### API

`````{py:class} ChromaClient(db: typing.Union[agents.vectordbs.DB, typing.Dict], host: str = '127.0.0.1', port: int = 8000, response_timeout: int = 30, init_on_activation: bool = True, logging_level: str = 'info', **kwargs)
:canonical: agents.clients.chroma.ChromaClient

Bases: {py:obj}`agents.clients.db_base.DBClient`

```{autodoc2-docstring} agents.clients.chroma.ChromaClient
```

````{py:method} serialize() -> typing.Dict
:canonical: agents.clients.chroma.ChromaClient.serialize

```{autodoc2-docstring} agents.clients.chroma.ChromaClient.serialize
```

````

````{py:method} check_connection() -> None
:canonical: agents.clients.chroma.ChromaClient.check_connection

```{autodoc2-docstring} agents.clients.chroma.ChromaClient.check_connection
```

````

````{py:method} initialize() -> None
:canonical: agents.clients.chroma.ChromaClient.initialize

```{autodoc2-docstring} agents.clients.chroma.ChromaClient.initialize
```

````

````{py:method} add(db_input: typing.Dict[str, typing.Any]) -> typing.Optional[typing.Dict]
:canonical: agents.clients.chroma.ChromaClient.add

```{autodoc2-docstring} agents.clients.chroma.ChromaClient.add
```

````

````{py:method} conditional_add(db_input: typing.Dict[str, typing.Any]) -> typing.Optional[typing.Dict]
:canonical: agents.clients.chroma.ChromaClient.conditional_add

```{autodoc2-docstring} agents.clients.chroma.ChromaClient.conditional_add
```

````

````{py:method} metadata_query(db_input: typing.Dict[str, typing.Any]) -> typing.Optional[typing.Dict]
:canonical: agents.clients.chroma.ChromaClient.metadata_query

```{autodoc2-docstring} agents.clients.chroma.ChromaClient.metadata_query
```

````

````{py:method} query(db_input: typing.Dict[str, typing.Any]) -> typing.Optional[typing.Dict]
:canonical: agents.clients.chroma.ChromaClient.query

```{autodoc2-docstring} agents.clients.chroma.ChromaClient.query
```

````

````{py:method} deinitialize() -> None
:canonical: agents.clients.chroma.ChromaClient.deinitialize

```{autodoc2-docstring} agents.clients.chroma.ChromaClient.deinitialize
```

````

`````

---
orphan: true
---

# {py:mod}`agents.components.mllm`

```{py:module} agents.components.mllm
```

```{autodoc2-docstring} agents.components.mllm
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MLLM <agents.components.mllm.MLLM>`
  - ```{autodoc2-docstring} agents.components.mllm.MLLM
    :summary:
    ```
````

### API

`````{py:class} MLLM(*, inputs: typing.List[typing.Union[agents.ros.Topic, agents.ros.FixedInput]], outputs: typing.List[agents.ros.Topic], model_client: agents.clients.model_base.ModelClient, config: typing.Optional[agents.config.MLLMConfig] = None, db_client: typing.Optional[agents.clients.db_base.DBClient] = None, trigger: typing.Union[agents.ros.Topic, typing.List[agents.ros.Topic], float] = 1.0, component_name: str, **kwargs)
:canonical: agents.components.mllm.MLLM

Bases: {py:obj}`agents.components.llm.LLM`

```{autodoc2-docstring} agents.components.mllm.MLLM
```

````{py:method} custom_on_configure()
:canonical: agents.components.mllm.MLLM.custom_on_configure

```{autodoc2-docstring} agents.components.mllm.MLLM.custom_on_configure
```

````

````{py:method} set_task(task: typing.Literal[general, pointing, affordance, trajectory, grounding]) -> None
:canonical: agents.components.mllm.MLLM.set_task

```{autodoc2-docstring} agents.components.mllm.MLLM.set_task
```

````

````{py:method} custom_on_deactivate()
:canonical: agents.components.mllm.MLLM.custom_on_deactivate

```{autodoc2-docstring} agents.components.mllm.MLLM.custom_on_deactivate
```

````

````{py:method} add_documents(ids: typing.List[str], metadatas: typing.List[typing.Dict], documents: typing.List[str]) -> None
:canonical: agents.components.mllm.MLLM.add_documents

```{autodoc2-docstring} agents.components.mllm.MLLM.add_documents
```

````

````{py:method} set_topic_prompt(input_topic: agents.ros.Topic, template: typing.Union[str, pathlib.Path]) -> None
:canonical: agents.components.mllm.MLLM.set_topic_prompt

```{autodoc2-docstring} agents.components.mllm.MLLM.set_topic_prompt
```

````

````{py:method} set_component_prompt(template: typing.Union[str, pathlib.Path]) -> None
:canonical: agents.components.mllm.MLLM.set_component_prompt

```{autodoc2-docstring} agents.components.mllm.MLLM.set_component_prompt
```

````

````{py:method} set_system_prompt(prompt: str) -> None
:canonical: agents.components.mllm.MLLM.set_system_prompt

```{autodoc2-docstring} agents.components.mllm.MLLM.set_system_prompt
```

````

````{py:method} register_tool(tool: typing.Callable, tool_description: typing.Dict, send_tool_response_to_model: bool = False) -> None
:canonical: agents.components.mllm.MLLM.register_tool

```{autodoc2-docstring} agents.components.mllm.MLLM.register_tool
```

````

````{py:property} warmup
:canonical: agents.components.mllm.MLLM.warmup
:type: bool

```{autodoc2-docstring} agents.components.mllm.MLLM.warmup
```

````

````{py:method} custom_on_activate()
:canonical: agents.components.mllm.MLLM.custom_on_activate

```{autodoc2-docstring} agents.components.mllm.MLLM.custom_on_activate
```

````

````{py:method} create_all_subscribers()
:canonical: agents.components.mllm.MLLM.create_all_subscribers

```{autodoc2-docstring} agents.components.mllm.MLLM.create_all_subscribers
```

````

````{py:method} activate_all_triggers() -> None
:canonical: agents.components.mllm.MLLM.activate_all_triggers

```{autodoc2-docstring} agents.components.mllm.MLLM.activate_all_triggers
```

````

````{py:method} destroy_all_subscribers() -> None
:canonical: agents.components.mllm.MLLM.destroy_all_subscribers

```{autodoc2-docstring} agents.components.mllm.MLLM.destroy_all_subscribers
```

````

````{py:method} trigger(trigger: typing.Union[agents.ros.Topic, typing.List[agents.ros.Topic], float]) -> None
:canonical: agents.components.mllm.MLLM.trigger

```{autodoc2-docstring} agents.components.mllm.MLLM.trigger
```

````

````{py:method} validate_topics(topics: typing.Sequence[typing.Union[agents.ros.Topic, agents.ros.FixedInput]], allowed_topic_types: typing.Optional[typing.Dict[str, typing.List[typing.Union[typing.Type[agents.ros.SupportedType], typing.List[typing.Type[agents.ros.SupportedType]]]]]] = None, topics_direction: str = 'Topics')
:canonical: agents.components.mllm.MLLM.validate_topics

```{autodoc2-docstring} agents.components.mllm.MLLM.validate_topics
```

````

`````

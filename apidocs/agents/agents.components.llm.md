---
orphan: true
---

# {py:mod}`agents.components.llm`

```{py:module} agents.components.llm
```

```{autodoc2-docstring} agents.components.llm
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LLM <agents.components.llm.LLM>`
  - ```{autodoc2-docstring} agents.components.llm.LLM
    :summary:
    ```
````

### API

`````{py:class} LLM(*, inputs: typing.List[typing.Union[agents.ros.Topic, agents.ros.FixedInput]], outputs: typing.List[agents.ros.Topic], model_client: agents.clients.model_base.ModelClient, config: typing.Optional[agents.config.LLMConfig] = None, db_client: typing.Optional[agents.clients.db_base.DBClient] = None, trigger: typing.Union[agents.ros.Topic, typing.List[agents.ros.Topic], float] = 1.0, component_name: str, **kwargs)
:canonical: agents.components.llm.LLM

Bases: {py:obj}`agents.components.model_component.ModelComponent`

```{autodoc2-docstring} agents.components.llm.LLM
```

````{py:method} custom_on_configure()
:canonical: agents.components.llm.LLM.custom_on_configure

```{autodoc2-docstring} agents.components.llm.LLM.custom_on_configure
```

````

````{py:method} custom_on_deactivate()
:canonical: agents.components.llm.LLM.custom_on_deactivate

```{autodoc2-docstring} agents.components.llm.LLM.custom_on_deactivate
```

````

````{py:method} add_documents(ids: typing.List[str], metadatas: typing.List[typing.Dict], documents: typing.List[str]) -> None
:canonical: agents.components.llm.LLM.add_documents

```{autodoc2-docstring} agents.components.llm.LLM.add_documents
```

````

````{py:method} set_topic_prompt(input_topic: agents.ros.Topic, template: typing.Union[str, pathlib.Path]) -> None
:canonical: agents.components.llm.LLM.set_topic_prompt

```{autodoc2-docstring} agents.components.llm.LLM.set_topic_prompt
```

````

````{py:method} set_component_prompt(template: typing.Union[str, pathlib.Path]) -> None
:canonical: agents.components.llm.LLM.set_component_prompt

```{autodoc2-docstring} agents.components.llm.LLM.set_component_prompt
```

````

````{py:method} set_system_prompt(prompt: str) -> None
:canonical: agents.components.llm.LLM.set_system_prompt

```{autodoc2-docstring} agents.components.llm.LLM.set_system_prompt
```

````

````{py:method} register_tool(tool: typing.Callable, tool_description: typing.Dict, send_tool_response_to_model: bool = False) -> None
:canonical: agents.components.llm.LLM.register_tool

```{autodoc2-docstring} agents.components.llm.LLM.register_tool
```

````

````{py:property} warmup
:canonical: agents.components.llm.LLM.warmup
:type: bool

```{autodoc2-docstring} agents.components.llm.LLM.warmup
```

````

````{py:method} custom_on_activate()
:canonical: agents.components.llm.LLM.custom_on_activate

```{autodoc2-docstring} agents.components.llm.LLM.custom_on_activate
```

````

````{py:method} create_all_subscribers()
:canonical: agents.components.llm.LLM.create_all_subscribers

```{autodoc2-docstring} agents.components.llm.LLM.create_all_subscribers
```

````

````{py:method} activate_all_triggers() -> None
:canonical: agents.components.llm.LLM.activate_all_triggers

```{autodoc2-docstring} agents.components.llm.LLM.activate_all_triggers
```

````

````{py:method} destroy_all_subscribers() -> None
:canonical: agents.components.llm.LLM.destroy_all_subscribers

```{autodoc2-docstring} agents.components.llm.LLM.destroy_all_subscribers
```

````

````{py:method} trigger(trigger: typing.Union[agents.ros.Topic, typing.List[agents.ros.Topic], float]) -> None
:canonical: agents.components.llm.LLM.trigger

```{autodoc2-docstring} agents.components.llm.LLM.trigger
```

````

````{py:method} validate_topics(topics: typing.Sequence[typing.Union[agents.ros.Topic, agents.ros.FixedInput]], allowed_topic_types: typing.Optional[typing.Dict[str, typing.List[typing.Union[typing.Type[agents.ros.SupportedType], typing.List[typing.Type[agents.ros.SupportedType]]]]]] = None, topics_direction: str = 'Topics')
:canonical: agents.components.llm.LLM.validate_topics

```{autodoc2-docstring} agents.components.llm.LLM.validate_topics
```

````

`````

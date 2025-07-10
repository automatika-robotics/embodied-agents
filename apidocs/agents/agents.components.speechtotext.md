---
orphan: true
---

# {py:mod}`agents.components.speechtotext`

```{py:module} agents.components.speechtotext
```

```{autodoc2-docstring} agents.components.speechtotext
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SpeechToText <agents.components.speechtotext.SpeechToText>`
  - ```{autodoc2-docstring} agents.components.speechtotext.SpeechToText
    :summary:
    ```
````

### API

`````{py:class} SpeechToText(*, inputs: typing.List[agents.ros.Topic], outputs: typing.List[agents.ros.Topic], model_client: agents.clients.model_base.ModelClient, config: typing.Optional[agents.config.SpeechToTextConfig] = None, trigger: typing.Union[agents.ros.Topic, typing.List[agents.ros.Topic]], component_name: str, **kwargs)
:canonical: agents.components.speechtotext.SpeechToText

Bases: {py:obj}`agents.components.model_component.ModelComponent`

```{autodoc2-docstring} agents.components.speechtotext.SpeechToText
```

````{py:method} custom_on_activate()
:canonical: agents.components.speechtotext.SpeechToText.custom_on_activate

```{autodoc2-docstring} agents.components.speechtotext.SpeechToText.custom_on_activate
```

````

````{py:method} custom_on_deactivate()
:canonical: agents.components.speechtotext.SpeechToText.custom_on_deactivate

```{autodoc2-docstring} agents.components.speechtotext.SpeechToText.custom_on_deactivate
```

````

````{py:method} custom_on_configure()
:canonical: agents.components.speechtotext.SpeechToText.custom_on_configure

```{autodoc2-docstring} agents.components.speechtotext.SpeechToText.custom_on_configure
```

````

````{py:property} warmup
:canonical: agents.components.speechtotext.SpeechToText.warmup
:type: bool

```{autodoc2-docstring} agents.components.speechtotext.SpeechToText.warmup
```

````

````{py:method} create_all_subscribers()
:canonical: agents.components.speechtotext.SpeechToText.create_all_subscribers

```{autodoc2-docstring} agents.components.speechtotext.SpeechToText.create_all_subscribers
```

````

````{py:method} activate_all_triggers() -> None
:canonical: agents.components.speechtotext.SpeechToText.activate_all_triggers

```{autodoc2-docstring} agents.components.speechtotext.SpeechToText.activate_all_triggers
```

````

````{py:method} destroy_all_subscribers() -> None
:canonical: agents.components.speechtotext.SpeechToText.destroy_all_subscribers

```{autodoc2-docstring} agents.components.speechtotext.SpeechToText.destroy_all_subscribers
```

````

````{py:method} trigger(trigger: typing.Union[agents.ros.Topic, typing.List[agents.ros.Topic], float]) -> None
:canonical: agents.components.speechtotext.SpeechToText.trigger

```{autodoc2-docstring} agents.components.speechtotext.SpeechToText.trigger
```

````

````{py:method} validate_topics(topics: typing.Sequence[typing.Union[agents.ros.Topic, agents.ros.FixedInput]], allowed_topic_types: typing.Optional[typing.Dict[str, typing.List[typing.Union[typing.Type[agents.ros.SupportedType], typing.List[typing.Type[agents.ros.SupportedType]]]]]] = None, topics_direction: str = 'Topics')
:canonical: agents.components.speechtotext.SpeechToText.validate_topics

```{autodoc2-docstring} agents.components.speechtotext.SpeechToText.validate_topics
```

````

`````

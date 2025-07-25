---
orphan: true
---

# {py:mod}`agents.components.texttospeech`

```{py:module} agents.components.texttospeech
```

```{autodoc2-docstring} agents.components.texttospeech
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TextToSpeech <agents.components.texttospeech.TextToSpeech>`
  - ```{autodoc2-docstring} agents.components.texttospeech.TextToSpeech
    :summary:
    ```
````

### API

`````{py:class} TextToSpeech(*, inputs: typing.List[agents.ros.Topic], outputs: typing.Optional[typing.List[agents.ros.Topic]] = None, model_client: agents.clients.model_base.ModelClient, config: typing.Optional[agents.config.TextToSpeechConfig] = None, trigger: typing.Union[agents.ros.Topic, typing.List[agents.ros.Topic]], component_name: str, **kwargs)
:canonical: agents.components.texttospeech.TextToSpeech

Bases: {py:obj}`agents.components.model_component.ModelComponent`

```{autodoc2-docstring} agents.components.texttospeech.TextToSpeech
```

````{py:method} custom_on_configure()
:canonical: agents.components.texttospeech.TextToSpeech.custom_on_configure

```{autodoc2-docstring} agents.components.texttospeech.TextToSpeech.custom_on_configure
```

````

````{py:method} custom_on_deactivate()
:canonical: agents.components.texttospeech.TextToSpeech.custom_on_deactivate

```{autodoc2-docstring} agents.components.texttospeech.TextToSpeech.custom_on_deactivate
```

````

````{py:method} stop_playback(wait_for_thread: bool = True)
:canonical: agents.components.texttospeech.TextToSpeech.stop_playback

```{autodoc2-docstring} agents.components.texttospeech.TextToSpeech.stop_playback
```

````

````{py:method} say(text: str)
:canonical: agents.components.texttospeech.TextToSpeech.say

```{autodoc2-docstring} agents.components.texttospeech.TextToSpeech.say
```

````

````{py:property} warmup
:canonical: agents.components.texttospeech.TextToSpeech.warmup
:type: bool

```{autodoc2-docstring} agents.components.texttospeech.TextToSpeech.warmup
```

````

````{py:method} custom_on_activate()
:canonical: agents.components.texttospeech.TextToSpeech.custom_on_activate

```{autodoc2-docstring} agents.components.texttospeech.TextToSpeech.custom_on_activate
```

````

````{py:method} create_all_subscribers()
:canonical: agents.components.texttospeech.TextToSpeech.create_all_subscribers

```{autodoc2-docstring} agents.components.texttospeech.TextToSpeech.create_all_subscribers
```

````

````{py:method} activate_all_triggers() -> None
:canonical: agents.components.texttospeech.TextToSpeech.activate_all_triggers

```{autodoc2-docstring} agents.components.texttospeech.TextToSpeech.activate_all_triggers
```

````

````{py:method} destroy_all_subscribers() -> None
:canonical: agents.components.texttospeech.TextToSpeech.destroy_all_subscribers

```{autodoc2-docstring} agents.components.texttospeech.TextToSpeech.destroy_all_subscribers
```

````

````{py:method} trigger(trigger: typing.Union[agents.ros.Topic, typing.List[agents.ros.Topic], float]) -> None
:canonical: agents.components.texttospeech.TextToSpeech.trigger

```{autodoc2-docstring} agents.components.texttospeech.TextToSpeech.trigger
```

````

````{py:method} validate_topics(topics: typing.Sequence[typing.Union[agents.ros.Topic, agents.ros.FixedInput]], allowed_topic_types: typing.Optional[typing.Dict[str, typing.List[typing.Union[typing.Type[agents.ros.SupportedType], typing.List[typing.Type[agents.ros.SupportedType]]]]]] = None, topics_direction: str = 'Topics')
:canonical: agents.components.texttospeech.TextToSpeech.validate_topics

```{autodoc2-docstring} agents.components.texttospeech.TextToSpeech.validate_topics
```

````

`````

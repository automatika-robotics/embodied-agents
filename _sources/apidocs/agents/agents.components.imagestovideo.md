---
orphan: true
---

# {py:mod}`agents.components.imagestovideo`

```{py:module} agents.components.imagestovideo
```

```{autodoc2-docstring} agents.components.imagestovideo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VideoMessageMaker <agents.components.imagestovideo.VideoMessageMaker>`
  - ```{autodoc2-docstring} agents.components.imagestovideo.VideoMessageMaker
    :summary:
    ```
````

### API

`````{py:class} VideoMessageMaker(*, inputs: typing.List[agents.ros.Topic], outputs: typing.List[agents.ros.Topic], config: typing.Optional[agents.config.VideoMessageMakerConfig] = None, trigger: typing.Union[agents.ros.Topic, typing.List[agents.ros.Topic]], component_name: str, **kwargs)
:canonical: agents.components.imagestovideo.VideoMessageMaker

Bases: {py:obj}`agents.components.component_base.Component`

```{autodoc2-docstring} agents.components.imagestovideo.VideoMessageMaker
```

````{py:method} custom_on_activate()
:canonical: agents.components.imagestovideo.VideoMessageMaker.custom_on_activate

```{autodoc2-docstring} agents.components.imagestovideo.VideoMessageMaker.custom_on_activate
```

````

````{py:method} create_all_subscribers()
:canonical: agents.components.imagestovideo.VideoMessageMaker.create_all_subscribers

```{autodoc2-docstring} agents.components.imagestovideo.VideoMessageMaker.create_all_subscribers
```

````

````{py:method} activate_all_triggers() -> None
:canonical: agents.components.imagestovideo.VideoMessageMaker.activate_all_triggers

```{autodoc2-docstring} agents.components.imagestovideo.VideoMessageMaker.activate_all_triggers
```

````

````{py:method} destroy_all_subscribers() -> None
:canonical: agents.components.imagestovideo.VideoMessageMaker.destroy_all_subscribers

```{autodoc2-docstring} agents.components.imagestovideo.VideoMessageMaker.destroy_all_subscribers
```

````

````{py:method} trigger(trigger: typing.Union[agents.ros.Topic, typing.List[agents.ros.Topic], float]) -> None
:canonical: agents.components.imagestovideo.VideoMessageMaker.trigger

```{autodoc2-docstring} agents.components.imagestovideo.VideoMessageMaker.trigger
```

````

````{py:method} validate_topics(topics: typing.Sequence[typing.Union[agents.ros.Topic, agents.ros.FixedInput]], allowed_topic_types: typing.Optional[typing.Dict[str, typing.List[typing.Union[typing.Type[agents.ros.SupportedType], typing.List[typing.Type[agents.ros.SupportedType]]]]]] = None, topics_direction: str = 'Topics')
:canonical: agents.components.imagestovideo.VideoMessageMaker.validate_topics

```{autodoc2-docstring} agents.components.imagestovideo.VideoMessageMaker.validate_topics
```

````

`````

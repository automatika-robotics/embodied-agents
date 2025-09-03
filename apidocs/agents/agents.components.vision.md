---
orphan: true
---

# {py:mod}`agents.components.vision`

```{py:module} agents.components.vision
```

```{autodoc2-docstring} agents.components.vision
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Vision <agents.components.vision.Vision>`
  - ```{autodoc2-docstring} agents.components.vision.Vision
    :summary:
    ```
````

### API

`````{py:class} Vision(*, inputs: typing.List[typing.Union[agents.ros.Topic, agents.ros.FixedInput]], outputs: typing.List[agents.ros.Topic], model_client: typing.Optional[agents.clients.model_base.ModelClient] = None, config: typing.Optional[agents.config.VisionConfig] = None, trigger: typing.Union[agents.ros.Topic, typing.List[agents.ros.Topic], float] = 1.0, component_name: str, **kwargs)
:canonical: agents.components.vision.Vision

Bases: {py:obj}`agents.components.model_component.ModelComponent`

```{autodoc2-docstring} agents.components.vision.Vision
```

````{py:method} custom_on_configure()
:canonical: agents.components.vision.Vision.custom_on_configure

```{autodoc2-docstring} agents.components.vision.Vision.custom_on_configure
```

````

````{py:method} custom_on_deactivate()
:canonical: agents.components.vision.Vision.custom_on_deactivate

```{autodoc2-docstring} agents.components.vision.Vision.custom_on_deactivate
```

````

````{py:property} warmup
:canonical: agents.components.vision.Vision.warmup
:type: bool

```{autodoc2-docstring} agents.components.vision.Vision.warmup
```

````

````{py:method} custom_on_activate()
:canonical: agents.components.vision.Vision.custom_on_activate

```{autodoc2-docstring} agents.components.vision.Vision.custom_on_activate
```

````

````{py:method} create_all_subscribers()
:canonical: agents.components.vision.Vision.create_all_subscribers

```{autodoc2-docstring} agents.components.vision.Vision.create_all_subscribers
```

````

````{py:method} activate_all_triggers() -> None
:canonical: agents.components.vision.Vision.activate_all_triggers

```{autodoc2-docstring} agents.components.vision.Vision.activate_all_triggers
```

````

````{py:method} destroy_all_subscribers() -> None
:canonical: agents.components.vision.Vision.destroy_all_subscribers

```{autodoc2-docstring} agents.components.vision.Vision.destroy_all_subscribers
```

````

````{py:method} trigger(trigger: typing.Union[agents.ros.Topic, typing.List[agents.ros.Topic], float]) -> None
:canonical: agents.components.vision.Vision.trigger

```{autodoc2-docstring} agents.components.vision.Vision.trigger
```

````

````{py:method} validate_topics(topics: typing.Sequence[typing.Union[agents.ros.Topic, agents.ros.FixedInput]], allowed_topic_types: typing.Optional[typing.Dict[str, typing.List[typing.Union[typing.Type[agents.ros.SupportedType], typing.List[typing.Type[agents.ros.SupportedType]]]]]] = None, topics_direction: str = 'Topics')
:canonical: agents.components.vision.Vision.validate_topics

```{autodoc2-docstring} agents.components.vision.Vision.validate_topics
```

````

`````

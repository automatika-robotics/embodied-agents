---
orphan: true
---

# {py:mod}`agents.components.semantic_router`

```{py:module} agents.components.semantic_router
```

```{autodoc2-docstring} agents.components.semantic_router
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SemanticRouter <agents.components.semantic_router.SemanticRouter>`
  - ```{autodoc2-docstring} agents.components.semantic_router.SemanticRouter
    :summary:
    ```
````

### API

`````{py:class} SemanticRouter(*, inputs: typing.List[agents.ros.Topic], routes: typing.List[agents.ros.Route], config: agents.config.SemanticRouterConfig, db_client: agents.clients.db_base.DBClient, default_route: typing.Optional[agents.ros.Route] = None, component_name: str, **kwargs)
:canonical: agents.components.semantic_router.SemanticRouter

Bases: {py:obj}`agents.components.component_base.Component`

```{autodoc2-docstring} agents.components.semantic_router.SemanticRouter
```

````{py:method} custom_on_activate()
:canonical: agents.components.semantic_router.SemanticRouter.custom_on_activate

```{autodoc2-docstring} agents.components.semantic_router.SemanticRouter.custom_on_activate
```

````

````{py:method} create_all_subscribers()
:canonical: agents.components.semantic_router.SemanticRouter.create_all_subscribers

```{autodoc2-docstring} agents.components.semantic_router.SemanticRouter.create_all_subscribers
```

````

````{py:method} activate_all_triggers() -> None
:canonical: agents.components.semantic_router.SemanticRouter.activate_all_triggers

```{autodoc2-docstring} agents.components.semantic_router.SemanticRouter.activate_all_triggers
```

````

````{py:method} destroy_all_subscribers() -> None
:canonical: agents.components.semantic_router.SemanticRouter.destroy_all_subscribers

```{autodoc2-docstring} agents.components.semantic_router.SemanticRouter.destroy_all_subscribers
```

````

````{py:method} trigger(trigger: typing.Union[agents.ros.Topic, typing.List[agents.ros.Topic], float]) -> None
:canonical: agents.components.semantic_router.SemanticRouter.trigger

```{autodoc2-docstring} agents.components.semantic_router.SemanticRouter.trigger
```

````

````{py:method} validate_topics(topics: typing.Sequence[typing.Union[agents.ros.Topic, agents.ros.FixedInput]], allowed_topic_types: typing.Optional[typing.Dict[str, typing.List[typing.Union[typing.Type[agents.ros.SupportedType], typing.List[typing.Type[agents.ros.SupportedType]]]]]] = None, topics_direction: str = 'Topics')
:canonical: agents.components.semantic_router.SemanticRouter.validate_topics

```{autodoc2-docstring} agents.components.semantic_router.SemanticRouter.validate_topics
```

````

`````

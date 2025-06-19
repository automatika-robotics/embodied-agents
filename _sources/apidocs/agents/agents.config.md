---
orphan: true
---

# {py:mod}`agents.config`

```{py:module} agents.config
```

```{autodoc2-docstring} agents.config
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LLMConfig <agents.config.LLMConfig>`
  - ```{autodoc2-docstring} agents.config.LLMConfig
    :summary:
    ```
* - {py:obj}`MLLMConfig <agents.config.MLLMConfig>`
  - ```{autodoc2-docstring} agents.config.MLLMConfig
    :summary:
    ```
* - {py:obj}`SpeechToTextConfig <agents.config.SpeechToTextConfig>`
  - ```{autodoc2-docstring} agents.config.SpeechToTextConfig
    :summary:
    ```
* - {py:obj}`TextToSpeechConfig <agents.config.TextToSpeechConfig>`
  - ```{autodoc2-docstring} agents.config.TextToSpeechConfig
    :summary:
    ```
* - {py:obj}`SemanticRouterConfig <agents.config.SemanticRouterConfig>`
  - ```{autodoc2-docstring} agents.config.SemanticRouterConfig
    :summary:
    ```
* - {py:obj}`MapConfig <agents.config.MapConfig>`
  - ```{autodoc2-docstring} agents.config.MapConfig
    :summary:
    ```
* - {py:obj}`VideoMessageMakerConfig <agents.config.VideoMessageMakerConfig>`
  - ```{autodoc2-docstring} agents.config.VideoMessageMakerConfig
    :summary:
    ```
* - {py:obj}`VisionConfig <agents.config.VisionConfig>`
  - ```{autodoc2-docstring} agents.config.VisionConfig
    :summary:
    ```
````

### API

`````{py:class} LLMConfig
:canonical: agents.config.LLMConfig

Bases: {py:obj}`agents.config.ModelComponentConfig`

```{autodoc2-docstring} agents.config.LLMConfig
```

````{py:method} get_inference_params() -> typing.Dict
:canonical: agents.config.LLMConfig.get_inference_params

```{autodoc2-docstring} agents.config.LLMConfig.get_inference_params
```

````

`````

`````{py:class} MLLMConfig
:canonical: agents.config.MLLMConfig

Bases: {py:obj}`agents.config.LLMConfig`

```{autodoc2-docstring} agents.config.MLLMConfig
```

````{py:method} get_inference_params() -> typing.Dict
:canonical: agents.config.MLLMConfig.get_inference_params

```{autodoc2-docstring} agents.config.MLLMConfig.get_inference_params
```

````

`````

`````{py:class} SpeechToTextConfig
:canonical: agents.config.SpeechToTextConfig

Bases: {py:obj}`agents.config.ModelComponentConfig`

```{autodoc2-docstring} agents.config.SpeechToTextConfig
```

````{py:method} get_inference_params() -> typing.Dict
:canonical: agents.config.SpeechToTextConfig.get_inference_params

```{autodoc2-docstring} agents.config.SpeechToTextConfig.get_inference_params
```

````

`````

`````{py:class} TextToSpeechConfig
:canonical: agents.config.TextToSpeechConfig

Bases: {py:obj}`agents.config.ModelComponentConfig`

```{autodoc2-docstring} agents.config.TextToSpeechConfig
```

````{py:method} get_inference_params() -> typing.Dict
:canonical: agents.config.TextToSpeechConfig.get_inference_params

```{autodoc2-docstring} agents.config.TextToSpeechConfig.get_inference_params
```

````

`````

````{py:class} SemanticRouterConfig
:canonical: agents.config.SemanticRouterConfig

Bases: {py:obj}`agents.ros.BaseComponentConfig`

```{autodoc2-docstring} agents.config.SemanticRouterConfig
```

````

````{py:class} MapConfig
:canonical: agents.config.MapConfig

Bases: {py:obj}`agents.ros.BaseComponentConfig`

```{autodoc2-docstring} agents.config.MapConfig
```

````

````{py:class} VideoMessageMakerConfig
:canonical: agents.config.VideoMessageMakerConfig

Bases: {py:obj}`agents.ros.BaseComponentConfig`

```{autodoc2-docstring} agents.config.VideoMessageMakerConfig
```

````

`````{py:class} VisionConfig
:canonical: agents.config.VisionConfig

Bases: {py:obj}`agents.config.ModelComponentConfig`

```{autodoc2-docstring} agents.config.VisionConfig
```

````{py:method} get_inference_params() -> typing.Dict
:canonical: agents.config.VisionConfig.get_inference_params

```{autodoc2-docstring} agents.config.VisionConfig.get_inference_params
```

````

`````

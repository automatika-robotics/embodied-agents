# Creating a Custom Component

This guide walks through building a new EmbodiedAgents component from scratch.

## When to Subclass What

Choose your base class based on whether your component needs a model client:

| Base Class | Use When |
|---|---|
| `Component` | Your component performs pure data processing, transformations, or routing without calling an ML model. |
| `ModelComponent` | Your component wraps an ML model and needs inference via a `ModelClient`. |

Most custom components will subclass `ModelComponent`.

## Defining Allowed Inputs and Outputs

Every component must declare what topic types it accepts. These are set as instance attributes before the `super().__init__()` call:

```python
from agents.ros import SupportedType, String, Image, Audio

class MySummarizer(ModelComponent):
    def __init__(self, ...):
        self.allowed_inputs = {
            "Required": [String],          # Must have at least one String input
            "Optional": [Image],           # May optionally accept Image inputs
        }
        self.allowed_outputs = {
            "Required": [String],          # Must have at least one String output
        }
        super().__init__(...)
```

### Cardinality Rules

- Each entry in the `"Required"` list must have at least one matching topic in the provided inputs/outputs.
- A nested list like `[String, Audio]` means "at least one topic of type `String` **or** `Audio`."
- `"Optional"` entries are accepted but not enforced.
- Subtypes are matched: if `StreamingString` is a subclass of the allowed type, it passes validation.

## Implementing `_execution_step()`

This is the core logic of your component. For `ModelComponent` subclasses, you must also implement `_create_input()`, `_warmup()`, and `_handle_websocket_streaming()`.

```python
from abc import abstractmethod

class MySummarizer(ModelComponent):

    @abstractmethod
    def _execution_step(self, **kwargs):
        """Called each time the component is triggered."""
        ...

    @abstractmethod
    def _create_input(self, *args, **kwargs):
        """Assemble the inference input dict from callback data."""
        ...

    @abstractmethod
    def _warmup(self, *args, **kwargs):
        """Optional warmup call during configure phase."""
        ...

    @abstractmethod
    def _handle_websocket_streaming(self):
        """Handle streaming responses from WebSocket clients."""
        ...
```

For `Component` subclasses (no model client), you only need to implement `_execution_step()`.

## Configuration Class Pattern

Define a config class using `attrs`:

```python
from attrs import define, field
from agents.config import ModelComponentConfig
from agents.ros import base_validators

@define(kw_only=True)
class SummarizerConfig(ModelComponentConfig):
    """Configuration for the Summarizer component."""

    max_summary_length: int = field(default=200, validator=base_validators.gt(0))
    style: str = field(default="concise")
    temperature: float = field(default=0.5, validator=base_validators.gt(0.0))
    max_new_tokens: int = field(default=300, validator=base_validators.gt(0))

    def _get_inference_params(self):
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
        }
```

Key points:

- Always use `@define(kw_only=True)`.
- Extend `ModelComponentConfig` (which itself extends `BaseComponentConfig`).
- Implement `_get_inference_params()` to return the dict passed to the model at inference time.
- Use `base_validators` for field validation (`gt`, `in_range`, `in_`).

## Wiring the Trigger

The trigger determines when your component's `_execution_step()` fires. Set it in the constructor:

```python
# Trigger on a specific input topic
summarizer = MySummarizer(
    inputs=[text_in, context_in],
    outputs=[summary_out],
    model_client=my_client,
    trigger=text_in,       # fires when text_in receives a message
)

# Trigger on a timer (2 Hz)
summarizer = MySummarizer(
    ...,
    trigger=2.0,           # fires twice per second
)

# Trigger on an external event
from agents.ros import Event
my_event = Event(name="summarize_now")
summarizer = MySummarizer(
    ...,
    trigger=my_event,
)
```

When a `Topic` is used as trigger, it must be one of the component's inputs. Internally, the topic's callback is moved from `self.callbacks` to `self.trig_callbacks`, and `_execution_step()` is wired as a post-callback.

## Complete Skeleton: A "Summarizer" Component

Below is a complete, working skeleton for a component that takes a text input, sends it to an LLM for summarization, and publishes the result.

```python
from typing import Any, Dict, List, Optional, Sequence, Type, Union
from types import NoneType

from attrs import define, field
from agents.components.model_component import ModelComponent
from agents.clients.model_base import ModelClient
from agents.config import ModelComponentConfig
from agents.ros import (
    Topic,
    FixedInput,
    String,
    StreamingString,
    SupportedType,
    Event,
    base_validators,
)


# --- Config ---
@define(kw_only=True)
class SummarizerConfig(ModelComponentConfig):
    """Configuration for the Summarizer component."""

    max_summary_length: int = field(default=200, validator=base_validators.gt(0))
    temperature: float = field(default=0.5, validator=base_validators.gt(0.0))
    max_new_tokens: int = field(default=300, validator=base_validators.gt(0))

    def _get_inference_params(self) -> Dict:
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
        }


# --- Component ---
class Summarizer(ModelComponent):
    """A component that summarizes incoming text using an LLM."""

    def __init__(
        self,
        inputs: Optional[Sequence[Union[Topic, FixedInput]]] = None,
        outputs: Optional[Sequence[Topic]] = None,
        model_client: Optional[ModelClient] = None,
        config: Optional[SummarizerConfig] = None,
        trigger: Union[Topic, List[Topic], float, Event, NoneType] = 1.0,
        component_name: str = "summarizer",
        **kwargs,
    ):
        # Declare allowed I/O before super().__init__
        self.allowed_inputs = {
            "Required": [String],
        }
        self.allowed_outputs = {
            "Required": [String],
        }

        # Which output types this component can publish natively
        self.handled_outputs: List[Type[SupportedType]] = [String, StreamingString]

        if not config:
            config = SummarizerConfig()

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            model_client=model_client,
            config=config,
            trigger=trigger,
            component_name=component_name,
            **kwargs,
        )

    def _create_input(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Assemble inference input from the latest callback data."""
        # Read from trigger or regular callbacks
        text = None
        for cb in self.trig_callbacks.values():
            text = cb.get_output()

        if text is None:
            for cb in self.callbacks.values():
                text = cb.get_output()

        if text is None:
            self.get_logger().warning("No input text received yet")
            return None

        prompt = f"Summarize the following text in under {self.config.max_summary_length} words:\n\n{text}"

        return {
            "query": prompt,
            "images": [],
        }

    def _execution_step(self, **kwargs):
        """Main processing loop."""
        inference_input = self._create_input()
        if inference_input is None:
            return

        result = self._call_inference(inference_input)
        if result is None:
            return

        self._publish(result)

    def _warmup(self, *args, **kwargs):
        """Send a dummy request to warm up the model."""
        warmup_input = {"query": "Hello", "images": []}
        self._call_inference(warmup_input)

    def _handle_websocket_streaming(self) -> Optional[Any]:
        """Handle streaming responses (not used in this example)."""
        pass
```

### Usage

```python
from agents.clients.ollama import OllamaClient
from agents.models import OllamaModel
from agents.ros import Topic, Launcher

text_in = Topic(name="long_text", msg_type="String")
summary_out = Topic(name="summary", msg_type="String")

model = OllamaModel(name="summarizer_llm", checkpoint="llama3.2:3b")
client = OllamaClient(model)

summarizer = Summarizer(
    inputs=[text_in],
    outputs=[summary_out],
    model_client=client,
    trigger=text_in,
    config=SummarizerConfig(max_summary_length=100, temperature=0.3),
)

launcher = Launcher()
launcher.add_pkg(components=[summarizer])
launcher.bringup()
```

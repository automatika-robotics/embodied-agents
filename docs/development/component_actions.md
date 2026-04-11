# Component Actions

Component actions are methods on a component that can be invoked externally — by the event system, by the Cortex planner, or via ROS services. They are the primary way to expose discrete capabilities (e.g. "take a picture", "say something", "start tracking") beyond the continuous `_execution_step()` loop.

Read [Creating a Custom Component](./custom_component.md) and [Advanced Components](./advanced_component.md) first.

## Defining a Component Action

Decorate a method with `@component_action` and provide an OpenAI-style tool description:

```python
from agents.ros import component_action

class MyVisionComponent(ModelComponent):

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "take_picture",
                "description": "Capture a photo from a camera topic and save it to disk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic_name": {
                            "type": "string",
                            "description": "Name of the input topic to capture from.",
                        },
                    },
                    "required": ["topic_name"],
                },
            },
        }
    )
    def take_picture(self, topic_name: str, save_path: str = "~/pictures") -> bool:
        """Capture and save a single frame."""
        # ... implementation ...
        return True
```

### Key Points

- The `description` dict follows the [OpenAI function calling](https://platform.openai.com/docs/guides/function-calling) schema. This is what the Cortex planner sees when deciding which tools to call.
- The method name in your Python code must match the `"name"` inside the description.
- Return `bool` for success/failure actions, or `str` to return a text result (e.g. a description of what the component sees).
- Actions are executed via ROS services (`ExecuteMethod`), so they run on the component's own process and can access its internal state.

## Defining a Component Fallback

Use `@component_fallback` for methods intended as recovery actions (model switching, local fallback). These are also discoverable as tools but are specifically validated when used with `on_component_fail()` or `on_algorithm_fail()`:

```python
from agents.ros import component_fallback

class MyComponent(ModelComponent):

    @component_fallback(
        description={
            "type": "function",
            "function": {
                "name": "switch_to_backup",
                "description": "Switch to the backup model client.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    )
    def switch_to_backup(self) -> bool:
        # ... implementation ...
        return True
```

See [Model-Specific Fallbacks](./advanced_component.md#model-specific-fallbacks) for the built-in `fallback_to_local()` and `change_model_client()` methods.

## How Actions Are Discovered

When the Cortex component activates, it scans all managed components for methods decorated with `@component_action` or `@component_fallback`. Each discovered method is registered as an execution tool with the tool description from the decorator.

Tool names are namespaced as `{component_name}.{method_name}` (e.g. `vision.take_picture`, `tts.say`).

Lifecycle methods (`start`, `stop`, `restart`, `reconfigure`, `set_param`, `set_params`, `broadcast_status`) are filtered out — they are managed by the Monitor, not by the planner.

## Built-in Component Actions

| Component | Action | Description |
|---|---|---|
| **Vision** | `take_picture(topic_name, save_path)` | Capture a frame and save to disk |
| **Vision** | `record_video(topic_name, duration, save_path, fps)` | Record video for a duration |
| **Vision** | `track(label)` | Start ByteTrack tracking for a label (requires RoboML client + Tracking output) |
| **VLM** | `describe(topic_name, query)` | Capture a frame and describe it using the VLM |
| **TextToSpeech** | `say(text)` | Convert text to speech and play on device |
| **TextToSpeech** | `stop_playback()` | Stop current audio playback |
| **MapEncoding** | `add_point(layer, point)` | Add a labeled point to a map layer |

All `ModelComponent` subclasses also inherit:

| Action | Description |
|---|---|
| `fallback_to_local()` | Switch from remote client to built-in local model |
| `change_model_client(model_client_name)` | Hot-swap to a registered additional model client |

## Example: Custom Action on a Component

```python
from agents.components import ModelComponent
from agents.ros import component_action, Topic, Image


class SecurityCamera(ModelComponent):
    """A vision component that can arm/disarm monitoring."""

    def __init__(self, **kwargs):
        self._armed = False
        self.allowed_inputs = {"Required": [Image]}
        self.handled_outputs = []
        super().__init__(**kwargs)

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "arm",
                "description": "Arm the security camera to start monitoring for intruders.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    )
    def arm(self) -> bool:
        """Start monitoring."""
        self._armed = True
        self.get_logger().info("Security camera armed.")
        return True

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "disarm",
                "description": "Disarm the security camera to stop monitoring.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    )
    def disarm(self) -> bool:
        """Stop monitoring."""
        self._armed = False
        self.get_logger().info("Security camera disarmed.")
        return True

    def _execution_step(self, **kwargs):
        if not self._armed:
            return
        # ... run detection, check for intruders ...
```

When this component is managed by Cortex, the planner can call `security_camera.arm` or `security_camera.disarm` as part of a task plan.

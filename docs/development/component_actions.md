# Component Actions

Component actions are methods on a component that can be invoked externally — by the event system, by the Cortex planner, or via ROS services. They are the primary way to expose discrete capabilities (e.g. "take a picture", "say something", "start tracking") beyond the continuous `_execution_step()` loop.

Read [Creating a Custom Component](./custom_component.md) and [Advanced Components](./advanced_component.md) first.

## Defining a Component Action

Decorate a method with `@component_action` and provide an OpenAI-style tool description:

```python
from agents.ros import ActionReturnType, component_action

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
    def take_picture(
        self, topic_name: str, save_path: str = "~/pictures"
    ) -> ActionReturnType:
        """Capture and save a single frame."""
        frame = ...  # grab a frame from the topic's callback
        if frame is None:
            return False, f"No image received on '{topic_name}'"
        path = ...  # save the frame under save_path
        return True, f"Picture from '{topic_name}' saved to {path}"
```

### Key Points

- The `description` dict follows the [OpenAI function calling](https://platform.openai.com/docs/guides/function-calling) schema. This is what the Cortex planner sees when deciding which tools to call.
- The method name in your Python code must match the `"name"` inside the description.
- With `active=True`, a call made while the component is not active returns a failure without running the method.

## The Action Contract

Every action returns a `(success, message)` pair and is annotated with `ActionReturnType`, an alias of `Tuple[bool, str]` (that spelling is accepted too). The annotation is checked when the class is defined: a method decorated with `@component_action` or `@component_fallback` without it raises `TypeError` at import, so a component that breaks the contract fails before it is ever launched.

- `success` tells the caller whether the action did what it was asked.
- `message` carries the result on success: a confirmation, a text answer, or JSON when the result is structured (`MLLM.run_task` returns its summary this way). On failure it carries the reason.
- Report a failure you can explain by returning `False` with the reason rather than raising. An exception is still caught and reported as a failure, but its text is all the caller gets.
- At runtime, a return value that is not a `(bool, str)` pair is logged as a contract violation and treated as a failure.

The same contract applies wherever sugarcoat runs a method as an action: through the `ExecuteMethod` service, as a fallback, or as an `Action` wired to an event.

### What the Caller Receives

Actions are executed through the component's `ExecuteMethod` service, so they run in the component's own process and can access its internal state. On success the message is JSON-encoded into the response's `response_json`; on failure it is placed in `error_msg`.

Cortex reaches actions through the Monitor, which reads that response back into the `(success, message)` pair. The `LLM` component calls the service directly and decodes the response itself. Either way, the tool result the model sees is:

| Response | Tool result |
|---|---|
| success, non-empty message | the message itself |
| success, empty message | `<tool_name> executed successfully` |
| failure | `Error: <tool_name> failed with error: <reason>` |

Write messages for that reader: a failure reason should say what went wrong in terms the planner can act on.

## Defining a Component Fallback

Use `@component_fallback` for methods intended as recovery actions (model switching, local fallback). They follow the same contract and are also discoverable as tools. Only a component's own fallback methods are accepted by `on_component_fail()` or `on_algorithm_fail()`. Unlike an action, a fallback also runs while the component is inactive or still activating, since recovery happens before the component is healthy again:

```python
from agents.ros import ActionReturnType, component_fallback

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
    def switch_to_backup(self) -> ActionReturnType:
        if self.backup_client is None:
            return False, "No backup model client is configured"
        # ... swap the clients ...
        return True, "Switched to the backup model client"
```

A `True` success resets the component's health status. A failure is logged with its reason, and the next retry or the next fallback in the list takes over.

See [Model-Specific Fallbacks](./advanced_component.md#model-specific-fallbacks) for the built-in `fallback_to_local()` and `change_model_client()` methods.

## How Actions Are Discovered

Cortex does not scan components itself. The `Launcher` builds a `SystemActionRegistry`, sugarcoat's catalogue of everything the stack can be asked to do by name, from every component in the recipe, whatever process it runs in, and hands it to Cortex. When Cortex activates, it walks that registry and turns its entries into tools:

- A method decorated with `@component_action` or `@component_fallback` becomes a tool when its decorator carries a description. The description is used whole, so what the planner sees is exactly what the component author wrote. A method without a description is not offered.
- An action server becomes a `send_goal_to_<server_name>` tool and a service a `send_request_to_<service_name>` tool, see [Action Servers as Tools](#action-servers-as-tools).

Tool names for methods are namespaced as `{component_name}.{method_name}` (e.g. `vision.take_picture`, `tts.say`). Cortex keeps the registry reference behind every tool, `{component_name}/{name}`, and dispatches a call by what the entry is: a method runs through the Monitor's own resolver over the component's `ExecuteMethod` service, a goal goes to the action server, a request to the service.

Lifecycle methods (`start`, `stop`, `restart`, `reconfigure`, `set_param`, `set_params`, `broadcast_status`) are filtered out — they are managed by the Monitor, not by the planner. Cortex's own actions and the Monitor's methods are left out as well.

### Planning and Execution Phases

Cortex keeps two tool sets: the planner's, used while it builds a plan, and the executor's, used while it carries the plan out. The `phase` argument of `agents.ros.component_action` decides where an action is registered. The wrapper writes the phase into the description dict the decorator stores, beside `type` and `function`, which is how it reaches Cortex through the registry. Sugarcoat's own decorator has no such argument, and fallbacks are always execution tools.

| `phase` | Registered with | Use for |
|---|---|---|
| `ActionPhase.EXECUTION` (default) | executor | actions that change state (`say`, `start_episode`) |
| `ActionPhase.PLANNING` | planner | introspection the executor has no reason to call |
| `ActionPhase.BOTH` | both | retrieval the planner benefits from before planning and the executor may need at run time (`describe`, `locate`) |

```python
from agents.ros import ActionPhase, ActionReturnType, component_action

@component_action(description={...}, phase=ActionPhase.BOTH)
def locate(self, **kwargs) -> ActionReturnType: ...
```

## Action Servers as Tools

Cortex also exposes action servers as execution tools, named `send_goal_to_<component>_<server>` from the registry reference: the component's node name, then the server's name with that node name prefix removed and slashes replaced by underscores. A server `vla/run` on component `vla` gives `send_goal_to_vla_run`. The goal message's fields become the tool parameters. The registry lists two kinds of server:

- the main action server of every managed component running as `ComponentRunType.ACTION_SERVER` (e.g. `VLA`, `MoveIt`)
- any additional action servers a component reports through `get_ros_entrypoints()`.

Services become `send_request_to_<component>_<service>` tools the same way, both the main service of a component running as `ComponentRunType.SERVER` and the additional ones reported through `get_ros_entrypoints()`. Because the name carries the component, two components whose servers share a bare name get two tools.

A goal is dispatched asynchronously. The tool returns once the server accepts the goal, and Cortex keeps reporting the goal's status, latest feedback and result to the model while the plan continues.

### One Goal at a Time

A component's main action server runs one goal at a time. While a goal is ongoing, a new goal request is rejected: the running goal has to finish or be canceled first, through the action's own cancel request, the inherited `cancel_main_goal` component action, or the component's `<node_name>/cancel_main_action` service (`std_srvs/Trigger`). A goal counts as ongoing until `main_action_callback()` returns, not only until it reaches a terminal state, so a new goal never starts while the previous one is still cleaning up.

When Cortex sends a goal through a tool that still has a goal of its own running, it replaces that goal: it cancels it, waits for the server to return the result, and then sends the new one. The wait is bounded by the action client's `feedback_check_timeout`. If the goal has not returned by then, the tool reports that it could not be canceled and the new goal is not sent. Cortex never cancels a goal that another client started on its own: that rejection reaches the planner as the server being busy and names the component's `cancel_main_goal` tool, so the planner can stop that goal and send its own.

This puts one requirement on a component implementing `main_action_callback()`: check `goal_handle.is_cancel_requested` inside the loop, transition the goal with `goal_handle.canceled()`, and return promptly. A callback that keeps running after a cancel request blocks every new goal, including the one Cortex is waiting to send.

## Cortex's Own Tools

| Tool | Phase | Description |
|---|---|---|
| `inspect_component(component)` | planning | A component's topics, configuration, model clients and tools |
| `update_parameter(component, param_name, new_value)` | execution | Change one configuration parameter |
| `wait(duration)` | execution | Hold for a number of seconds. Not for waiting on a running goal, whose progress the planner is shown. Cut short if the task is cancelled |

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

Action server components also inherit `cancel_main_goal()`, which stops the goal their main action server is running.

## Example: Custom Action on a Component

```python
from agents.components import ModelComponent
from agents.ros import ActionReturnType, component_action, Topic, Image


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
    def arm(self) -> ActionReturnType:
        """Start monitoring."""
        self._armed = True
        self.get_logger().info("Security camera armed.")
        return True, "Security camera armed"

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
    def disarm(self) -> ActionReturnType:
        """Stop monitoring."""
        self._armed = False
        self.get_logger().info("Security camera disarmed.")
        return True, "Security camera disarmed"

    def _execution_step(self, **kwargs):
        if not self._armed:
            return
        # ... run detection, check for intruders ...
```

When this component is managed by Cortex, the planner can call `security_camera.arm` or `security_camera.disarm` as part of a task plan.

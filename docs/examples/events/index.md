# Events & Actions Overview

This section unlocks the true potential of **EmbodiedAgents**: creating systems that are **robust**, **reactive**, and **self-referential**.

While the Foundation recipes taught you how to build a static graph of components, the real world is dynamic. A truly intelligent physical agent must be able to **adapt** its behavior based on its environment and its own internal state. This is where the framework's **Event-Driven Architecture** shines.

At its core, _EmbodiedAgents_ inherits a powerful event-handling mechanism from its underlying base framework, Sugarcoat. To deeply understand the mechanics of this architecture, we highly recommend reading the core concepts of [Events](https://automatika-robotics.github.io/sugarcoat/design/events.html) and [Actions](https://automatika-robotics.github.io/sugarcoat/design/actions.html) in the Sugarcoat documentation.

## Building "Gödel Machines"

_EmbodiedAgents_ allows you to create agents that are self-aware and self-modifying, thus providing a framework for **Adaptive Intelligence** utilizing various AI models as building blocks of a larger system. The recipes in this section demonstrate how to break free from linear execution loops and embrace adaptive behaviors:

- **Event-Driven Execution**: Move beyond simple timed loops and input topic triggers. Learn to configure components that sleep until triggered by specific changes in the environment, such as a person entering a room or a specific keyword being detected.
- **Dynamic Reconfiguration**: Discover how an agent can modify its own structure at runtime. Imagine an agent that switches from a fast, low-latency model to a powerful reasoning model only when it encounters a complex problem, or one that changes its manipulation model based on visual inputs.
- **Robust Production Ready Agents**: Learn how components run as their own execution units thus their failure does not cascade to the rest of the system. Add fallback behaviors based on component failures to make them reconfigure or restart themselves so the overall system never fails.

These tutorials will guide you through building agents that don't just follow instructions, but understand and react to the nuance of their physical reality.

## Recipes

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {material-regular}`smart_toy;1.2em;sd-text-primary` Complete Agent, But Better
:link: multiprocessing
:link-type: doc

Transition your agent from a prototype to a production-ready system by running components in separate processes. This recipe demonstrates how to isolate failures so one crash doesn't stop the robot, and how to configure global fallback rules to automatically restart unhealthy components.
:::

:::{grid-item-card} {material-regular}`backup;1.2em;sd-text-primary` Runtime Fallbacks
:link: fallback
:link-type: doc

Build a self-healing agent that can handle API outages or connection drops. This tutorial teaches you to implement a "Plan B" strategy where the agent automatically swaps its primary cloud-based brain for a smaller, local backup model if the primary connection fails.
:::

:::{grid-item-card} {material-regular}`bolt;1.2em;sd-text-primary` Event Driven Triggering
:link: event_driven_description
:link-type: doc

Optimize your agent's compute resources by creating a "Reflex-Cognition" loop. Learn to use a lightweight vision detector to monitor the scene continuously (Reflex), and only trigger a heavy VLM (Cognition) to describe the scene when a specific event is detected.
:::
::::

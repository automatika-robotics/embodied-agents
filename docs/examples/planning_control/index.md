# Embodied Planning & Control Overview

Once you understand how to route data and maintain state using the Foundation recipes, it is time to give your agent true physical agency.

The recipes in this section move beyond simple tools and navigation. They focus on high-level cognitive planning and direct motor control, bridging the gap between reasoning about the world and physically interacting with it.

Here, you will learn how to orchestrate advanced models to manipulate the physical world:

- **Task Decomposition**: Use Multimodal LLMs to break down abstract user goals into sequences of concrete, executable actions.
- **End-to-End Control**: Deploy Vision Language Action (VLA) models to translate camera pixels and language instructions directly into robot joint trajectories.
- **Closed-Loop Verification**: Combine perception, physical action, and events to create self-correcting agents that know exactly when a physical task is complete.

## Recipes

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {material-regular}`psychology;1.2em;sd-text-primary` Utilizing Multimodal Planning
:link: planning_model
:link-type: doc

Configure a specific **VLM** to act as a high-level planner, decomposing complex user instructions into a sequence of executable low-level actions.
:::

:::{grid-item-card} {material-regular}`precision_manufacturing;1.2em;sd-text-primary` Robot Manipulation
:link: vla
:link-type: doc

Control physical actuators using end-to-end Vision Language Action (VLA) models. This recipe demonstrates how to use the VLA component and LeRobot policies to map visual inputs directly to robot joint commands.
:::

:::{grid-item-card} {material-regular}`bolt;1.2em;sd-text-primary` Event Driven Robot Manipulation
:link: vla_with_event
:link-type: doc

Build a closed-loop agent where a VLM acts as a referee for a VLA. This recipe demonstrates how to use Events to automatically stop physical actions based on visual verification of task completion.
:::
::::

from launch.action import Action as ROSLaunchAction
from launch.actions import RegisterEventHandler

from ros_sugar import Launcher as BaseLauncher
from ros_sugar.core.component import BaseComponent
from ros_sugar.core.monitor import Monitor
from ros_sugar.core.event import OnInternalEvent
from ros_sugar.core.action import Action
from ros_sugar.launch.launch_actions import ComponentLaunchAction
from typing import List, Optional, Dict, Union


class Launcher(BaseLauncher):
    def __init__(
        self, namespace="", config_file=None, activation_timeout=None, robot_plugin=None
    ):
        super().__init__(namespace, config_file, activation_timeout, robot_plugin)

    def _setup_additional_internal_actions(
        self,
        additional_internal_actions: Dict[str, Union[ROSLaunchAction, Action]],
    ) -> None:
        # Add event handling actions
        entities_dict: Dict = {}

        for event_name, action in additional_internal_actions.items():
            entities_dict[event_name] = []
            if isinstance(action, ROSLaunchAction):
                entities_dict[event_name].append(action)

            # Check action type
            elif action._is_lifecycle_action:
                # Re-parse action for component related actions
                entities = self._get_action_launch_entity(action)
                if isinstance(entities, list):
                    entities_dict[event_name].extend(entities)
                else:
                    entities_dict[event_name].append(entities)

            elif isinstance(action, Action) and action.parent_component:
                continue  # skip, as it will be added directly in the cortex monitor configuration

            # If the action is not related to a component -> add opaque executable to launch
            else:
                entities_dict[event_name].append(
                    action.launch_action(monitor_node=self.monitor_node)
                )

            # Register a new internal event handler
            internal_events_handler = RegisterEventHandler(
                OnInternalEvent(
                    internal_event_name=event_name,
                    entities=entities_dict[event_name],
                )
            )
            self._description.add_action(internal_events_handler)

    def _init_monitor_node(
        self,
        components_names: List[str],
        services_components: List[BaseComponent],
        action_components: List[BaseComponent],
        all_components_to_activate_on_start: List[str],
    ) -> None:
        """Override to replace the classic Monitor by the Cortex component if it is configured in the recipe. The Cortex component will then be responsible to monitor the components and trigger events/actions based on the recipe configuration, but also to provide a centralized place to have an overview of the state of the agent and its components."""
        from .components.cortex import Cortex

        cortex_monitor: Optional[Cortex] = None
        for component in self._components:
            if isinstance(component, Cortex):
                cortex_monitor = component
                break
        if cortex_monitor:
            # remove the Cortex component from the lists of components to monitor, as it will be the monitor itself
            self._components.remove(cortex_monitor)
            components_names.remove(cortex_monitor.node_name)
            if cortex_monitor in services_components:
                services_components.remove(cortex_monitor)
            if cortex_monitor in action_components:
                action_components.remove(cortex_monitor)
            if cortex_monitor.node_name in all_components_to_activate_on_start:
                all_components_to_activate_on_start.remove(cortex_monitor.node_name)
            self.monitor_node = cortex_monitor
            self.monitor_node._init_internal_monitor(
                components_names=components_names,
                components=self._components,
                events_actions=self._monitor_events_actions,
                events_to_emit=self._internal_events,
                services_components=services_components,
                action_servers_components=action_components,
                activate_on_start=all_components_to_activate_on_start,
                activation_timeout=self._components_activation_timeout,
            )
            # Add any additional internal actions related to the monitor (e.g. events handling actions)
            self._setup_additional_internal_actions(
                self.monitor_node._additional_internal_actions
            )
        else:
            self.monitor_node = Monitor(
                components_names=components_names,
                events_actions=self._monitor_events_actions,
                events_to_emit=self._internal_events,
                services_components=services_components,
                action_servers_components=action_components,
                activate_on_start=all_components_to_activate_on_start,
                activation_timeout=self._components_activation_timeout,
            )

        monitor_action = ComponentLaunchAction(
            node=self.monitor_node,
            namespace=self._namespace,
            name=self.monitor_node.node_name,
        )
        self._description.add_action(monitor_action)

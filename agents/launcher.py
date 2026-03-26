from ros_sugar import Launcher as BaseLauncher
from ros_sugar.core.component import BaseComponent
from ros_sugar.core.monitor import Monitor
from ros_sugar.launch.launch_actions import ComponentLaunchAction
from typing import List, Optional


class Launcher(BaseLauncher):
    def __init__(
        self, namespace="", config_file=None, activation_timeout=None, robot_plugin=None
    ):
        super().__init__(namespace, config_file, activation_timeout, robot_plugin)

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
                all_components=self._components,
                events_actions=self._monitor_events_actions,
                events_to_emit=self._internal_events,
                services_components=services_components,
                action_servers_components=action_components,
                activate_on_start=all_components_to_activate_on_start,
                activation_timeout=self._components_activation_timeout,
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

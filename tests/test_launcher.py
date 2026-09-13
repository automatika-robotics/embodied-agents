"""Tests for the Launcher's monitor setup — requires the ROS launch system."""

from unittest.mock import MagicMock, patch

from agents.launcher import Launcher
from agents.components.cortex import Cortex


def _launcher(components):
    """A Launcher carrying only what _init_monitor_node reads, without the
    launch description a real one builds"""
    launcher = Launcher.__new__(Launcher)
    launcher._components = components
    launcher._monitor_events_actions = {}
    launcher._internal_events = []
    launcher._components_activation_timeout = 1.0
    launcher._action_registry = MagicMock(name="registry")
    # _robot_plugin is a property reading the attached plugins
    launcher._plugins = {}
    launcher._namespace = ""
    launcher._description = MagicMock()
    launcher._setup_additional_internal_actions = MagicMock()
    return launcher


def _install_monitor(launcher, names):
    launcher._init_monitor_node(
        components_names=names,
        services_components=[],
        action_components=[],
        all_components_to_activate_on_start=[],
    )


class TestTheMonitorGetsTheActionRegistry:
    """The registry of what the stack can be asked to do is built by the base
    Launcher and handed over as an attribute. Both monitors this launcher can
    install must receive it"""

    def test_a_plain_monitor(self):
        launcher = _launcher([])

        with (
            patch("agents.launcher.Monitor") as monitor,
            patch("agents.launcher.ComponentLaunchAction"),
        ):
            _install_monitor(launcher, [])

        assert monitor.call_args.kwargs["action_registry"] is launcher._action_registry

    def test_cortex_standing_in_for_it(self):
        cortex = MagicMock(spec=Cortex)
        cortex.node_name = "cortex"
        cortex._additional_internal_actions = {}
        launcher = _launcher([cortex])

        with patch("agents.launcher.ComponentLaunchAction"):
            _install_monitor(launcher, ["cortex"])

        assert launcher.monitor_node is cortex
        kwargs = cortex._init_internal_monitor.call_args.kwargs
        assert kwargs["action_registry"] is launcher._action_registry

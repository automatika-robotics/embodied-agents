"""Tests for the Launcher's monitor setup — requires the ROS launch system."""

from unittest.mock import MagicMock, patch

from ros_sugar.robot import PluginMetadata, RobotPlugin, SensorPlugin

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


class _Robot(RobotPlugin):
    def __init__(self):
        self.metadata = PluginMetadata(name="Robot")


class _Camera(SensorPlugin):
    def __init__(self):
        self.metadata = PluginMetadata(name="Camera")


def _cortex_launcher(*plugins):
    cortex = MagicMock(spec=Cortex)
    cortex.node_name = "cortex"
    cortex._additional_internal_actions = {}
    launcher = _launcher([cortex])
    launcher._plugins = {plugin.id: plugin for plugin in plugins}
    with patch("agents.launcher.ComponentLaunchAction"):
        _install_monitor(launcher, ["cortex"])
    return cortex


class TestCortexGetsEveryPlugin:
    """Sensor plugins carry actions too, such as aiming a camera, so Cortex is
    given every attached plugin, not only the robot's"""

    def test_robot_and_sensor_plugins(self):
        robot, camera = _Robot(), _Camera(id="front_cam")

        cortex = _cortex_launcher(robot, camera)

        registered = [c.args[0] for c in cortex.add_plugin_actions.call_args_list]
        assert registered == [robot, camera]
        cortex.set_robot_description.assert_called_once_with(robot)
        cortex.set_sensor_descriptions.assert_called_once_with([camera])

    def test_a_recipe_with_sensors_and_no_robot(self):
        camera = _Camera(id="front_cam")

        cortex = _cortex_launcher(camera)

        cortex.add_plugin_actions.assert_called_once_with(camera)
        cortex.set_robot_description.assert_not_called()
        cortex.set_sensor_descriptions.assert_called_once_with([camera])

    def test_a_recipe_with_only_a_robot_describes_no_sensors(self):
        robot = _Robot()

        cortex = _cortex_launcher(robot)

        cortex.add_plugin_actions.assert_called_once_with(robot)
        cortex.set_sensor_descriptions.assert_not_called()

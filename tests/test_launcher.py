"""Tests for the Launcher's monitor setup — requires the ROS launch system."""

from unittest.mock import MagicMock, patch

from ros_sugar.core import BaseComponent
from ros_sugar.robot import PluginMetadata, RobotPlugin, SensorPlugin

from agents.config import CortexConfig
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
    launcher._pkg_executable = {}
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
    Launcher. A plain Monitor receives it directly. Cortex is built without
    it, because the registry built so far still lists Cortex as a component,
    and the base Launcher rebuilds and hands it over once the override
    returns"""

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
        assert "action_registry" not in kwargs

    def test_the_rebuilt_registry_leaves_cortex_out(
        self, rclpy_init, mock_model_client
    ):
        """The handover the base Launcher does after the override: the registry
        is built from the components with Cortex removed, so Cortex's own
        methods are never listed as something the stack can be asked to do"""
        cortex = Cortex(
            actions=[],
            model_client=mock_model_client,
            config=CortexConfig(),
            component_name="test_launcher_cortex",
        )
        worker = BaseComponent(component_name="test_launcher_worker")
        launcher = _launcher([cortex, worker])

        with patch("agents.launcher.ComponentLaunchAction"):
            _install_monitor(launcher, [cortex.node_name, worker.node_name])
            launcher._hand_registry_to_monitor()

        assert cortex._registry_given
        owners = cortex._action_registry.owners()
        assert worker.node_name in owners
        assert "monitor" in owners
        assert cortex.node_name not in owners


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
    """Cortex is told about every attached plugin for its planning prompt: the
    robot's identity and the sensors. Their actions reach it through the
    action registry, not through the launcher"""

    def test_robot_and_sensor_plugins(self):
        robot, camera = _Robot(), _Camera(id="front_cam")

        cortex = _cortex_launcher(robot, camera)

        cortex.set_robot_description.assert_called_once_with(robot)
        cortex.set_sensor_descriptions.assert_called_once_with([camera])

    def test_a_recipe_with_sensors_and_no_robot(self):
        camera = _Camera(id="front_cam")

        cortex = _cortex_launcher(camera)

        cortex.set_robot_description.assert_not_called()
        cortex.set_sensor_descriptions.assert_called_once_with([camera])

    def test_a_recipe_with_only_a_robot_describes_no_sensors(self):
        robot = _Robot()

        cortex = _cortex_launcher(robot)

        cortex.set_sensor_descriptions.assert_not_called()

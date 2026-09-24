"""Tests for the base Component's trigger partition timing — requires rclpy.

Trigger inputs must stay in `callbacks` until activation, because a robot
plugin adapts the callbacks dict at activation time: partitioning at
construction made a plugin-backed trigger invisible to adaptation (a dead
ROS subscription), and kept the pre-adaptation callback object in
`trig_callbacks` even when adaptation replaced it.
"""

from unittest.mock import MagicMock

import pytest

from agents.components.component_base import Component
from agents.ros import Image, String, Topic


class _Probe(Component):
    """Smallest concrete component: a text trigger, an image, nothing done."""

    def __init__(self, **kwargs):
        self.allowed_inputs = {"Required": [String], "Optional": [Image]}
        super().__init__(**kwargs)

    def _execution_step(self, *args, **kwargs):
        pass


@pytest.fixture
def probe(rclpy_init):
    comp = _Probe(
        inputs=[
            Topic(name="trig_in", msg_type="String"),
            Topic(name="side_in", msg_type="Image"),
        ],
        trigger=Topic(name="trig_in", msg_type="String"),
        component_name="test_probe",
    )
    comp.get_logger = MagicMock()
    return comp


class TestTriggerPartitionTiming:
    def test_construction_records_triggers_but_keeps_the_callbacks(self, probe):
        assert probe._trigger_topic_names == ["trig_in"]
        assert "trig_in" in probe.callbacks
        assert not hasattr(probe, "trig_callbacks")

    def test_an_unknown_trigger_still_raises_at_construction(self, rclpy_init):
        with pytest.raises(TypeError, match="trigger"):
            _Probe(
                inputs=[Topic(name="text_in", msg_type="String")],
                trigger=Topic(name="nowhere", msg_type="String"),
                component_name="test_probe_bad",
            )

    def test_init_variables_partitions_the_adapted_callbacks(self, probe):
        # a plugin adapted the trigger's callback before init_variables ran
        adapted = MagicMock()
        adapted.input_topic = probe.callbacks["trig_in"].input_topic
        probe.callbacks["trig_in"] = adapted

        probe.init_variables()

        assert probe.trig_callbacks["trig_in"] is adapted
        assert "trig_in" not in probe.callbacks
        assert "side_in" in probe.callbacks

    def test_init_variables_is_idempotent_across_reactivations(self, probe):
        probe.init_variables()
        first = probe.trig_callbacks["trig_in"]
        # deactivate/activate cycle: the names are no longer in callbacks,
        # and the trigger must survive rather than be wiped
        probe.init_variables()
        assert probe.trig_callbacks["trig_in"] is first
        probe.get_logger().error.assert_not_called()

    def test_a_vanished_trigger_is_reported_not_crashed(self, probe):
        del probe.callbacks["trig_in"]
        probe.init_variables()
        assert "trig_in" not in getattr(probe, "trig_callbacks", {})
        assert probe.get_logger().error.called

    def test_plugin_fed_inputs_get_no_ros_subscription(self, probe):
        probe.init_variables()
        probe._external_topics = {"side_in"}
        probe._add_ros_subscriber = MagicMock()

        probe.create_all_subscribers()

        subscribed = [
            call[0][0].input_topic.name
            for call in probe._add_ros_subscriber.call_args_list
        ]
        assert "side_in" not in subscribed
        assert "trig_in" in subscribed

    def test_declared_frames_are_wired_for_every_input(self, probe):
        """transform_input_to only takes effect if the subscriber pass hands
        each callback its transform resolver, plugin-fed inputs included."""
        probe.init_variables()
        probe.transform_input_to("side_in", "base_link", static_tf=True)
        probe.transform_input_to("trig_in", "base_link")
        probe._external_topics = {"side_in"}
        probe._add_ros_subscriber = MagicMock()

        probe.create_all_subscribers()

        assert probe.callbacks["side_in"]._transform_provider is not None
        assert probe.trig_callbacks["trig_in"]._transform_provider is not None

    def test_replacing_a_trigger_before_activation_renames_the_bookkeeping(
        self, probe
    ):
        """Pre-activation the trigger is a plain callback, so the parent
        replaces it there — and the partition must follow the rename."""
        error = probe._replace_input_topic("trig_in", "elsewhere", "String")

        assert error is None
        assert "elsewhere" in probe.callbacks and "trig_in" not in probe.callbacks
        assert probe._trigger_topic_names == ["elsewhere"]
        probe.init_variables()
        assert "elsewhere" in probe.trig_callbacks

    def test_replacing_an_unknown_topic_reports_instead_of_crashing(self, probe):
        # trig_callbacks does not exist before activation: adaptation calling
        # this must get an answer, not an AttributeError
        error = probe._replace_input_topic("nope", "elsewhere", "String")
        assert error and "not found" in error
        # and the same answer once the partition exists
        probe.init_variables()
        assert "not found" in probe._replace_input_topic("nope", "elsewhere", "String")

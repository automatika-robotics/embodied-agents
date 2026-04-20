"""Tests for the Memory component — requires rclpy and the ``emem`` package.

These tests exercise the Memory component's construction, layer routing,
and write path without standing up a real eMEM backend: the
``SpatioTemporalMemory`` instance on the component is replaced with a
``MagicMock`` so we can assert exactly how observations get dispatched
between ``add`` (perception) and ``add_body_state`` (internal state).
"""

from unittest.mock import MagicMock

import pytest

from agents.config import MemoryConfig
from agents.ros import MemLayer, MapLayer, Topic
from agents.components.memory import Memory


def _prep_for_inspect(component):
    """Minimal patching so ``inspect_component`` can run on a component
    that has not gone through the full ROS lifecycle. Memory doesn't
    have an inference_params path, so we don't reuse the shared helper
    in conftest (which assumes LLM components)."""
    component.get_logger = MagicMock()
    component.health_status = MagicMock()
    return component


@pytest.fixture
def odom_topic():
    return Topic(name="odom", msg_type="Odometry")


@pytest.fixture
def detections_topic():
    return Topic(name="detections", msg_type="String")


@pytest.fixture
def scene_topic():
    return Topic(name="scene", msg_type="String")


@pytest.fixture
def battery_topic():
    return Topic(name="battery", msg_type="String")


@pytest.fixture
def cpu_temp_topic():
    return Topic(name="cpu_temp", msg_type="String")


def _build_memory(layers, position, component_name, tmp_path):
    """Construct a Memory component with a temp db path."""
    return Memory(
        layers=layers,
        position=position,
        config=MemoryConfig(db_path=str(tmp_path / f"{component_name}.db")),
        component_name=component_name,
    )


def _stub_memory_backend(component):
    """Replace component.memory with a MagicMock so we can assert calls."""
    component.memory = MagicMock()
    return component.memory


def _stub_callback(component, topic_name, value):
    """Replace a callback with a MagicMock whose ``get_output`` returns ``value``.

    ``_get_ui_content`` is also stubbed so we can assert the component
    does NOT call it (the current design uses ``get_output`` only).
    """
    cb = MagicMock()
    cb.get_output = MagicMock(return_value=value)
    cb._get_ui_content = MagicMock(return_value=value)
    component.callbacks[topic_name] = cb
    return cb


class TestMemLayerAlias:
    def test_map_layer_is_mem_layer(self):
        assert MapLayer is MemLayer

    def test_is_internal_state_defaults_false(self, detections_topic):
        layer = MemLayer(subscribes_to=detections_topic)
        assert layer.is_internal_state is False

    def test_is_internal_state_can_be_set(self, battery_topic):
        layer = MemLayer(subscribes_to=battery_topic, is_internal_state=True)
        assert layer.is_internal_state is True

    def test_is_internal_state_roundtrips_through_json(self, battery_topic):
        import json

        layer = MemLayer(subscribes_to=battery_topic, is_internal_state=True)
        data = json.loads(layer.to_json())
        assert data["is_internal_state"] is True
        rehydrated = MapLayer(**data)
        assert rehydrated.is_internal_state is True


class TestMemoryConstruction:
    def test_construct_with_perception_layer(
        self, rclpy_init, odom_topic, detections_topic, tmp_path
    ):
        layer = MemLayer(subscribes_to=detections_topic)
        comp = _build_memory([layer], odom_topic, "mem_perception", tmp_path)
        assert "detections" in comp.layers_dict
        assert comp.layers_dict["detections"].is_internal_state is False

    def test_construct_with_internal_state_layer(
        self, rclpy_init, odom_topic, battery_topic, tmp_path
    ):
        layer = MemLayer(subscribes_to=battery_topic, is_internal_state=True)
        comp = _build_memory([layer], odom_topic, "mem_internal", tmp_path)
        assert comp.layers_dict["battery"].is_internal_state is True

    def test_construct_with_mixed_layers(
        self,
        rclpy_init,
        odom_topic,
        detections_topic,
        battery_topic,
        tmp_path,
    ):
        layers = [
            MemLayer(subscribes_to=detections_topic),
            MemLayer(subscribes_to=battery_topic, is_internal_state=True),
        ]
        comp = _build_memory(layers, odom_topic, "mem_mixed", tmp_path)
        assert set(comp.layers_dict.keys()) == {"detections", "battery"}
        assert comp.layers_dict["detections"].is_internal_state is False
        assert comp.layers_dict["battery"].is_internal_state is True

    def test_callbacks_created_for_layers_and_position(
        self,
        rclpy_init,
        odom_topic,
        detections_topic,
        battery_topic,
        tmp_path,
    ):
        layers = [
            MemLayer(subscribes_to=detections_topic),
            MemLayer(subscribes_to=battery_topic, is_internal_state=True),
        ]
        comp = _build_memory(layers, odom_topic, "mem_cb", tmp_path)
        assert set(comp.callbacks.keys()) == {"detections", "battery", "odom"}

    def test_perception_layer_with_disallowed_type_raises(
        self, rclpy_init, odom_topic, tmp_path
    ):
        """A perception-layer topic whose msg_type is outside the allowed
        set (e.g. Image) must be rejected at configuration time so mis-wired
        topics fail loudly instead of silently blowing up in the embedder."""
        image_topic = Topic(name="camera", msg_type="Image")
        with pytest.raises(TypeError):
            _build_memory(
                [MemLayer(subscribes_to=image_topic)],
                odom_topic,
                "mem_bad_type",
                tmp_path,
            )

    def test_internal_state_layer_skips_type_validation(
        self, rclpy_init, odom_topic, tmp_path
    ):
        """Internal-state layers are not validated — their message types
        come from robot plugins and the component trusts the plugin's
        callback contract."""
        # Image is not in allowed_inputs but flagging the layer as
        # internal-state must bypass the validator.
        image_topic = Topic(name="thermal_camera", msg_type="Image")
        comp = _build_memory(
            [MemLayer(subscribes_to=image_topic, is_internal_state=True)],
            odom_topic,
            "mem_plugin_bypass",
            tmp_path,
        )
        assert "thermal_camera" in comp.layers_dict
        assert comp.layers_dict["thermal_camera"].is_internal_state is True


class TestStoreLayers:
    def test_perception_layer_routes_to_add(
        self, rclpy_init, odom_topic, detections_topic, tmp_path
    ):
        comp = _build_memory(
            [MemLayer(subscribes_to=detections_topic)],
            odom_topic,
            "mem_route_add",
            tmp_path,
        )
        backend = _stub_memory_backend(comp)
        _stub_callback(comp, "detections", "chair, table")

        comp._store_layers(position=[1.0, 2.0, 0.0], time_stamp=10)

        backend.add.assert_called_once()
        kwargs = backend.add.call_args.kwargs
        assert kwargs["text"] == "chair, table"
        assert kwargs["layer_name"] == "detections"
        assert kwargs["x"] == 1.0
        assert kwargs["y"] == 2.0
        assert kwargs["z"] == 0.0
        assert kwargs["timestamp"] == 10.0
        backend.add_body_state.assert_not_called()

    def test_internal_state_layer_routes_to_add_body_state(
        self, rclpy_init, odom_topic, battery_topic, tmp_path
    ):
        comp = _build_memory(
            [MemLayer(subscribes_to=battery_topic, is_internal_state=True)],
            odom_topic,
            "mem_route_body",
            tmp_path,
        )
        backend = _stub_memory_backend(comp)
        _stub_callback(comp, "battery", "battery: 42%")

        comp._store_layers(position=[5.0, 5.0, 0.0], time_stamp=20)

        backend.add_body_state.assert_called_once()
        kwargs = backend.add_body_state.call_args.kwargs
        assert kwargs["text"] == "battery: 42%"
        assert kwargs["layer_name"] == "battery"
        assert kwargs["x"] == 5.0
        assert kwargs["y"] == 5.0
        assert kwargs["z"] == 0.0
        assert kwargs["timestamp"] == 20.0
        backend.add.assert_not_called()

    def test_mixed_layers_route_independently(
        self,
        rclpy_init,
        odom_topic,
        detections_topic,
        battery_topic,
        tmp_path,
    ):
        comp = _build_memory(
            [
                MemLayer(subscribes_to=detections_topic),
                MemLayer(subscribes_to=battery_topic, is_internal_state=True),
            ],
            odom_topic,
            "mem_mixed_route",
            tmp_path,
        )
        backend = _stub_memory_backend(comp)
        _stub_callback(comp, "detections", "chair, table")
        _stub_callback(comp, "battery", "battery: 42%")

        comp._store_layers(position=[7.0, 8.0, 0.1], time_stamp=30)

        backend.add.assert_called_once()
        assert backend.add.call_args.kwargs["layer_name"] == "detections"
        backend.add_body_state.assert_called_once()
        assert backend.add_body_state.call_args.kwargs["layer_name"] == "battery"

        # Both calls share the same position so spatial-interoceptive
        # associations work downstream.
        assert backend.add.call_args.kwargs["x"] == 7.0
        assert backend.add_body_state.call_args.kwargs["x"] == 7.0
        assert backend.add.call_args.kwargs["z"] == 0.1
        assert backend.add_body_state.call_args.kwargs["z"] == 0.1

    def test_store_layers_uses_get_output_not_ui_content(
        self, rclpy_init, odom_topic, detections_topic, tmp_path
    ):
        """_store_layers must read text through get_output. _get_ui_content
        is intended for UI rendering (which may produce annotated JPEGs
        for image-bearing callbacks) and must not be on the memory path."""
        comp = _build_memory(
            [MemLayer(subscribes_to=detections_topic)],
            odom_topic,
            "mem_text_source",
            tmp_path,
        )
        _stub_memory_backend(comp)

        cb = MagicMock()
        cb.get_output = MagicMock(return_value="from_get_output")
        cb._get_ui_content = MagicMock(return_value="from_ui_content")
        comp.callbacks["detections"] = cb

        comp._store_layers(position=[0.0, 0.0, 0.0], time_stamp=1)

        cb.get_output.assert_called_once()
        cb._get_ui_content.assert_not_called()
        assert comp.memory.add.call_args.kwargs["text"] == "from_get_output"

    def test_store_layers_stringifies_non_string_output(
        self, rclpy_init, odom_topic, battery_topic, tmp_path
    ):
        """If a plugin callback for an internal-state topic returns
        something that isn't a string, the component should coerce it
        via ``str(...)`` as a defensive fallback rather than crashing
        later in the embedding path."""
        comp = _build_memory(
            [MemLayer(subscribes_to=battery_topic, is_internal_state=True)],
            odom_topic,
            "mem_stringify",
            tmp_path,
        )
        backend = _stub_memory_backend(comp)
        cb = MagicMock()
        cb.get_output = MagicMock(return_value=42)  # non-string
        comp.callbacks["battery"] = cb

        comp._store_layers(position=[0.0, 0.0, 0.0], time_stamp=0)

        backend.add_body_state.assert_called_once()
        assert backend.add_body_state.call_args.kwargs["text"] == "42"

    def test_empty_callback_output_is_skipped(
        self, rclpy_init, odom_topic, detections_topic, tmp_path
    ):
        comp = _build_memory(
            [MemLayer(subscribes_to=detections_topic)],
            odom_topic,
            "mem_empty",
            tmp_path,
        )
        backend = _stub_memory_backend(comp)
        _stub_callback(comp, "detections", None)

        comp._store_layers(position=[0.0, 0.0, 0.0], time_stamp=0)

        backend.add.assert_not_called()
        backend.add_body_state.assert_not_called()

    def test_position_without_z_defaults_to_zero(
        self, rclpy_init, odom_topic, detections_topic, tmp_path
    ):
        comp = _build_memory(
            [MemLayer(subscribes_to=detections_topic)],
            odom_topic,
            "mem_2d",
            tmp_path,
        )
        backend = _stub_memory_backend(comp)
        _stub_callback(comp, "detections", "something")

        comp._store_layers(position=[1.5, 2.5], time_stamp=0)

        assert backend.add.call_args.kwargs["z"] == 0.0


class TestInspectComponent:
    def test_perception_only(self, rclpy_init, odom_topic, detections_topic, tmp_path):
        comp = _build_memory(
            [MemLayer(subscribes_to=detections_topic)],
            odom_topic,
            "mem_inspect_perception",
            tmp_path,
        )
        _prep_for_inspect(comp)

        result = comp.inspect_component()
        assert "Perception layers" in result
        assert "detections" in result
        assert "Internal-state layers" not in result

    def test_internal_state_only(self, rclpy_init, odom_topic, battery_topic, tmp_path):
        comp = _build_memory(
            [MemLayer(subscribes_to=battery_topic, is_internal_state=True)],
            odom_topic,
            "mem_inspect_internal",
            tmp_path,
        )
        _prep_for_inspect(comp)

        result = comp.inspect_component()
        assert "Internal-state layers" in result
        assert "battery" in result
        assert "body_status" in result
        assert "Perception layers: none configured" in result

    def test_mixed_layers_both_sections(
        self,
        rclpy_init,
        odom_topic,
        detections_topic,
        battery_topic,
        cpu_temp_topic,
        tmp_path,
    ):
        comp = _build_memory(
            [
                MemLayer(subscribes_to=detections_topic),
                MemLayer(subscribes_to=battery_topic, is_internal_state=True),
                MemLayer(subscribes_to=cpu_temp_topic, is_internal_state=True),
            ],
            odom_topic,
            "mem_inspect_mixed",
            tmp_path,
        )
        _prep_for_inspect(comp)

        result = comp.inspect_component()
        assert "Perception layers" in result
        assert "Internal-state layers" in result
        assert "detections" in result
        assert "battery" in result
        assert "cpu_temp" in result
        # Perception section must appear before internal-state section.
        assert result.index("Perception layers") < result.index("Internal-state layers")

    def test_inspect_includes_position_topic(
        self, rclpy_init, odom_topic, detections_topic, tmp_path
    ):
        comp = _build_memory(
            [MemLayer(subscribes_to=detections_topic)],
            odom_topic,
            "mem_inspect_pos",
            tmp_path,
        )
        _prep_for_inspect(comp)

        result = comp.inspect_component()
        assert "odom" in result

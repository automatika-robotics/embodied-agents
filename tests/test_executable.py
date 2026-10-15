"""Tests for the launch executable (scripts/executable), which rebuilds a
component in its own process from the arguments the launcher serialized"""

import importlib
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from ros_sugar.io import supported_types

from agents.ros import Event, Topic

EXECUTABLE = Path(__file__).parents[1] / "scripts" / "executable"

# A type of another package, defined in a module that does not register it,
# as kompass defines MissionStatus in kompass.data_types and registers it elsewhere
EXTERNAL_TYPES_MODULE = '''
from std_msgs.msg import UInt8
from ros_sugar.supported_types import SupportedType


class ExternalStatus(SupportedType):
    _ros_type = UInt8
'''


@pytest.fixture(scope="module")
def executable():
    """The executable, loaded as a module without running its main"""
    loader = importlib.machinery.SourceFileLoader("agents_executable", str(EXECUTABLE))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.fixture
def fresh_registry(monkeypatch):
    """Give each test its own copy of the global type registry. Returns a
    function that resets it to what it was, as a new process starts with"""
    initial = dict(supported_types._additional_types)

    def reset():
        monkeypatch.setattr(supported_types, "_additional_types", dict(initial))

    reset()
    return reset


@pytest.fixture
def external_type(tmp_path, monkeypatch, fresh_registry):
    """The module path of the external type, as the launcher passes it"""
    (tmp_path / "external_types.py").write_text(EXTERNAL_TYPES_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "external_types.ExternalStatus"
    sys.modules.pop("external_types", None)


@pytest.fixture
def recipe_args(external_type, fresh_registry):
    """What the recipe process serializes for a component process: the
    registered types, and an event and an input topic on the external type"""
    module = importlib.import_module("external_types")
    supported_types.add_additional_datatypes([module.ExternalStatus])
    status = Topic(name="status", msg_type="ExternalStatus")
    event = Event(status.msg.data == 3)
    args = {
        "additional_types": json.dumps([external_type]),
        "event": event.to_json(),
        "inputs": json.dumps([status.to_json()]),
    }
    # The component process starts without it
    fresh_registry()
    return args


class TestAdditionalTypes:
    """Types other packages registered in the recipe process reach a
    component process as --additional_types"""

    def test_importing_the_types_module_does_not_register_them(self, external_type):
        importlib.import_module("external_types")
        with pytest.raises(ValueError, match="not in available datatypes"):
            Topic(name="status", msg_type="ExternalStatus")

    def test_parsed_types_are_registered(self, executable, external_type):
        types = executable._parse_additional_types(json.dumps([external_type]))

        assert [t.__name__ for t in types] == ["ExternalStatus"]
        assert Topic(name="status", msg_type="ExternalStatus").msg_type is types[0]

    def test_an_event_on_an_external_type_is_rebuilt(self, executable, recipe_args):
        with pytest.raises(ValueError, match="not in available datatypes"):
            Event.from_json(recipe_args["event"])

        executable._parse_additional_types(recipe_args["additional_types"])

        event = Event.from_json(recipe_args["event"])
        assert [t.msg_type.__name__ for t in event.get_involved_topics()] == [
            "ExternalStatus"
        ]

    def test_an_event_trigger_on_an_external_type_is_rebuilt(
        self, executable, recipe_args
    ):
        trigger = json.dumps({"trigger_type": "Event", "trigger": recipe_args["event"]})

        executable._parse_additional_types(recipe_args["additional_types"])

        assert isinstance(executable._parse_trigger(trigger), Event)

    def test_an_input_of_an_external_type_is_rebuilt(self, executable, recipe_args):
        types = executable._parse_additional_types(recipe_args["additional_types"])

        topic_dicts = executable._deserialize_topics(recipe_args["inputs"])
        assert Topic(**topic_dicts[0]).msg_type is types[0]

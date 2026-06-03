"""Tests for the offline recording dataset builders.

Exercise the trace -> chat-row reconstruction, replanning/ordering, outcome
derivation, the JSONL emitter, and the reader's gzipped-JSONL fallback, all from
synthetic CortexTrace events (no real recording / ROS spin required).
"""

import gzip
import json

from agents.recording.builders.jsonl import write_jsonl
from agents.recording.reader import CORTEX_TRACE_TYPE, read_cortex_traces
from agents.recording.rows import build_rows


def _ev(event_type, payload=None, *, episode_id="ep1", t_ns=0, step_index=0):
    """Build a synthetic trace event dict (matches reader._event output)."""
    return {
        "t_ns": t_ns,
        "episode_id": episode_id,
        "component_node": "cortex",
        "event_type": event_type,
        "step_index": step_index,
        "task": "",
        "tool_name": "",
        "payload": payload or {},
    }


def _plan_payload():
    """A representative PLAN_GENERATED payload (OpenAI tool format)."""
    return {
        "messages": [
            {"role": "system", "content": "PLAN"},
            {"role": "user", "content": "audit aisle 4"},
        ],
        "tools": [{"type": "function", "function": {"name": "navigate"}}],
        "output": "I'll navigate.",
        "tool_calls": [
            {
                "id": "c0",
                "type": "function",
                "function": {"name": "navigate", "arguments": '{"x": 4}'},
            }
        ],
    }


def _confirm_payload():
    """A representative CONFIRMATION payload with a resolved tool call."""
    return {
        "system": "CONFIRM",
        "user": "Next: navigate {x:4}",
        "tools": [{"type": "function", "function": {"name": "navigate"}}],
        "output": "EXECUTE",
        "decision": "EXECUTE",
        "resolved_step": {
            "id": "c1",
            "type": "function",
            "function": {"name": "navigate", "arguments": '{"x": 4}'},
        },
    }


_MANIFEST = {
    "recipe": {"path": "/r/recipe.py", "sha256": "abc"},
    "stack": {"emos": "1.4.2", "packages": {"automatika-embodied-agents": "0.5.1"}},
    "runtime": {"namespace": "robot1"},
}


def test_one_row_per_llm_call():
    traces = [
        _ev("TASK_RECEIVED", {"task": "t"}, t_ns=0),
        _ev("PLAN_GENERATED", _plan_payload(), t_ns=1),
        _ev("CONFIRMATION", _confirm_payload(), t_ns=2),
        _ev("STEP_EXECUTING", {}, t_ns=3),
        _ev("STEP_COMPLETED", {"result": "ok"}, t_ns=4),
        _ev("EPISODE_COMPLETE", {}, t_ns=5),
    ]
    rows = build_rows(traces, _MANIFEST)
    # only PLAN_GENERATED and CONFIRMATION become rows
    assert [r["metadata"]["phase"] for r in rows] == ["planning", "execution"]
    assert all(r["metadata"]["outcome"] == "success" for r in rows)

    plan = rows[0]
    assert plan["messages"][-1]["role"] == "assistant"
    # OpenAI tool-call args preserved verbatim (JSON-encoded string)
    tc = plan["messages"][-1]["tool_calls"][0]
    assert tc["function"]["name"] == "navigate"
    assert tc["function"]["arguments"] == '{"x": 4}'
    assert plan["tools"]
    assert plan["metadata"]["recipe"] == "/r/recipe.py"
    assert plan["metadata"]["recipe_sha256"] == "abc"
    assert plan["metadata"]["robot_id"] == "robot1"

    conf = rows[1]
    assert [m["role"] for m in conf["messages"]] == ["system", "user", "assistant"]
    assert conf["messages"][-1]["tool_calls"][0]["function"]["name"] == "navigate"


def test_replanning_seq_orders_rows_step_index_resets():
    traces = [
        _ev("TASK_RECEIVED", {"task": "t"}, t_ns=0),
        _ev("PLAN_GENERATED", _plan_payload(), t_ns=1, step_index=0),
        _ev("CONFIRMATION", _confirm_payload(), t_ns=2, step_index=0),
        # Cortex replans -> step_index resets to 0
        _ev("PLAN_GENERATED", _plan_payload(), t_ns=3, step_index=0),
        _ev("CONFIRMATION", _confirm_payload(), t_ns=4, step_index=0),
        _ev("EPISODE_COMPLETE", {}, t_ns=5),
    ]
    rows = build_rows(traces, _MANIFEST)
    assert [r["metadata"]["phase"] for r in rows] == [
        "planning",
        "execution",
        "planning",
        "execution",
    ]
    assert [r["metadata"]["seq"] for r in rows] == [0, 1, 2, 3]
    assert all(r["metadata"]["step_index"] == 0 for r in rows)


def test_outcome_aborted_and_unknown():
    aborted = build_rows(
        [_ev("PLAN_GENERATED", _plan_payload(), t_ns=1), _ev("EPISODE_ABORTED", {}, t_ns=2)],
        _MANIFEST,
    )
    assert aborted[0]["metadata"]["outcome"] == "aborted"

    unknown = build_rows([_ev("PLAN_GENERATED", _plan_payload(), t_ns=1)], _MANIFEST)
    assert unknown[0]["metadata"]["outcome"] == "unknown"


def test_build_rows_without_manifest():
    rows = build_rows(
        [_ev("PLAN_GENERATED", _plan_payload(), t_ns=1), _ev("EPISODE_COMPLETE", {}, t_ns=2)]
    )
    assert rows[0]["metadata"]["recipe"] is None
    assert rows[0]["metadata"]["robot_id"] is None
    assert rows[0]["metadata"]["outcome"] == "success"


def test_write_jsonl_roundtrip(tmp_path):
    rows = build_rows(
        [_ev("PLAN_GENERATED", _plan_payload(), t_ns=1), _ev("EPISODE_COMPLETE", {}, t_ns=2)],
        _MANIFEST,
    )
    out = str(tmp_path / "out.jsonl")
    write_jsonl(rows, out)
    with open(out, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 1
    assert set(lines[0]) == {"messages", "tools", "metadata"}
    assert lines[0]["metadata"]["phase"] == "planning"


def test_reader_jsonl_fallback(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()

    def _line(event_type, payload, t_ns):
        # mirror the recorder JSONLSink line format
        return json.dumps(
            {
                "topic": "/cortex/trace",
                "t_ns": t_ns,
                "type": CORTEX_TRACE_TYPE,
                "data": {
                    "episode_id": "ep1",
                    "component_node": "cortex",
                    "event_type": event_type,
                    "step_index": 0,
                    "task": "",
                    "tool_name": "",
                    "payload_json": json.dumps(payload),
                },
            }
        )

    with gzip.open(rec / "trace_0000.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(_line("PLAN_GENERATED", _plan_payload(), 1) + "\n")
        f.write(_line("EPISODE_COMPLETE", {}, 2) + "\n")
        # a non-CortexTrace line must be ignored
        f.write(
            json.dumps(
                {"topic": "/other", "t_ns": 3, "type": "std_msgs/msg/String", "data": {}}
            )
            + "\n"
        )

    traces = read_cortex_traces(str(rec))
    assert [t["event_type"] for t in traces] == ["PLAN_GENERATED", "EPISODE_COMPLETE"]
    rows = build_rows(traces)
    assert len(rows) == 1
    assert rows[0]["metadata"]["phase"] == "planning"

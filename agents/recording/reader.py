"""Read recorded CortexTrace events (and the manifest) from a recording produced
by the Sugarcoat recorder.
"""

import glob
import gzip
import json
import os
from typing import Dict, List

from automatika_embodied_agents.msg import CortexTrace

# NOTE: rosbag2 (and the JSONL sink) store message types as "pkg/msg/Name"
# strings. Derive that from the imported class so a rename/move of CortexTrace
# fails loudly here.
CORTEX_TRACE_TYPE = f"{CortexTrace.__module__.split('.')[0]}/msg/{CortexTrace.__name__}"


def load_manifest(recording_dir: str) -> Dict:
    """Load ``manifest.json`` from a recording directory."""
    path = os.path.join(recording_dir, "manifest.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def read_cortex_traces(recording_dir: str) -> List[Dict]:
    """Return CortexTrace events (payload parsed) in time order.

    Each event is a dict: ``{t_ns, episode_id, component_node, event_type,
    step_index, task, tool_name, payload}``. Reads MCAP if present, else the
    gzipped JSONL sink.
    """
    bag = os.path.join(recording_dir, "mcap")
    if os.path.isdir(bag):
        return _read_from_mcap(bag)
    return _read_from_jsonl(recording_dir)


def _event(
    *,
    t_ns,
    episode_id,
    component_node,
    event_type,
    step_index,
    task,
    tool_name,
    payload_json,
) -> Dict:
    """Build a normalized trace-event dict with the payload JSON parsed."""
    payload: Dict = {}
    if payload_json:
        try:
            payload = json.loads(payload_json)
        except (ValueError, TypeError):
            payload = {}
    return {
        "t_ns": int(t_ns),
        "episode_id": episode_id,
        "component_node": component_node,
        "event_type": event_type,
        "step_index": int(step_index),
        "task": task,
        "tool_name": tool_name,
        "payload": payload,
    }


def _open_bag_reader(bag_dir: str):
    """Open a rosbag2 reader, choosing the compression-aware reader when the
    bag was written with file/message-level compression (per metadata.yaml)."""
    import rosbag2_py

    compressed = False
    meta = os.path.join(bag_dir, "metadata.yaml")
    if os.path.isfile(meta):
        with open(meta, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.startswith("compression_mode:"):
                    val = s.split(":", 1)[1].strip().strip("\"'").upper()
                    compressed = val not in ("", "NONE")
                    break

    storage = rosbag2_py.StorageOptions(uri=bag_dir, storage_id="mcap")
    converter = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader = (
        rosbag2_py.SequentialCompressionReader()
        if compressed
        else rosbag2_py.SequentialReader()
    )
    reader.open(storage, converter)
    return reader


def _read_from_mcap(bag_dir: str) -> List[Dict]:
    """Read CortexTrace events from an MCAP bag directory, time-ordered."""
    from rclpy.serialization import deserialize_message

    reader = _open_bag_reader(bag_dir)
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    out: List[dict] = []
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if types.get(topic) != CORTEX_TRACE_TYPE:
            continue
        msg = deserialize_message(data, CortexTrace)
        out.append(
            _event(
                t_ns=t_ns,
                episode_id=msg.episode_id,
                component_node=msg.component_node,
                event_type=msg.event_type,
                step_index=msg.step_index,
                task=msg.task,
                tool_name=msg.tool_name,
                payload_json=msg.payload_json,
            )
        )
    out.sort(key=lambda e: e["t_ns"])
    return out


def _read_from_jsonl(recording_dir: str) -> List[Dict]:
    """Read CortexTrace events from the gzipped JSONL sink files, time-ordered."""
    out: List[Dict] = []
    for path in sorted(glob.glob(os.path.join(recording_dir, "trace_*.jsonl.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if rec.get("type") != CORTEX_TRACE_TYPE:
                    continue
                d = rec.get("data", {})
                out.append(
                    _event(
                        t_ns=rec.get("t_ns", 0),
                        episode_id=d.get("episode_id", ""),
                        component_node=d.get("component_node", ""),
                        event_type=d.get("event_type", ""),
                        step_index=d.get("step_index", 0),
                        task=d.get("task", ""),
                        tool_name=d.get("tool_name", ""),
                        payload_json=d.get("payload_json", ""),
                    )
                )
    out.sort(key=lambda e: e["t_ns"])
    return out

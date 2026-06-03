"""Reconstruct OpenAI-tool-format chat training rows from CortexTrace events.

One row per Cortex LLM call.

- ``PLAN_GENERATED`` -> a planning row whose ``messages`` are the full planning
  conversation up to that call (system + task + prior tool_calls/results)
  followed by the assistant response (text + tool_calls); ``tools`` is the full
  catalog at planning time.
- ``CONFIRMATION`` -> an ``execution`` phase row (the per-step confirmation call:
  system + rendered context + assistant decision/tool_call); ``tools`` is the
  execution tool catalog.

Every row is tagged with metadata, including the episode ``outcome`` derived
from that episode's terminal ``EPISODE_COMPLETE`` / ``EPISODE_ABORTED`` event.
"""

from collections import defaultdict
from typing import Dict, List, Optional

_OUTCOME = {"EPISODE_COMPLETE": "success", "EPISODE_ABORTED": "aborted"}


def _episode_outcome(events: List[Dict]) -> str:
    """Return the episode outcome from its terminal trace event."""
    for e in reversed(events):
        if e["event_type"] in _OUTCOME:
            return _OUTCOME[e["event_type"]]
    return "unknown"


def _row_for_event(e: Dict) -> Optional[Dict]:
    """Build the chat row for one trace event (None if it is not an LLM call)."""
    et = e["event_type"]
    p = e["payload"]

    if et == "PLAN_GENERATED":
        assistant: Dict = {"role": "assistant", "content": p.get("output", "")}
        tool_calls = p.get("tool_calls")
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        return {
            "messages": list(p.get("messages", [])) + [assistant],
            "tools": p.get("tools", []),
            "_phase": "planning",
        }

    if et == "CONFIRMATION":
        assistant = {"role": "assistant", "content": p.get("output", "")}
        resolved = p.get("resolved_step")
        if resolved:
            assistant["tool_calls"] = [resolved]
        return {
            "messages": [
                {"role": "system", "content": p.get("system", "")},
                {"role": "user", "content": p.get("user", "")},
                assistant,
            ],
            "tools": p.get("tools", []),
            "_phase": "execution",
        }

    return None


def build_rows(traces: List[Dict], manifest: Optional[Dict] = None) -> List[Dict]:
    """Convert CortexTrace events into training rows with metadata."""
    manifest = manifest or {}
    recipe = manifest.get("recipe", {}) or {}
    stack = manifest.get("stack", {}) or {}
    runtime = manifest.get("runtime", {}) or {}

    by_episode: Dict[str, List[dict]] = defaultdict(list)
    for t in traces:
        by_episode[t["episode_id"]].append(t)

    rows: List[dict] = []
    for episode_id, events in by_episode.items():
        outcome = _episode_outcome(events)
        # NOTE: Cortex runs a plan-execute loop (it can replan), so one episode
        # yields multiple planning and execution rows. `step_index` is local to
        # each plan/execute call and resets on replan, so we add a per-episode
        # monotonic `seq` (rows are processed in time order) to order and
        # uniquely identify rows across replanning iterations.
        seq = 0
        for e in events:
            row = _row_for_event(e)
            if row is None:
                continue
            row["metadata"] = {
                "recipe": recipe.get("path"),
                "recipe_sha256": recipe.get("sha256"),
                "episode_id": episode_id,
                "seq": seq,
                "t_ns": e["t_ns"],
                "phase": row.pop("_phase"),
                "step_index": e["step_index"],
                "outcome": outcome,
                "stack": stack,
                "robot_id": runtime.get("namespace") or runtime.get("hostname"),
            }
            rows.append(row)
            seq += 1
    return rows

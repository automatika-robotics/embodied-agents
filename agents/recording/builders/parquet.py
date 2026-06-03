"""Parquet emitter for OpenAI tool-format chat rows.

Writes one Parquet file the HuggingFace ``datasets`` library can load directly.
``messages`` and ``tools`` are stored as JSON strings for schema stability
(rows have varied tool-call structure); the consumer loads the json and does
``tokenizer.apply_chat_template``. Metadata is flattened into columns so a
training pipeline can split by ``phase`` / ``outcome``.

Requires ``datasets`` (and ``pyarrow``).
"""

import json
import os
from typing import List, Dict

try:
    import datasets
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "The parquet builder requires the 'datasets' library. "
        "Install it with `pip install datasets pyarrow`."
    ) from e


def write_parquet(rows: List[Dict], out_path: str) -> str:
    """Write rows as a HuggingFace-loadable Parquet file; returns the path."""
    if not out_path.endswith(".parquet"):
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, "cortex_sft.parquet")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    records = [
        {
            "messages": json.dumps(r["messages"], default=str),
            "tools": json.dumps(r["tools"], default=str),
            "phase": r["metadata"].get("phase"),
            "outcome": r["metadata"].get("outcome"),
            "episode_id": r["metadata"].get("episode_id"),
            "seq": r["metadata"].get("seq"),
            "step_index": r["metadata"].get("step_index"),
            "recipe": r["metadata"].get("recipe"),
        }
        for r in rows
    ]
    datasets.Dataset.from_list(records).to_parquet(out_path)
    return out_path

"""JSONL emitter for OpenAI tool-format chat rows.

One JSON object per line: ``{"messages": [...], "tools": [...], "metadata":
{...}}`` in OpenAI tool-calling format. A generic chat-SFT format
consumed directly by various fine tuning APIs.
"""

import json
import os
from typing import List, Dict


def write_jsonl(rows: List[Dict], out_path: str) -> str:
    """Write rows as OpenAI tool-format chat JSONL; returns the output path."""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "messages": r["messages"],
                        "tools": r["tools"],
                        "metadata": r["metadata"],
                    },
                    default=str,
                )
                + "\n"
            )
    return out_path

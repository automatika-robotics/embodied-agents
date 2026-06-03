"""Offline dataset builders for EMOS recordings.

Turn recorded ``CortexTrace`` events (captured generically by the sugarcoat
Recorder) into trainable datasets: one OpenAI-tool-format chat row per Cortex
LLM call. Two serializations of the same schema: ``jsonl`` (Tinker / Unsloth /
TRL / OpenAI FT) and ``parquet`` (HuggingFace ``datasets``-loadable).

Read a recording and build a dataset with:

    ros2 run automatika_embodied_agents dataset_builder <recording_dir> --format jsonl -o out.jsonl
"""

from .reader import load_manifest, read_cortex_traces
from .rows import build_rows

__all__ = ["load_manifest", "read_cortex_traces", "build_rows"]

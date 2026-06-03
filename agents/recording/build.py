"""Offline dataset build CLI.

Turn a recording into a Cortex finetuning dataset. Run via the ROS-package
executable:

    ros2 run automatika_embodied_agents dataset_builder <recording_dir> --format jsonl   -o out.jsonl
    ros2 run automatika_embodied_agents dataset_builder <recording_dir> --format parquet -o ./hfds
    ros2 run automatika_embodied_agents dataset_builder <recording_dir> --format jsonl   -o out.jsonl --phase planning --outcome success

Both formats carry the same OpenAI tool-format chat schema (one row per Cortex
LLM call); they differ only in serialization: ``jsonl`` (Tinker / Unsloth / TRL /
OpenAI FT) vs ``parquet`` (HuggingFace ``datasets``-loadable).
"""

import argparse
import sys

from .reader import load_manifest, read_cortex_traces
from .rows import build_rows


def main(argv=None) -> int:
    """CLI entry point: read a recording and write the requested dataset format."""
    parser = argparse.ArgumentParser(prog="dataset_builder")
    parser.add_argument(
        "recording_dir", help="Recording directory (contains mcap/ and manifest.json)"
    )
    parser.add_argument("--format", choices=["jsonl", "parquet"], required=True)
    parser.add_argument("-o", "--output", required=True, help="Output file or directory")
    parser.add_argument(
        "--phase", choices=["planning", "execution"], help="Keep only this phase"
    )
    parser.add_argument(
        "--outcome",
        choices=["success", "aborted", "unknown"],
        help="Keep only episodes with this outcome",
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.recording_dir)
    traces = read_cortex_traces(args.recording_dir)
    rows = build_rows(traces, manifest)

    if args.phase:
        rows = [r for r in rows if r["metadata"]["phase"] == args.phase]
    if args.outcome:
        rows = [r for r in rows if r["metadata"]["outcome"] == args.outcome]

    if not rows:
        print("No Cortex trace rows found in the recording.", file=sys.stderr)

    if args.format == "jsonl":
        from .builders.jsonl import write_jsonl

        out = write_jsonl(rows, args.output)
    else:
        from .builders.parquet import write_parquet

        out = write_parquet(rows, args.output)

    print(f"Wrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

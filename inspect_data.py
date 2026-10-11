"""Inspect AEGIS structure before connecting APIs."""
import argparse
import json
from pathlib import Path


def object_or_empty(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            item = json.loads(value)
            if isinstance(item, dict):
                return item
        except (ValueError, TypeError):
            pass
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/test.jsonl")
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"File not found: {path}. Download it first.")
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= args.count:
                break
            r = json.loads(line)
            meta = object_or_empty(r.get("metadata"))
            inp = object_or_empty(r.get("input"))
            out = object_or_empty(r.get("output"))
            history = inp.get("conversation_history", [])
            print("\n" + "=" * 50)
            print("Example", i + 1, "ID:", r.get("id"))
            print("Top-level keys:", list(r))
            print("Metadata:", {k:meta.get(k) for k in ("framework", "benchmark", "task_type")})
            print("Question:", str(inp.get("query", ""))[:220])
            print("History type:", type(history).__name__, "length:", len(history) if isinstance(history, list) else "unknown")
            if isinstance(history, list):
                for event in history[:2]:
                    print("Event preview:", str(event)[:400])
            print("Label example (for evaluation ONLY):", str(out.get("faulty_agents", []))[:350])

if __name__ == "__main__":
    main()

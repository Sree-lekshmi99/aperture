"""Bounded, validation-only measurement using Aperture's unchanged investigators.

Run from the project: .venv/bin/python -B -u context_experiment.py [--analyze-only]
No Voyage calls, policy tuning, model training, or test-split reads.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import parse_qs, urlsplit

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)
import aperture as harness

STUDY_ID = "validation-context-study-20261011-v1"
STRATEGIES = ("large", "outline")
MAX_NEW_CALLS = 16
OUT = ROOT / "reports" / STUDY_ID


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save_json(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_validation():
    profiles, records = [], {}
    for lineno, line in enumerate((ROOT / "data" / "val.jsonl").open(), 1):
        raw = json.loads(line)
        data = harness.as_dict(raw.get("input"))
        meta = harness.as_dict(raw.get("metadata"))
        history = data.get("conversation_history")
        if not isinstance(history, list) or not history:
            continue
        trace_id = f"val:{raw.get('id', str(lineno))}"
        trace = {
            "_id": trace_id, "source_id": str(raw.get("id", lineno)), "split": "val",
            "task_type": str(meta.get("task_type") or "unknown"),
            "benchmark": str(meta.get("benchmark") or "unknown"),
            "task": str(data.get("query") or ""), "history": history,
        }
        events = [harness.event_details(event, i + 1) for i, event in enumerate(history)]
        full = json.dumps(events, ensure_ascii=False)
        large, agents, large_cut = harness.context(trace, "large")
        outline, _, outline_cut = harness.context(trace, "outline")
        profile = {
            "trace_id": trace_id, "source_line": lineno, "events": len(events),
            "full_context_chars": len(full), "content_chars": sum(len(e["content"]) for e in events),
            "raw_history_chars": len(json.dumps(history, ensure_ascii=False)),
            "task_chars": len(trace["task"]), "approx_context_tokens": math.ceil(len(full) / 4),
            "agents": len(agents), "known_agents": agents,
            "large_context_chars": len(large), "large_context_truncated": large_cut,
            "outline_context_chars": len(outline), "outline_context_truncated": outline_cut,
            "outline_snippet_clipped_events": sum(len(e["content"]) > 180 for e in events),
            "task_type": trace["task_type"], "benchmark": trace["benchmark"],
            "framework": str(meta.get("framework") or "unknown"),
        }
        profiles.append(profile)
        # Ground-truth output is not inspected during selection or profiling.
        records[trace_id] = (trace, raw)
    return profiles, records


def select_cases(profiles):
    ordered = sorted(profiles, key=lambda p: (p["full_context_chars"], p["trace_id"]))
    p90 = ordered[int(0.9 * len(ordered))]["full_context_chars"]
    selections = [
        ("L1", "long/complex", "Largest serialized context", max(profiles, key=lambda p: p["full_context_chars"])),
        ("L2", "long/complex", "Most events; break ties by context size", max(profiles, key=lambda p: (p["events"], p["full_context_chars"]))),
        ("L3", "long/complex", "Most agents; break ties by event count and size", max(profiles, key=lambda p: (p["agents"], p["events"], p["full_context_chars"]))),
        ("L4", "long/complex", "Math/debate trace closest to validation p90 size", min(
            [p for p in profiles if p["task_type"] == "math" and p["framework"] == "llm_debate"],
            key=lambda p: (abs(p["full_context_chars"] - p90), p["trace_id"]))),
    ]
    for index, fraction in enumerate((0.15, 0.25, 0.40, 0.50), 1):
        selections.append((f"S{index}", "short/medium", f"Validation size percentile {fraction:.0%}", ordered[int(fraction * (len(ordered) - 1))]))
    assert len({p["trace_id"] for _, _, _, p in selections}) == 8
    return [{**profile, "case_id": case_id, "group": group, "selection_reason": reason}
            for case_id, group, reason, profile in selections]


def summarize(profiles):
    summary = {"validation_records": len(profiles), "large_truncated_count": sum(p["large_context_truncated"] for p in profiles),
               "outline_cap_truncated_count": sum(p["outline_context_truncated"] for p in profiles)}
    for field in ("events", "full_context_chars", "approx_context_tokens", "agents"):
        values = sorted(p[field] for p in profiles)
        summary[field] = {"min": values[0], "p25": values[len(values)//4], "median": statistics.median(values),
                          "p75": values[3*len(values)//4], "p90": values[int(.9*len(values))], "max": values[-1]}
    return summary


def observation(profile, trace, result, cached, wall_seconds):
    predicted = harness.normalize_names(result["predicted_agents"])
    actual = harness.normalize_names(result["true_agents"])
    strategy = result["strategy"]
    payload, _, truncated = harness.context(trace, strategy)
    return {
        "_id": f"{STUDY_ID}:{profile['case_id']}:{strategy}", "experiment_id": STUDY_ID,
        "case_id": profile["case_id"], "group": profile["group"], "result_id": result["_id"],
        "trace_id": trace["_id"], "strategy": strategy, "model": result["model"],
        "predicted_agents": sorted(predicted), "actual_agents": sorted(actual),
        "f1": result["f1"], "exact_match": result["exact_match"],
        "false_positives": sorted(predicted - actual), "false_negatives": sorted(actual - predicted),
        "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
        "execution_seconds": result["elapsed_seconds"], "wall_seconds": round(wall_seconds, 3),
        "context_truncated": result["context_truncated"], "cache_used": cached,
        "context_chars": len(payload), "context_sha256": digest(payload),
        "full_context_chars": profile["full_context_chars"], "events": profile["events"], "agents": profile["agents"],
        "outline_snippet_clipped_events": profile["outline_snippet_clipped_events"] if strategy == "outline" else 0,
        "evidence_steps": result.get("evidence_steps", []), "explanation": result.get("explanation", ""),
        "recorded_at": utcnow(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    profiles, records = load_validation()
    selected = select_cases(profiles)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    manifest = {
        "_id": STUDY_ID, "status": "selected", "created_at": utcnow(), "split": "val",
        "model": model, "temperature": 0, "response_format": "json_object",
        "strategies": list(STRATEGIES), "max_new_investigator_calls": MAX_NEW_CALLS,
        "max_context_chars": harness.MAX_CHARS, "outline_characters_per_event": 180,
        "selection_uses_ground_truth": False, "selection_note": "Purposive stress/reference sample, not a random population estimate.",
        "token_estimate_note": "ceil(normalized serialized history characters / 4); excludes task and system prompt.",
        "harness_sha256": hashlib.sha256((ROOT / "aperture.py").read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256((ROOT / "data" / "val.jsonl").read_bytes()).hexdigest(),
        "dataset_summary": summarize(profiles), "selected": selected,
    }
    save_json("profiles.json", profiles)
    save_json("selection.json", manifest)
    print(json.dumps({"study": STUDY_ID, "summary": manifest["dataset_summary"], "selected": selected}), flush=True)
    if args.analyze_only:
        return
    uri = os.environ.get("MONGODB_URI", "")
    parsed = urlsplit(uri)
    if parsed.scheme != "mongodb+srv" or not (parsed.hostname or "").endswith(".mongodb.net"):
        raise RuntimeError("An Atlas SRV connection is required; configure the official sandbox URI.")
    options = {key.lower(): values for key, values in parse_qs(parsed.query).items()}
    if any(value.lower() == "true" for key in ("tlsinsecure", "tlsallowinvalidcertificates", "tlsallowinvalidhostnames") for value in options.get(key, [])):
        raise RuntimeError("TLS verification must remain enabled.")
    if os.getenv("OPENAI_BASE_URL") and urlsplit(os.environ["OPENAI_BASE_URL"]).hostname != "api.openai.com":
        raise RuntimeError("Unexpected investigator API endpoint.")
    client = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=30000,
                         connectTimeoutMS=10000, socketTimeoutMS=45000)
    api = None
    try:
        client.admin.command("ping")
        db = client[harness.DB_NAME]
        existing = db.experiments.find_one({"_id": STUDY_ID})
        if existing:
            for field in ("model", "harness_sha256", "validation_sha256", "selected"):
                if existing[field] != manifest[field]:
                    raise RuntimeError("Study configuration changed; refusing to mix results.")
        else:
            db.experiments.insert_one(manifest)
        traces, labels = [], []
        for profile in selected:
            trace, raw = records[profile["trace_id"]]
            truth = harness.as_dict(raw.get("output")).get("faulty_agents")
            if not isinstance(truth, list):
                raise RuntimeError("Selected record has invalid evaluation labels.")
            traces.append(UpdateOne({"_id": trace["_id"]}, {"$set": trace}, upsert=True))
            labels.append(UpdateOne({"_id": trace["_id"]}, {"$set": {"faulty_agents": truth}}, upsert=True))
        db.traces.bulk_write(traces, ordered=False)
        db.labels.bulk_write(labels, ordered=False)
        assert db.traces.count_documents({"_id": {"$in": [p["trace_id"] for p in selected]}}) == 8
        assert db.labels.count_documents({"_id": {"$in": [p["trace_id"] for p in selected]}}) == 8
        print("Persisted selection and exactly eight selected validation traces; labels are separate.", flush=True)
        api = harness.model_client().with_options(timeout=90, max_retries=0)
        output = list(db.experiment_results.find({"experiment_id": STUDY_ID}))
        completed = {row["_id"] for row in output}
        attempts = db.experiment_calls.count_documents({"experiment_id": STUDY_ID})
        for index, profile in enumerate(selected):
            trace = db.traces.find_one({"_id": profile["trace_id"]})
            assert set(trace) == {"_id", "source_id", "split", "task_type", "benchmark", "task", "history"}
            # Alternate order to reduce a systematic first/second strategy timing effect.
            order = STRATEGIES if index % 2 == 0 else tuple(reversed(STRATEGIES))
            for strategy in order:
                observation_id = f"{STUDY_ID}:{profile['case_id']}:{strategy}"
                if observation_id in completed:
                    continue
                key = f"{trace['_id']}:{strategy}:{model}"
                cached_result = db.results.find_one({"_id": key})
                cached = cached_result is not None
                prior_attempt = db.experiment_calls.find_one({"_id": observation_id})
                if not cached and prior_attempt:
                    raise RuntimeError("An earlier request has no saved result; refusing an automatic duplicate investigation.")
                if not cached:
                    if attempts >= MAX_NEW_CALLS:
                        raise RuntimeError("The 16-call experiment budget is exhausted.")
                    db.experiment_calls.insert_one({"_id": observation_id, "experiment_id": STUDY_ID,
                        "trace_id": trace["_id"], "strategy": strategy, "model": model, "started_at": utcnow(), "status": "started"})
                    attempts += 1
                print(f"START {profile['case_id']} {strategy}; cache={cached}; new calls started={attempts}/16", flush=True)
                started = time.monotonic()
                try:
                    result = cached_result or harness.run_case(db, trace, strategy, api)
                except Exception as exc:
                    db.experiment_calls.update_one({"_id": observation_id}, {"$set": {
                        "status": "error", "error_class": type(exc).__name__, "http_status": getattr(exc, "status_code", None), "finished_at": utcnow()}})
                    raise
                row = observation(profile, trace, result, cached, time.monotonic() - started)
                db.experiment_results.replace_one({"_id": row["_id"]}, row, upsert=True)
                if not cached:
                    db.experiment_calls.update_one({"_id": observation_id}, {"$set": {"status": "complete", "finished_at": utcnow()}})
                output.append(row)
                completed.add(row["_id"])
                save_json("results.json", output)
                print(json.dumps({key: row[key] for key in ("case_id", "strategy", "f1", "exact_match", "false_positives", "false_negatives", "prompt_tokens", "completion_tokens", "context_truncated", "cache_used")}), flush=True)
        assert len(output) == 16
        assert db.experiment_results.count_documents({"experiment_id": STUDY_ID}) == 16
        db.experiments.update_one({"_id": STUDY_ID}, {"$set": {"status": "measurements_complete", "new_calls_started": attempts, "completed_at": utcnow()}})
        save_json("results.json", sorted(output, key=lambda row: (row["case_id"], row["strategy"])))
        print(f"MEASUREMENTS COMPLETE: 16 results persisted; {attempts} new calls. No test runs or policy updates.", flush=True)
    finally:
        client.close()
        if api is not None:
            api.close()


if __name__ == "__main__":
    try:
        main()
    except (Exception, SystemExit) as exc:
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            raise
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__,
                          "http_status": getattr(exc, "status_code", None), "raw_error_redacted": True}), flush=True)
        raise SystemExit(1) from None

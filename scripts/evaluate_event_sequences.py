"""One-to-one temporal event scoring (not official MultiSports tube mAP).

Ground truth: a JSON object with `videos` and `events`. Videos specify video_id,
fps, frame_count, split; events specify video_id, type, start_frame, end_frame
(zero-based inclusive action intervals). Point annotations use equal endpoints.
Predictions are exact <video_id>_analysis.json paths under analysis_dir. Missing
videos count as missed events, never silently disappear from recall.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else 0.0}


def _distance(truth, prediction):
    return max(truth["start_seconds"] - prediction["time_seconds"],
               prediction["time_seconds"] - truth["end_seconds"], 0.0)


def _match(truth, predictions, tolerance, *, same_class=True):
    """Maximum-cardinality bipartite matching, deterministic distance tie-break.

    Unlike greedy matching, one ambiguous event cannot consume the sole match
    of another event. Metrics are class-aware; wrong-class matches are only
    used afterward to diagnose confusion, never credited as true positives.
    """
    adjacency = {
        i: sorted((j for j, p in enumerate(predictions)
                   if (not same_class or g["type"] == p["type"])
                   and _distance(g, p) <= tolerance + 1e-9),
                  key=lambda j: (_distance(g, predictions[j]), j))
        for i, g in enumerate(truth)
    }
    owners = {}

    def visit(i, seen):
        for j in adjacency[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or visit(owners[j], seen):
                owners[j] = i
                return True
        return False

    for i in sorted(adjacency, key=lambda index: (len(adjacency[index]), index)):
        visit(i, set())
    return sorted((i, j) for j, i in owners.items())


def _predictions(report):
    fps = float(report["source"]["fps"])
    offset = float(report["source"].get("startSeconds", 0))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Prediction fps must be positive and finite")
    events = list(report.get("events", []))
    # Throw-ins are deliberately excluded from public pass totals, but remain
    # independently scoreable through the inspectable lifecycle decisions.
    for decision in report.get("diagnostics", {}).get("shotAttemptTimeline", {}).get("arbitration", []):
        if decision.get("reason") == "throw_in":
            events.append({**decision["event"], "type": "throw_in"})
    normalized = []
    for event in events:
        evidence = event.get("evidence", event)
        frame = evidence.get("release_frame") if event["type"] in ("pass", "shot_attempt", "throw_in") else None
        if frame is None:
            frame = event.get("frameIndex", event.get("frame_index"))
        if frame is None or not math.isfinite(float(frame)) or float(frame) < 0:
            raise ValueError("Prediction event has no valid frame")
        normalized.append({"type": event["type"], "time_seconds": float(frame) / fps + offset,
                           "event": event, "evidence": evidence})
    return normalized


def evaluate(annotations, analysis_dir, *, tolerance_seconds=0.25, split="all"):
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("Tolerance must be nonnegative and finite")
    videos = [v for v in annotations["videos"] if split == "all" or v["split"] == split]
    if not videos or len({v["video_id"] for v in videos}) != len(videos):
        raise ValueError("Expected nonempty, unique video IDs in selected split")
    known = {v["video_id"] for v in annotations["videos"]}
    if any(e["video_id"] not in known for e in annotations["events"]):
        raise ValueError("Annotation references an unknown video")
    grouped = defaultdict(list)
    for event in annotations["events"]:
        grouped[event["video_id"]].append(event)
    totals = defaultdict(Counter)
    confusion = defaultdict(Counter)
    results, missing = [], []
    duplicate_count = prediction_count = 0
    classes = set(annotations.get("scored_types", ["pass", "shot_attempt", "interception", "throw_in"]))
    for video in videos:
        video_id = video["video_id"]
        fps = float(video["fps"])
        if not math.isfinite(fps) or fps <= 0 or int(video["frame_count"]) <= 0:
            raise ValueError(f"Invalid video metadata: {video_id}")
        path = (analysis_dir / f"{video_id}_analysis.json").resolve()
        if not path.is_relative_to(analysis_dir.resolve()):
            raise ValueError("Unsafe video ID")
        truth = []
        for event in grouped[video_id]:
            start, end = float(event["start_frame"]), float(event["end_frame"])
            if not (0 <= start <= end < video["frame_count"]):
                raise ValueError(f"Invalid event interval in {video_id}")
            if event["type"] in classes:
                truth.append({**event, "start_seconds": start / fps, "end_seconds": end / fps})
        if path.exists():
            predictions = [p for p in _predictions(json.loads(path.read_text(encoding="utf-8")))
                           if p["type"] in classes]
        else:
            missing.append(video_id)
            predictions = []
        matches = _match(truth, predictions, tolerance_seconds)
        used_g, used_p = {i for i, _ in matches}, {j for _, j in matches}
        for g, p in matches:
            totals[truth[g]["type"]]["tp"] += 1
            confusion[truth[g]["type"]][predictions[p]["type"]] += 1
        for i, g in enumerate(truth):
            if i not in used_g:
                totals[g["type"]]["fn"] += 1
        for j, p in enumerate(predictions):
            if j not in used_p:
                totals[p["type"]]["fp"] += 1
        unmatched_g = [g for i, g in enumerate(truth) if i not in used_g]
        unmatched_p = [p for j, p in enumerate(predictions) if j not in used_p]
        confused = _match(unmatched_g, unmatched_p, tolerance_seconds, same_class=False)
        for i, j in confused:
            confusion[unmatched_g[i]["type"]][unmatched_p[j]["type"]] += 1
        for i, g in enumerate(unmatched_g):
            if i not in {i for i, _ in confused}:
                confusion[g["type"]]["missed"] += 1
        duplicates = 0
        for j, p in enumerate(unmatched_p):
            if j not in {j for _, j in confused}:
                confusion["no_event"][p["type"]] += 1
                duplicates += any(truth[i]["type"] == p["type"]
                                  and _distance(truth[i], p) <= tolerance_seconds
                                  for i in used_g)
        duplicate_count += duplicates
        prediction_count += len(predictions)
        results.append({"video_id": video_id, "truth": len(truth), "predicted": len(predictions),
                        "tp": len(matches), "fp": len(unmatched_p), "fn": len(unmatched_g),
                        "duplicate_predictions": duplicates,
                        "matches": [{"truth_index": i, "prediction_index": j,
                                     "interval_distance_seconds": round(_distance(truth[i], predictions[j]), 4)}
                                    for i, j in matches]})
    aggregate = Counter()
    for counts in totals.values():
        aggregate.update(counts)
    return {"benchmark": annotations.get("name"), "split": split,
            "metric": "temporal event point-to-action-interval F1; not official spatial tube mAP",
            "tolerance_seconds": tolerance_seconds, "video_count": len(videos),
            "processed_video_count": len(videos) - len(missing), "missing_videos": missing,
            "coverage": round((len(videos) - len(missing)) / len(videos), 4),
            "micro": _metrics(aggregate["tp"], aggregate["fp"], aggregate["fn"]),
            "by_class": {key: _metrics(totals[key]["tp"], totals[key]["fp"], totals[key]["fn"])
                         for key in sorted(classes)},
            "confusion": {key: dict(value) for key, value in sorted(confusion.items())},
            "duplicate_predictions": duplicate_count,
            "duplicate_rate": round(duplicate_count / prediction_count, 4) if prediction_count else 0.0,
            "per_video": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-seconds", type=float, default=0.25)
    parser.add_argument("--split", default="all")
    args = parser.parse_args()
    report = evaluate(json.loads(args.annotations.read_text(encoding="utf-8")),
                      args.analysis_dir, tolerance_seconds=args.tolerance_seconds, split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_video"}, indent=2))


if __name__ == "__main__":
    main()

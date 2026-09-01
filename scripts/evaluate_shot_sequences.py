import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "benchmarks" / "shot_sequences_v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score CourtVision shot-attempt events against BARD labels."
    )
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("calibration", "holdout", "all"), default="all")
    return parser.parse_args()


def _load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _analysis_path(directory, video_id):
    direct = sorted(directory.glob(f"{video_id}*_analysis.json"))
    if direct:
        exact = [path for path in direct if path.name == f"{video_id}_analysis.json"]
        return exact[0] if exact else direct[0]
    candidates = sorted(directory.rglob(f"{video_id}*_analysis.json"))
    exact = [path for path in candidates if path.name == f"{video_id}_analysis.json"]
    return exact[0] if exact else candidates[0] if candidates else None


def _prediction_labels(events):
    shots = [event for event in events if event.get("type") == "shot_attempt"]
    return Counter({"shot_attempt": len(shots)}), shots


def _truth_labels(sequences):
    return Counter({"shot_attempt": len(sequences)})


def _metrics(tp, predicted, truth):
    precision = tp / predicted if predicted else 0.0
    recall = tp / truth if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "predicted": predicted,
        "truth": truth,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate(analysis_dir, benchmark_dir, split="all"):
    records = _load_jsonl(benchmark_dir / "sequences.jsonl")
    if split != "all":
        records = [record for record in records if record["split"] == split]
    grouped = defaultdict(list)
    for record in records:
        grouped[record["video_id"]].append(record)

    totals_truth = Counter()
    totals_predicted = Counter()
    totals_tp = Counter()
    per_video = []
    style = defaultdict(lambda: {"truth": 0, "ordinal_covered": 0})
    missing = []
    for video_id, truth_sequences in grouped.items():
        truth_sequences.sort(key=lambda item: item["sequence_index"])
        path = _analysis_path(analysis_dir, video_id)
        if path is None:
            missing.append(video_id)
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        prediction, shots = _prediction_labels(report.get("events", []))
        truth = _truth_labels(truth_sequences)
        keys = set(truth) | set(prediction)
        tp = Counter({key: min(truth[key], prediction[key]) for key in keys})
        totals_truth.update(truth)
        totals_predicted.update(prediction)
        totals_tp.update(tp)
        for ordinal, sequence in enumerate(truth_sequences):
            bucket = style[sequence["shot_style"]]
            bucket["truth"] += 1
            if ordinal < len(shots):
                bucket["ordinal_covered"] += 1
        per_video.append({
            "video_id": video_id,
            "analysis": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
            "truth": dict(truth),
            "predicted": dict(prediction),
            "true_positive": dict(tp),
        })

    scored_classes = ("shot_attempt",)
    by_class = {
        key: _metrics(totals_tp[key], totals_predicted[key], totals_truth[key])
        for key in scored_classes
    }
    micro_tp = sum(totals_tp[key] for key in scored_classes)
    micro_predicted = sum(totals_predicted[key] for key in scored_classes)
    micro_truth = sum(totals_truth[key] for key in scored_classes)
    style_report = {
        key: {
            **value,
            "ordinal_recall_proxy": round(
                value["ordinal_covered"] / value["truth"], 4
            ) if value["truth"] else 0.0,
        }
        for key, value in sorted(style.items())
    }
    return {
        "benchmark": benchmark_dir.relative_to(ROOT).as_posix(),
        "analysis_dir": analysis_dir.relative_to(ROOT).as_posix() if analysis_dir.is_relative_to(ROOT) else str(analysis_dir),
        "split": split,
        "scored_video_count": len(per_video),
        "scored_sequence_count": sum(len(grouped[item["video_id"]]) for item in per_video),
        "missing_videos": sorted(missing),
        "micro_event": _metrics(micro_tp, micro_predicted, micro_truth),
        "by_class": by_class,
        "shot_style_ordinal_coverage": style_report,
        "per_video": per_video,
        "scoring_note": (
            "BARD labels are ordered clip-level actions without timestamps. "
            "Shot-attempt F1 uses per-clip event-count multisets. Make/miss, "
            "rebound, and dead-ball outcomes are intentionally not produced "
            "or scored; style coverage is an ordinal proxy, not temporal F1."
        ),
    }


def main():
    args = parse_args()
    report = evaluate(args.analysis_dir.resolve(), args.benchmark_dir.resolve(), args.split)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output.resolve()}")
    print(text)


if __name__ == "__main__":
    main()

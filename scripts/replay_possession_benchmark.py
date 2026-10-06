"""Replay the local benchmark clips and score possession/event changes.

`run` executes main.py --analysis-only on the six BARD shot clips and the five
courtvision_v1 clips. Detector caches are reused, so only tracking selection,
possession and event logic run afresh. `score` reports:

- temporal shot-window and pass/interception F1 (evaluate_event_sequences);
- courtvision_v1 controlled-possession frames that receive the correct team,
  wrong-team calls, and published teams on every other labelled state;
- why each remaining controlled frame has no published team.

`multisports` registers and runs one frozen post-change MultiSports replay via
prepare_multisports_replay.py and run_multisports_validation.py. It is the
independent confirmation for a change, never a tuning loop.

These are calibration diagnostics on development clips, not accuracy claims.
"""
import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_event_sequences import evaluate

SHOT_WINDOWS = ROOT / "benchmarks/event_sequences_v2/local_shot_windows.json"
COURTVISION_V1 = ROOT / "benchmarks/courtvision_v1"
DEFAULT_REPLAY_ROOT = ROOT / "runs/replays"
OCCLUDED_REASONS = {
    "ball_missing",
    "pending_candidate_ball_missing",
    "interpolated_ball_not_confirmable",
}


def clip_ids():
    shots = json.loads(SHOT_WINDOWS.read_text(encoding="utf-8"))
    dataset = json.loads((COURTVISION_V1 / "dataset.json").read_text(encoding="utf-8"))
    return [v["video_id"] for v in shots["videos"]] + [v["id"] for v in dataset["videos"]]


def run(label, replay_root, video_ids=None, main_args=()):
    output_dir = replay_root / label
    output_dir.mkdir(parents=True, exist_ok=True)
    for video_id in video_ids or clip_ids():
        log_path = output_dir / f"{video_id}.log"
        with log_path.open("w", encoding="utf-8") as log:
            # A fresh process per clip so module state cannot leak between clips.
            result = subprocess.run(
                [
                    sys.executable, str(ROOT / "main.py"),
                    str(ROOT / "input_videos" / f"{video_id}.mp4"),
                    "--analysis-only", "--allow-uncertain-teams",
                    "--output-analysis", str(output_dir / f"{video_id}_analysis.json"),
                    *main_args,
                ],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        cold = log_text.count("Running one shared E-BARD scene-detection pass")
        print(f"{video_id}: exit={result.returncode} cold_detector_passes={cold}", flush=True)


def _possession_event_annotations():
    dataset = json.loads((COURTVISION_V1 / "dataset.json").read_text(encoding="utf-8"))
    annotations = {
        "name": "courtvision_v1 verified possession events",
        "scored_types": ["pass", "interception"],
        "videos": [
            {"video_id": v["id"], "fps": v["fps"], "frame_count": v["frame_count"], "split": "calibration"}
            for v in dataset["videos"]
        ],
        "events": [],
    }
    for line in (COURTVISION_V1 / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("review_status") != "verified":
            continue
        boundary = event.get("release_frame") if event["event_type"] == "pass" else event.get("catch_frame")
        if boundary is not None:
            annotations["events"].append({
                "video_id": event["video_id"], "type": event["event_type"],
                "start_frame": boundary, "end_frame": boundary,
            })
    return annotations


def _verified_frames():
    labels = defaultdict(list)
    for line in (COURTVISION_V1 / "annotations.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("review_status") == "verified":
            labels[record["video_id"]].append(record)
    return labels


def _team_mapping(records, frames, team_ids):
    """Cluster IDs have no semantic order; pick the mapping that best fits labels."""
    def correct(lookup):
        return sum(
            1 for r in records
            if r["possession"]["state"] == "controlled"
            and lookup.get(frames[r["frame_index"]].get("possessionTeamId")) == r["possession"]["team"]
        )
    options = [dict(zip(team_ids, m)) for m in permutations(["team_a", "team_b"], min(2, len(team_ids)))]
    return max(options or [{}], key=correct)


def _nearest_holders(timeline, index):
    left = index
    while left >= 0 and timeline[left].get("holder_id") is None:
        left -= 1
    right = index
    while right < len(timeline) and timeline[right].get("holder_id") is None:
        right += 1
    return left, right


def _miss_cause(timeline, index, teams, fps):
    """Classify why a labelled controlled frame has no published team."""
    if timeline[index].get("holder_id") is not None:
        return "holder found, team not published", []
    left, right = _nearest_holders(timeline, index)
    if left < 0 or right >= len(timeline):
        return "no holder on one side (clip edge)", []
    left_id, right_id = timeline[left]["holder_id"], timeline[right]["holder_id"]
    if left_id == right_id:
        reasons = {timeline[k].get("reason") for k in range(left + 1, right)}
        if right - left - 1 > round(fps):
            return "same holder, gap longer than 1 s", []
        if reasons - OCCLUDED_REASONS:
            return "same holder, ball seen or other reason in gap", []
        return "same holder, other", []
    left_team, right_team = teams.get(int(left_id), {}), teams.get(int(right_id), {})
    unknown = [
        (int(track_id), track)
        for track_id, track in ((left_id, left_team), (right_id, right_team))
        if track.get("team_id") is None
    ]
    if unknown:
        return "different holders, a holder's team unknown", unknown
    if left_team["team_id"] == right_team["team_id"]:
        return "different holders, same team", []
    return "different holders, opposite teams", []


def score(replay_dir):
    report = {"replay": str(replay_dir)}
    shots = json.loads(SHOT_WINDOWS.read_text(encoding="utf-8"))
    for name, annotations in (("shot_windows", shots), ("possession_events", _possession_event_annotations())):
        report[name] = evaluate(annotations, replay_dir, tolerance_seconds=0.25)["micro"]

    totals = Counter()
    causes = Counter()
    unknown_holder_tracks = {}
    for video_id, records in _verified_frames().items():
        analysis = json.loads((replay_dir / f"{video_id}_analysis.json").read_text(encoding="utf-8"))
        frames = analysis["frames"]
        fps = analysis["source"]["fps"]
        timeline = analysis["diagnostics"]["possessionTimeline"]["semantic"]["frames"]
        teams = {
            int(track_id): track
            for track_id, track in analysis["diagnostics"]["teamAssignment"]["track_assignments"].items()
        }
        team_ids = sorted({t["team_id"] for t in teams.values() if t.get("team_id") is not None})
        lookup = _team_mapping(records, frames, team_ids)
        for record in records:
            index = record["frame_index"]
            published = frames[index].get("possessionTeamId")
            state = record["possession"]["state"]
            if state != "controlled":
                # A team keeps possession during its own pass, so in_flight is
                # reported separately; shot and dead frames are errors.
                totals[f"{state}_frames"] += 1
                totals[f"team_on_{state}_frames"] += published is not None
                continue
            totals["controlled_frames"] += 1
            if published is not None:
                totals["controlled_with_a_team"] += 1
                if lookup.get(published) == record["possession"]["team"]:
                    totals["team_correct_on_controlled"] += 1
                else:
                    totals["team_wrong_on_controlled"] += 1
                continue
            cause, unknown = _miss_cause(timeline, index, teams, fps)
            causes[cause] += 1
            for track_id, track in unknown:
                entry = unknown_holder_tracks.setdefault(f"{video_id}:{track_id}", {
                    "video_id": video_id,
                    "track_id": track_id,
                    "missed_frames": 0,
                    **{key: track.get(key) for key in (
                        "status", "reason", "observation_count", "accepted_observation_count",
                        "confident_observation_count", "agreement", "weight_share",
                        "team_vote_counts", "rejection_counts",
                    )},
                })
                entry["missed_frames"] += 1
    totals["end_to_end_team_accuracy"] = round(
        totals["team_correct_on_controlled"] / totals["controlled_frames"], 4
    ) if totals["controlled_frames"] else 0.0
    report["possession_frames"] = dict(totals)
    report["missed_controlled_frame_causes"] = dict(causes.most_common())
    report["unknown_team_holder_tracks"] = sorted(
        unknown_holder_tracks.values(), key=lambda item: -item["missed_frames"]
    )
    return report


def multisports(label, source):
    """Register one frozen post-change replay, run it, and print its reports."""
    directory = ROOT / "runs" / f"multisports-{label}"
    if not directory.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/prepare_multisports_replay.py"),
             "--source", str(source), "--output", str(directory)],
            cwd=ROOT, check=True,
        )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_multisports_validation.py"),
         "--directory", str(directory)],
        cwd=ROOT, check=True,
    )
    summary = {}
    for name in ("primary", "sensitivity"):
        report = json.loads((directory / f"{name}_report.json").read_text(encoding="utf-8"))
        summary[name] = {"tolerance_seconds": report["tolerance_seconds"], "coverage": report["coverage"],
                         "micro": report["micro"], "by_class": report["by_class"]}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Replay the benchmark clips into runs/replays/<label>.")
    run_parser.add_argument("label")
    run_parser.add_argument("--video-id", action="append", dest="video_ids")
    run_parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY_ROOT)
    run_parser.add_argument(
        "--main-arg", action="append", dest="main_args", default=[],
        help="Extra main.py argument, repeatable, e.g. --main-arg=--team-possession-max-gap-seconds=1.5")
    score_parser = sub.add_parser("score", help="Score one replay label.")
    score_parser.add_argument("label")
    score_parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY_ROOT)
    score_parser.add_argument("--output", type=Path)
    multisports_parser = sub.add_parser(
        "multisports", help="Run one frozen MultiSports replay into runs/multisports-<label>.")
    multisports_parser.add_argument("label")
    multisports_parser.add_argument(
        "--source", type=Path, default=ROOT / "runs/multisports-independent-v1",
        help="Registered parent selection whose annotations, videos and stubs are reused.")
    args = parser.parse_args()
    if args.command == "run":
        run(args.label, args.replay_root, args.video_ids, args.main_args)
        return
    if args.command == "multisports":
        print(json.dumps(multisports(args.label, args.source.resolve()), indent=2))
        return
    report = score(args.replay_root / args.label)
    text = json.dumps(report, indent=2)
    (args.output or args.replay_root / args.label / "score.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

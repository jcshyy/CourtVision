import argparse
import copy
import json
import math
import pickle
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_BENCHMARK = ROOT / "benchmarks" / "courtvision_v1"
STATE_LABELS = ["controlled", "loose", "in_flight", "shot", "dead", "unknown"]
TEAM_LABELS = ["team_a", "team_b"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score cached CourtVision predictions against a labeled benchmark."
    )
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--event-tolerance", type=int, default=15)
    parser.add_argument(
        "--ball-cache-version",
        default=None,
        help="Use an explicitly generated experimental ball-track cache version.",
    )
    parser.add_argument("--video-id", action="append", dest="video_ids")
    return parser.parse_args()


def _jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_pickle(path):
    with path.open("rb") as source:
        return pickle.load(source)


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _prf(tp, fp, fn):
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
    }


def classification_summary(pairs, labels):
    confusion = {actual: Counter() for actual in labels}
    for actual, predicted in pairs:
        confusion[actual][predicted] += 1
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        per_class[label] = {
            **_prf(tp, fp, fn),
            "support": sum(confusion[label].values()),
        }
    supported = [item for item in per_class.values() if item["support"]]
    total = len(pairs)
    correct = sum(actual == predicted for actual, predicted in pairs)
    return {
        "count": total,
        "accuracy": _safe_div(correct, total),
        "macro_f1_supported": (
            statistics.mean(item["f1"] for item in supported) if supported else 0.0
        ),
        "per_class": per_class,
        "confusion": {
            actual: {predicted: confusion[actual][predicted] for predicted in labels}
            for actual in labels
        },
    }


def optimal_team_mapping(samples):
    mappings = (
        {1: "team_a", 2: "team_b"},
        {1: "team_b", 2: "team_a"},
    )
    scored = [
        (sum(mapping.get(raw_team) == actual for raw_team, actual in samples), mapping)
        for mapping in mappings
    ]
    _, mapping = max(scored, key=lambda item: (item[0], item[1][1] == "team_a"))
    return mapping


def _event_frame(event):
    for field in ("catch_frame", "release_frame", "frame_index", "start_frame"):
        if event.get(field) is not None:
            return int(event[field])
    raise ValueError(f"Event has no usable frame: {event}")


def match_events(ground_truth, predictions, tolerance):
    candidates = []
    for truth_index, truth in enumerate(ground_truth):
        for prediction_index, prediction in enumerate(predictions):
            prediction_type = prediction.get("type", prediction.get("event_type"))
            if (
                truth["video_id"] == prediction["video_id"]
                and truth["event_type"] == prediction_type
            ):
                delta = abs(_event_frame(truth) - _event_frame(prediction))
                if delta <= tolerance:
                    candidates.append((delta, truth_index, prediction_index))
    matched_truth = set()
    matched_predictions = set()
    matches = []
    for delta, truth_index, prediction_index in sorted(candidates):
        if truth_index in matched_truth or prediction_index in matched_predictions:
            continue
        matched_truth.add(truth_index)
        matched_predictions.add(prediction_index)
        matches.append((truth_index, prediction_index, delta))
    return {
        "matches": matches,
        "unmatched_truth": [
            index for index in range(len(ground_truth)) if index not in matched_truth
        ],
        "unmatched_predictions": [
            index for index in range(len(predictions))
            if index not in matched_predictions
        ],
    }


def _cache_dir(video):
    report_name = video.get("suggestion_report")
    if report_name:
        report_path = ROOT / report_name
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            relative = Path(str(report["cache_dir"]).replace("\\", "/"))
            candidate = relative if relative.is_absolute() else ROOT / relative
            if candidate.is_dir():
                return candidate
    sha_candidate = ROOT / "backend" / "stubs" / f"{video['id']}-{video['sha256']}"
    if sha_candidate.is_dir():
        return sha_candidate
    candidates = sorted((ROOT / "backend" / "stubs").glob(f"{video['id']}-*"))
    if not candidates:
        raise FileNotFoundError(f"No cache directory found for {video['id']}")
    return candidates[-1]


def _first_existing(cache_dir, preferred, fallback_pattern):
    preferred_path = cache_dir / preferred
    if preferred_path.is_file():
        return preferred_path
    fallbacks = sorted(cache_dir.glob(fallback_pattern))
    if not fallbacks:
        raise FileNotFoundError(
            f"Missing {preferred} and {fallback_pattern} in {cache_dir}"
        )
    return fallbacks[-1]


def _pipeline_predictions(video, ball_cache_version=None):
    from backend.app.analytics import (
        BallAcquisitionDetector,
        PassInterceptionDetector,
    )
    from backend.app.team_assignment import TeamAssigner
    from backend.app.detection.player_pose_detector import (
        PLAYER_POSE_CACHE_VERSION,
        attach_player_poses,
    )
    from backend.app.tracking.ball_tracker import BallTracker, BALL_TRACKING_CACHE_VERSION
    from backend.app.tracking.player_tracker import PLAYER_TRACKING_ALGORITHM_VERSION

    cache_dir = _cache_dir(video)
    player_path = _first_existing(
        cache_dir,
        f"player_track_{PLAYER_TRACKING_ALGORITHM_VERSION}.pkl",
        "player_track_stubs.pkl",
    )
    pose_path = _first_existing(
        cache_dir,
        f"player_pose_{PLAYER_POSE_CACHE_VERSION}.pkl",
        "player_pose_*.pkl",
    )
    if ball_cache_version:
        ball_path = cache_dir / f"ball_track_stubs_{ball_cache_version}.pkl"
        if not ball_path.is_file():
            raise FileNotFoundError(ball_path)
    else:
        ball_path = _first_existing(
            cache_dir,
            f"ball_track_stubs_{BALL_TRACKING_CACHE_VERSION}.pkl",
            "ball_track_stubs.pkl",
        )
    assigner = TeamAssigner(
        tracking_algorithm_version=PLAYER_TRACKING_ALGORITHM_VERSION,
    )
    assignment_path = _first_existing(
        cache_dir,
        assigner.cache_filename,
        "player_assignment_v14_*.pkl",
    )
    player_tracks = _load_pickle(player_path)
    player_poses = _load_pickle(pose_path)
    raw_ball_tracks = _load_pickle(ball_path)
    assignments = _load_pickle(assignment_path)
    expected = video["frame_count"]
    for name, values in (
        ("player tracks", player_tracks),
        ("player poses", player_poses),
        ("ball tracks", raw_ball_tracks),
        ("assignments", assignments),
    ):
        if len(values) != expected:
            raise ValueError(
                f"{video['id']} {name} has {len(values)} frames; expected {expected}"
            )

    player_tracks = attach_player_poses(player_tracks, player_poses)

    filtered_ball_tracks = BallTracker.remove_wrong_detections(
        None,
        copy.deepcopy(raw_ball_tracks),
        player_tracks=player_tracks,
    )
    ball_tracks = BallTracker.interpolate_positions(None, filtered_ball_tracks)
    source_aware_hybrid = any(
        "semantic_raw_candidates" in frame.get(1, {})
        for frame in raw_ball_tracks
    )
    semantic_ball_tracks = (
        BallTracker.build_semantic_tracks(
            copy.deepcopy(raw_ball_tracks),
            player_tracks,
            fused_tracks=ball_tracks,
        )
        if source_aware_hybrid
        else ball_tracks
    )
    acquisition_detector = BallAcquisitionDetector(fps=video["fps"])
    holder_states = acquisition_detector.detect_holder_states(
        player_tracks,
        semantic_ball_tracks,
    )
    acquisitions = [
        state["holder_id"] if state["holder_id"] is not None else -1
        for state in holder_states
    ]
    event_detector = PassInterceptionDetector(
        max_holder_gap_frames=max(1, round(video["fps"] * 0.9)),
        minimum_catch_frames=max(2, round(video["fps"] * 0.1)),
        catch_confirmation_frames=max(3, round(video["fps"])),
    )
    acquisitions = event_detector.clean_transient_control_chains(
        acquisitions,
        assignments,
        holder_states=holder_states,
    )
    events = event_detector.detect_events(
        acquisitions,
        assignments,
        holder_states=holder_states,
        ball_tracks=semantic_ball_tracks,
        player_tracks=player_tracks,
    )
    for event in events:
        event["video_id"] = video["id"]
    return {
        "cache_dir": str(cache_dir.relative_to(ROOT)),
        "ball_tracks": ball_tracks,
        "semantic_ball_tracks": semantic_ball_tracks,
        "player_tracks": player_tracks,
        "holder_states": holder_states,
        "acquisitions": acquisitions,
        "assignments": assignments,
        "events": events,
    }


def _bbox_center(frame):
    bbox = frame.get(1, {}).get("bbox")
    if not bbox or len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        return None
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _ball_summary(samples):
    visible = [sample for sample in samples if sample["visibility"] == "visible"]
    errors = [sample["error_px"] for sample in visible if sample["error_px"] is not None]
    observed_errors = [
        sample["error_px"]
        for sample in visible
        if sample["error_px"] is not None and sample["source"] == "observed"
    ]
    observed_count = sum(sample["source"] == "observed" for sample in visible)
    predicted_count = sum(sample["predicted_center"] is not None for sample in visible)
    within = {
        str(threshold): _safe_div(
            sum(error <= threshold for error in errors), len(visible)
        )
        for threshold in (25, 50, 100)
    }
    return {
        "visible_frames": len(visible),
        "observed_detection_recall": _safe_div(observed_count, len(visible)),
        "final_position_coverage": _safe_div(predicted_count, len(visible)),
        "interpolated_fraction_of_visible": _safe_div(
            predicted_count - observed_count, len(visible)
        ),
        "within_px": within,
        "center_error_px": {
            "count": len(errors),
            "mean": statistics.mean(errors) if errors else None,
            "median": statistics.median(errors) if errors else None,
            "p90": _percentile(errors, 0.9),
        },
        "observed_center_error_px": {
            "count": len(observed_errors),
            "mean": statistics.mean(observed_errors) if observed_errors else None,
            "median": statistics.median(observed_errors) if observed_errors else None,
            "p90": _percentile(observed_errors, 0.9),
        },
        "observed_rate_by_ground_truth_visibility": {
            visibility: _safe_div(
                sum(
                    sample["source"] == "observed"
                    for sample in samples
                    if sample["visibility"] == visibility
                ),
                sum(sample["visibility"] == visibility for sample in samples),
            )
            for visibility in ("visible", "occluded", "out_of_frame", "uncertain")
        },
    }


def _frame_metrics(annotations, videos, pipelines):
    state_pairs = []
    controlled_truth_prediction = []
    team_samples_by_video = defaultdict(list)
    frame_rows = []
    false_negative_reasons = Counter()
    false_positive_reasons = Counter()
    ball_samples = []
    per_video_rows = defaultdict(list)

    for annotation in annotations:
        video_id = annotation["video_id"]
        frame_index = annotation["frame_index"]
        pipeline = pipelines[video_id]
        holder_state = pipeline["holder_states"][frame_index]
        holder_id = pipeline["acquisitions"][frame_index]
        predicted_controlled = holder_id not in (-1, None)
        predicted_state = "controlled" if predicted_controlled else "loose"
        actual_state = annotation["possession"]["state"]
        actual_controlled = actual_state == "controlled"
        state_pairs.append((actual_state, predicted_state))
        controlled_truth_prediction.append((actual_controlled, predicted_controlled))
        reason = holder_state.get("reason", "unknown")
        if actual_controlled and not predicted_controlled:
            false_negative_reasons[reason] += 1
        if not actual_controlled and predicted_controlled:
            false_positive_reasons[reason] += 1

        raw_team = None
        if predicted_controlled:
            raw_team = pipeline["assignments"][frame_index].get(holder_id)
            raw_team = int(raw_team) if raw_team in (1, 2) else None
        actual_team = annotation["possession"]["team"]
        if actual_controlled and raw_team is not None:
            team_samples_by_video[video_id].append((raw_team, actual_team))

        predicted_center = _bbox_center(pipeline["ball_tracks"][frame_index])
        info = pipeline["ball_tracks"][frame_index].get(1, {})
        source = None
        if predicted_center is not None:
            source = "interpolated" if info.get("interpolated") else "observed"
        actual_center = annotation["ball"]["center_px"]
        error = None
        if actual_center is not None and predicted_center is not None:
            error = math.dist(actual_center, predicted_center)
        ball_sample = {
            "video_id": video_id,
            "frame_index": frame_index,
            "visibility": annotation["ball"]["visibility"],
            "actual_center": actual_center,
            "predicted_center": predicted_center,
            "source": source,
            "error_px": error,
        }
        ball_samples.append(ball_sample)
        row = {
            "video_id": video_id,
            "actual_state": actual_state,
            "predicted_state": predicted_state,
            "actual_controlled": actual_controlled,
            "predicted_controlled": predicted_controlled,
            "actual_team": actual_team,
            "raw_team": raw_team,
            "reason": reason,
            "ball": ball_sample,
        }
        frame_rows.append(row)
        per_video_rows[video_id].append(row)

    team_mappings = {
        video_id: optimal_team_mapping(team_samples_by_video[video_id])
        for video_id in videos
    }
    team_total = 0
    team_predicted = 0
    team_correct = 0
    for row in frame_rows:
        if not row["actual_controlled"]:
            continue
        team_total += 1
        if row["predicted_controlled"] and row["raw_team"] is not None:
            team_predicted += 1
            predicted_team = team_mappings[row["video_id"]].get(row["raw_team"])
            team_correct += predicted_team == row["actual_team"]

    controlled_tp = sum(actual and predicted for actual, predicted in controlled_truth_prediction)
    controlled_fp = sum(not actual and predicted for actual, predicted in controlled_truth_prediction)
    controlled_fn = sum(actual and not predicted for actual, predicted in controlled_truth_prediction)
    per_video = {}
    for video_id, rows in per_video_rows.items():
        actual_controlled = sum(row["actual_controlled"] for row in rows)
        predicted_team_rows = [
            row
            for row in rows
            if row["actual_controlled"]
            and row["predicted_controlled"]
            and row["raw_team"] is not None
        ]
        team_correct_rows = sum(
            team_mappings[video_id].get(row["raw_team"]) == row["actual_team"]
            for row in predicted_team_rows
        )
        tp = sum(row["actual_controlled"] and row["predicted_controlled"] for row in rows)
        fp = sum(not row["actual_controlled"] and row["predicted_controlled"] for row in rows)
        fn = sum(row["actual_controlled"] and not row["predicted_controlled"] for row in rows)
        per_video[video_id] = {
            "sample_count": len(rows),
            "ball": _ball_summary([row["ball"] for row in rows]),
            "controlled_detection": _prf(tp, fp, fn),
            "team_end_to_end_accuracy": _safe_div(team_correct_rows, actual_controlled),
            "team_prediction_coverage": _safe_div(len(predicted_team_rows), actual_controlled),
        }

    worst_ball = sorted(
        (
            sample for sample in ball_samples
            if sample["visibility"] == "visible" and sample["error_px"] is not None
        ),
        key=lambda sample: sample["error_px"],
        reverse=True,
    )[:15]
    return {
        "ball": {**_ball_summary(ball_samples), "worst_visible_errors": worst_ball},
        "possession": {
            "state": classification_summary(state_pairs, STATE_LABELS),
            "controlled_detection": _prf(controlled_tp, controlled_fp, controlled_fn),
            "false_negative_reasons": dict(false_negative_reasons.most_common()),
            "false_positive_reasons": dict(false_positive_reasons.most_common()),
            "team": {
                "ground_truth_controlled_frames": team_total,
                "prediction_coverage": _safe_div(team_predicted, team_total),
                "conditional_accuracy": _safe_div(team_correct, team_predicted),
                "end_to_end_accuracy": _safe_div(team_correct, team_total),
                "correct": team_correct,
                "predicted": team_predicted,
                "mappings": {
                    video_id: {str(key): value for key, value in mapping.items()}
                    for video_id, mapping in team_mappings.items()
                },
                "mapping_method": "best two-cluster permutation on controlled benchmark frames",
            },
        },
        "per_video": per_video,
        "team_mappings": team_mappings,
    }


def _event_metrics(ground_truth, predictions, team_mappings, tolerance):
    tolerances = sorted(set((5, tolerance, 30)))
    scores = {}
    for current_tolerance in tolerances:
        matching = match_events(ground_truth, predictions, current_tolerance)
        scores[str(current_tolerance)] = _prf(
            len(matching["matches"]),
            len(matching["unmatched_predictions"]),
            len(matching["unmatched_truth"]),
        )
    matching = match_events(ground_truth, predictions, tolerance)
    per_type = {}
    event_types = sorted(
        {event["event_type"] for event in ground_truth}
        | {event["type"] for event in predictions}
    )
    for event_type in event_types:
        truth_subset = [event for event in ground_truth if event["event_type"] == event_type]
        prediction_subset = [event for event in predictions if event["type"] == event_type]
        subset_matching = match_events(truth_subset, prediction_subset, tolerance)
        per_type[event_type] = _prf(
            len(subset_matching["matches"]),
            len(subset_matching["unmatched_predictions"]),
            len(subset_matching["unmatched_truth"]),
        )

    release_errors = []
    catch_errors = []
    team_checks = 0
    team_correct = 0
    match_records = []
    for truth_index, prediction_index, delta in matching["matches"]:
        truth = ground_truth[truth_index]
        prediction = predictions[prediction_index]
        if truth.get("release_frame") is not None and prediction.get("release_frame") is not None:
            release_errors.append(abs(truth["release_frame"] - prediction["release_frame"]))
        if truth.get("catch_frame") is not None and prediction.get("catch_frame") is not None:
            catch_errors.append(abs(truth["catch_frame"] - prediction["catch_frame"]))
        mapping = team_mappings[truth["video_id"]]
        for truth_field, prediction_field in (
            ("from_team", "from_team_id"),
            ("to_team", "to_team_id"),
        ):
            if truth.get(truth_field) is None or prediction.get(prediction_field) is None:
                continue
            team_checks += 1
            team_correct += mapping.get(int(prediction[prediction_field])) == truth[truth_field]
        match_records.append({
            "video_id": truth["video_id"],
            "event_type": truth["event_type"],
            "ground_truth_frame": _event_frame(truth),
            "predicted_frame": _event_frame(prediction),
            "absolute_frame_error": delta,
        })
    unmatched_truth = [ground_truth[index] for index in matching["unmatched_truth"]]
    unmatched_predictions = [predictions[index] for index in matching["unmatched_predictions"]]
    return {
        "ground_truth_count": len(ground_truth),
        "prediction_count": len(predictions),
        "tolerance_frames": tolerance,
        "scores_by_tolerance": scores,
        "per_type_at_default_tolerance": per_type,
        "release_timing_mae_frames": (
            statistics.mean(release_errors) if release_errors else None
        ),
        "catch_timing_mae_frames": statistics.mean(catch_errors) if catch_errors else None,
        "matched_team_accuracy": _safe_div(team_correct, team_checks),
        "matched_team_checks": team_checks,
        "matches": match_records,
        "unmatched_ground_truth": unmatched_truth,
        "unmatched_predictions": unmatched_predictions,
        "ground_truth_completeness": "candidate review plus explicitly identified misses",
    }


def _failure_summary(frame_metrics, event_metrics):
    ball = frame_metrics["ball"]
    possession = frame_metrics["possession"]
    controlled = possession["controlled_detection"]
    team = possession["team"]
    events = event_metrics["scores_by_tolerance"][str(event_metrics["tolerance_frames"])]
    state = possession["state"]
    candidates = [
        {
            "area": "ball_detection",
            "error_rate": 1 - ball["observed_detection_recall"],
            "evidence": (
                f"Only {ball['observed_detection_recall']:.1%} of visible balls have an "
                "observed post-filter detection; the rest depend on interpolation."
            ),
        },
        {
            "area": "ball_localization_50px",
            "error_rate": 1 - ball["within_px"]["50"],
            "evidence": (
                f"Only {ball['within_px']['50']:.1%} of visible balls are localized "
                "within 50 pixels after interpolation."
            ),
        },
        {
            "area": "possession_state",
            "error_rate": 1 - state["accuracy"],
            "evidence": (
                f"Six-state possession accuracy is {state['accuracy']:.1%}; the pipeline "
                "currently emits only controlled or loose for this benchmark."
            ),
        },
        {
            "area": "controlled_possession_recall",
            "error_rate": 1 - controlled["recall"],
            "evidence": f"Controlled-possession recall is {controlled['recall']:.1%}.",
        },
        {
            "area": "possession_team_end_to_end",
            "error_rate": 1 - team["end_to_end_accuracy"],
            "evidence": (
                f"End-to-end possession-team accuracy is {team['end_to_end_accuracy']:.1%} "
                f"with {team['prediction_coverage']:.1%} team coverage."
            ),
        },
        {
            "area": "events",
            "error_rate": 1 - events["f1"],
            "evidence": (
                f"Event F1 is {events['f1']:.1%} at ±{event_metrics['tolerance_frames']} "
                f"frames ({events['true_positive']} TP, {events['false_positive']} FP, "
                f"{events['false_negative']} FN)."
            ),
        },
    ]
    return sorted(candidates, key=lambda item: item["error_rate"], reverse=True)


def _pct(value):
    return f"{value * 100:.1f}%"


def _number(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _json_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _markdown(report):
    ball = report["frame_metrics"]["ball"]
    possession = report["frame_metrics"]["possession"]
    events = report["event_metrics"]
    default_events = events["scores_by_tolerance"][str(events["tolerance_frames"])]
    lines = [
        "# CourtVision v1 baseline",
        "",
        f"Scored {report['coverage']['scored_frames']} verified frames across "
        f"{report['coverage']['video_count']} videos using cached pipeline artifacts.",
        "",
        "## Headline metrics",
        "",
        "| Area | Metric | Result |",
        "|---|---|---:|",
        f"| Ball | Observed detection recall on visible frames | {_pct(ball['observed_detection_recall'])} |",
        f"| Ball | Visible frames within 50 px | {_pct(ball['within_px']['50'])} |",
        f"| Ball | Median center error | {_number(ball['center_error_px']['median'])} px |",
        f"| Ball | 90th-percentile center error | {_number(ball['center_error_px']['p90'])} px |",
        f"| Possession | Six-state accuracy | {_pct(possession['state']['accuracy'])} |",
        f"| Possession | Controlled precision / recall / F1 | {_pct(possession['controlled_detection']['precision'])} / {_pct(possession['controlled_detection']['recall'])} / {_pct(possession['controlled_detection']['f1'])} |",
        f"| Team | End-to-end / conditional accuracy | {_pct(possession['team']['end_to_end_accuracy'])} / {_pct(possession['team']['conditional_accuracy'])} |",
        f"| Events | Precision / recall / F1 at ±{events['tolerance_frames']} frames | {_pct(default_events['precision'])} / {_pct(default_events['recall'])} / {_pct(default_events['f1'])} |",
        "",
        "## Largest measured failures",
        "",
    ]
    for index, failure in enumerate(report["largest_failures"], 1):
        lines.append(f"{index}. **{failure['area']}** — {failure['evidence']}")
    top_false_negative_reason = next(
        iter(possession["false_negative_reasons"].items()),
        ("none", 0),
    )
    lines.extend([
        "",
        "## Root-cause signals",
        "",
        f"- {top_false_negative_reason[1]} controlled-possession misses are attributed to `{top_false_negative_reason[0]}`.",
        f"- Team assignment is {_pct(possession['team']['conditional_accuracy'])} correct when the pipeline actually identifies a holder; low end-to-end team accuracy is therefore a holder-coverage problem.",
        f"- Ball localization has a {_number(ball['center_error_px']['median'])} px median but a {_number(ball['center_error_px']['p90'])} px p90, showing a severe wrong-object/interpolation tail.",
        f"- Shot event recall is {_pct(events['per_type_at_default_tolerance'].get('shot', {}).get('recall', 0.0))}; the current event detector emits only passes and interceptions.",
    ])
    lines.extend([
        "",
        "## Per-video frame metrics",
        "",
        "| Video | Samples | Ball observed recall | Ball within 50 px | Controlled F1 | Team end-to-end |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for video_id, metrics in report["frame_metrics"]["per_video"].items():
        lines.append(
            f"| {video_id} | {metrics['sample_count']} | "
            f"{_pct(metrics['ball']['observed_detection_recall'])} | "
            f"{_pct(metrics['ball']['within_px']['50'])} | "
            f"{_pct(metrics['controlled_detection']['f1'])} | "
            f"{_pct(metrics['team_end_to_end_accuracy'])} |"
        )
    lines.extend([
        "",
        "## Event results",
        "",
        "| Type | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for event_type, metrics in events["per_type_at_default_tolerance"].items():
        lines.append(
            f"| {event_type} | {metrics['true_positive']} | {metrics['false_positive']} | "
            f"{metrics['false_negative']} | {_pct(metrics['precision'])} | "
            f"{_pct(metrics['recall'])} | {_pct(metrics['f1'])} |"
        )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- Team IDs are arbitrary clusters, so the scorer selects the better of the two possible per-video cluster mappings.",
        "- Event labels contain reviewed pipeline candidates plus explicitly identified misses; they are not yet a complete play-by-play inventory.",
        "- The current possession model does not predict in-flight, shot, dead-ball, or unknown states, so six-state scoring exposes those as unsupported classes.",
        "- Interpolated ball positions count toward localization but not observed-detection recall.",
        "",
    ])
    return "\n".join(lines)


def score(
    benchmark_dir,
    event_tolerance=15,
    ball_cache_version=None,
    video_ids=None,
):
    dataset = json.loads((benchmark_dir / "dataset.json").read_text(encoding="utf-8"))
    videos = {
        video["id"]: video
        for video in dataset["videos"]
        if not video_ids or video["id"] in video_ids
    }
    if not videos:
        raise ValueError("No benchmark videos matched --video-id")
    annotations = [
        record for record in _jsonl(benchmark_dir / "annotations.jsonl")
        if record.get("review_status") == "verified"
        and record["video_id"] in videos
    ]
    ground_truth_events = [
        event for event in _jsonl(benchmark_dir / "events.jsonl")
        if event.get("review_status") == "verified"
        and event["video_id"] in videos
    ]
    pipelines = {
        video_id: _pipeline_predictions(video, ball_cache_version)
        for video_id, video in videos.items()
    }
    frame_metrics = _frame_metrics(annotations, videos, pipelines)
    predicted_events = [
        event
        for video_id in videos
        for event in pipelines[video_id]["events"]
    ]
    event_metrics = _event_metrics(
        ground_truth_events,
        predicted_events,
        frame_metrics.pop("team_mappings"),
        event_tolerance,
    )
    report = {
        "benchmark_id": dataset["benchmark_id"],
        "baseline": (
            f"experimental ball cache {ball_cache_version}"
            if ball_cache_version
            else "current cached pipeline"
        ),
        "coverage": {
            "video_count": len(videos),
            "scored_frames": len(annotations),
            "ground_truth_events": len(ground_truth_events),
            "predicted_events": len(predicted_events),
            "cache_dirs": {
                video_id: pipelines[video_id]["cache_dir"] for video_id in videos
            },
        },
        "frame_metrics": frame_metrics,
        "event_metrics": event_metrics,
    }
    report["largest_failures"] = _failure_summary(frame_metrics, event_metrics)
    return report


def main():
    args = parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    output_json = (args.output_json or benchmark_dir / "baseline_report.json").resolve()
    output_markdown = (
        args.output_markdown or benchmark_dir / "baseline_report.md"
    ).resolve()
    report = score(
        benchmark_dir,
        event_tolerance=args.event_tolerance,
        ball_cache_version=args.ball_cache_version,
        video_ids=set(args.video_ids or ()),
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, default=_json_scalar) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_markdown}")


if __name__ == "__main__":
    main()

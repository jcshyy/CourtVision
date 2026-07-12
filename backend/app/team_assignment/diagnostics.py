import json
import logging
import pickle
import struct
from pathlib import Path

from backend.app.cache_paths import cache_path, video_cache_dir


logger = logging.getLogger(__name__)
DEFAULT_ASSIGNMENT_CACHE_NAME = "player_assignment_stub_v3.pkl"


def diagnose_team_track(
    video_path,
    cache_root,
    track_id,
    assignment_cache_name=None,
    track_cache_name="player_track_stubs.pkl",
    cache_dir_override=None,
    cache_only=False,
    team_1_color=None,
    team_2_color=None,
):
    video_path = Path(video_path)
    cache_dir = (
        Path(cache_dir_override)
        if cache_dir_override is not None
        else video_cache_dir(cache_root, video_path)
    )
    track_cache = cache_path(cache_dir, track_cache_name)
    assignment_cache = cache_path(
        cache_dir,
        assignment_cache_name or DEFAULT_ASSIGNMENT_CACHE_NAME,
    )

    player_tracks = _load_diagnostic_cache(track_cache)
    cached_assignments = (
        _load_diagnostic_cache(assignment_cache)
        if assignment_cache.exists()
        else None
    )
    if cache_only:
        return _diagnose_cached_track(
            video_path,
            cache_dir,
            player_tracks,
            cached_assignments,
            track_id,
        )

    import cv2

    from backend.app.team_assignment.team_assigner import (
        TeamAssigner,
        _discover_team_colors_result,
        _confident_nearest_team,
        _jersey_observation,
        _nearest_team,
        _team_distances,
        _track_team_decision,
    )

    frames = _read_video(video_path)
    if len(player_tracks) != len(frames):
        raise ValueError(
            f"Frame/cache length mismatch: video={len(frames)}, "
            f"player_tracks={len(player_tracks)}"
        )

    visible_frames = [
        frame_index
        for frame_index, frame_tracks in enumerate(player_tracks)
        if track_id in frame_tracks
    ]
    if not visible_frames:
        raise ValueError(f"Track {track_id} does not appear in the cached tracks")

    assigner = TeamAssigner(
        team_1_color=team_1_color,
        team_2_color=team_2_color,
    )
    discovery = None
    if assigner.assignment_mode == "automatic":
        discovery = _discover_team_colors_result(frames, player_tracks)
        if discovery["status"] != "confident":
            return {
                "mode": "pixel",
                "status": "needs_team_colors",
                "video": str(video_path),
                "cache_dir": str(cache_dir),
                "track_id": track_id,
                "assignment_mode": assigner.assignment_mode,
                "normalized_team_colors": None,
                "discovery_result": discovery,
                "visible_frame_count": len(visible_frames),
                "visible_frame_range": [visible_frames[0], visible_frames[-1]],
                "observations": [],
            }
        discovered_colors = discovery["prototypes"]
        assigner.team_colors = {1: discovered_colors[0], 2: discovered_colors[1]}

    observations = []
    for frame_index, (frame, frame_tracks) in enumerate(zip(frames, player_tracks)):
        track = frame_tracks.get(track_id)
        if track is None:
            continue

        observation = _jersey_observation(frame, track["bbox"])
        feature = observation["feature"]
        distances = _team_distances(feature, assigner.team_colors)
        numeric_distances = [
            distance for distance in distances.values() if distance is not None
        ]
        margin = (
            abs(numeric_distances[0] - numeric_distances[1])
            if len(numeric_distances) == 2
            else None
        )
        nearest_team = (
            _nearest_team(feature, assigner.team_colors)
            if feature is not None
            else None
        )
        assignment_team = _confident_nearest_team(feature, assigner.team_colors)
        observations.append(
            {
                "frame": frame_index,
                "bbox": [float(value) for value in track["bbox"]],
                **observation,
                "prototype_distances": distances,
                "distance_margin": margin,
                "nearest_team": nearest_team,
                "assignment_accepted": assignment_team is not None,
                "assignment_team": assignment_team,
                "bootstrap_selected": False,
                "cached_team": _cached_team(
                    cached_assignments,
                    frame_index,
                    track_id,
                ),
            }
        )

    if not observations:
        raise ValueError(f"Track {track_id} does not appear in the cached tracks")

    track_decision = _track_team_decision(observations, assigner.team_colors)
    selected_frames = set(track_decision["selected_evidence_frames"])
    bootstrap_team = track_decision["team_id"]
    for observation in observations:
        observation["bootstrap_selected"] = observation["frame"] in selected_frames
    assigner.player_team_dict[track_id] = (
        bootstrap_team if bootstrap_team is not None else -1
    )
    assigner.player_team_votes[track_id] = (
        [bootstrap_team] if bootstrap_team is not None else []
    )

    for observation in observations:
        computed_team = assigner.player_team_dict[track_id]
        observation["refresh_frame"] = False
        observation["votes"] = list(assigner.player_team_votes.get(track_id, []))
        observation["computed_team"] = computed_team

    return {
        "mode": "pixel",
        "status": "ok",
        "video": str(video_path),
        "cache_dir": str(cache_dir),
        "track_id": track_id,
        "assignment_mode": assigner.assignment_mode,
        "normalized_team_colors": assigner.normalized_team_colors,
        "discovery_result": discovery,
        "team_prototypes": assigner.team_colors,
        "bootstrap_feature": None,
        "bootstrap_team": bootstrap_team,
        "track_decision": track_decision,
        "visible_frame_count": len(observations),
        "visible_frame_range": [
            observations[0]["frame"],
            observations[-1]["frame"],
        ],
        "decision_summary": _pixel_decision_summary(observations),
        "observations": observations,
    }


def write_diagnostic(diagnostic, output_path=None):
    serialized = json.dumps(diagnostic, indent=2)
    if output_path is None:
        print(serialized)
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")
    logger.info("Wrote team-track diagnostic to %s", path.resolve())


def _read_video(video_path):
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame)
    capture.release()
    return frames


def _cached_team(assignments, frame_index, track_id):
    if assignments is None or frame_index >= len(assignments):
        return None
    return assignments[frame_index].get(track_id)


def _diagnose_cached_track(
    video_path,
    cache_dir,
    player_tracks,
    cached_assignments,
    track_id,
):
    observations = []
    for frame_index, frame_tracks in enumerate(player_tracks):
        track = frame_tracks.get(track_id)
        if track is None:
            continue
        observations.append(
            {
                "frame": frame_index,
                "bbox": [float(value) for value in track["bbox"]],
                "cached_team": _cached_team(
                    cached_assignments,
                    frame_index,
                    track_id,
                ),
            }
        )

    if not observations:
        raise ValueError(f"Track {track_id} does not appear in the cached tracks")

    return {
        "mode": "cache_only",
        "video": str(video_path),
        "cache_dir": str(cache_dir),
        "track_id": track_id,
        "frame_count": len(player_tracks),
        "assignment_frame_count": (
            len(cached_assignments) if cached_assignments is not None else None
        ),
        "visible_frame_count": len(observations),
        "visible_frame_range": [
            observations[0]["frame"],
            observations[-1]["frame"],
        ],
        "assignment_transitions": _assignment_transitions(observations),
        "decision_summary": {
            "cached_assignment_transitions": _assignment_transitions(observations),
            "unassigned_visible_frames": [
                observation["frame"]
                for observation in observations
                if observation["cached_team"] is None
            ],
        },
        "pixel_diagnostics_available": False,
        "observations": observations,
    }


def _assignment_transitions(observations):
    transitions = []
    previous_team = None
    for observation in observations:
        team = observation["cached_team"]
        if team is None:
            continue
        if team != previous_team:
            transitions.append(
                {
                    "frame": observation["frame"],
                    "team": team,
                }
            )
            previous_team = team
    return transitions


def _pixel_decision_summary(observations):
    accepted = [
        observation for observation in observations if observation["accepted"]
    ]
    rejected = [
        observation for observation in observations if not observation["accepted"]
    ]
    rejection_counts = {}
    for observation in rejected:
        reason = observation["rejection_reason"] or "unspecified"
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    accepted_with_margin = [
        observation
        for observation in accepted
        if observation["distance_margin"] is not None
    ]
    weakest_margins = sorted(
        accepted_with_margin,
        key=lambda observation: observation["distance_margin"],
    )[:5]

    return {
        "acceptance_policy": (
            "minimum_torso_area_and_nonempty_feature_with_"
            "normalized_prototype_margin"
        ),
        "measured_but_not_rejected": [
            "crop_fraction",
            "blur_variance",
            "visible_fraction",
            "filtered_fraction",
        ],
        "ambiguous_assignment_observation_count": sum(
            not observation.get("assignment_accepted", False)
            for observation in accepted
        ),
        "accepted_observation_count": len(accepted),
        "rejected_observation_count": len(rejected),
        "rejection_reason_counts": rejection_counts,
        "bootstrap_frames": [
            observation["frame"]
            for observation in observations
            if observation["bootstrap_selected"]
        ],
        "nearest_team_transitions": _value_transitions(
            observations,
            "nearest_team",
        ),
        "computed_team_transitions": _value_transitions(
            observations,
            "computed_team",
        ),
        "cached_assignment_transitions": _value_transitions(
            observations,
            "cached_team",
        ),
        "weakest_accepted_margins": [
            {
                "frame": observation["frame"],
                "distance_margin": observation["distance_margin"],
                "nearest_team": observation["nearest_team"],
                "prototype_distances": observation["prototype_distances"],
            }
            for observation in weakest_margins
        ],
    }


def _value_transitions(observations, key):
    transitions = []
    previous_value = object()
    for observation in observations:
        value = observation.get(key)
        if value is None:
            continue
        if not transitions or value != previous_value:
            transitions.append(
                {
                    "frame": observation["frame"],
                    key: value,
                }
            )
            previous_value = value
    return transitions


def _load_diagnostic_cache(path):
    path = Path(path)
    logger.info("Loading diagnostic cache: %s", path.resolve())
    try:
        with path.open("rb") as cache_file:
            return pickle.load(cache_file)
    except ModuleNotFoundError as error:
        if error.name != "numpy":
            raise

    logger.warning(
        "NumPy is unavailable; loading trusted local cache %s with scalar "
        "compatibility mode",
        path.resolve(),
    )
    with path.open("rb") as cache_file:
        return _NumpyScalarUnpickler(cache_file).load()


class _NumpyDtype:
    def __init__(self, code, *_args):
        self.code = code

    def __setstate__(self, _state):
        return None


def _numpy_scalar(dtype, raw_value):
    formats = {
        "i4": "i",
        "i8": "q",
        "u4": "I",
        "u8": "Q",
        "f4": "f",
        "f8": "d",
    }
    try:
        format_code = formats[dtype.code]
    except KeyError as error:
        raise pickle.UnpicklingError(
            f"Unsupported NumPy scalar dtype in diagnostic cache: {dtype.code}"
        ) from error
    return struct.unpack(f"<{format_code}", raw_value)[0]


class _NumpyScalarUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy" and name == "dtype":
            return _NumpyDtype
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"}:
            if name == "scalar":
                return _numpy_scalar
        return super().find_class(module, name)

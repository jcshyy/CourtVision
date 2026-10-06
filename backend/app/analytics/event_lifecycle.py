"""Final event arbitration; holder changes remain provisional until this stage."""
import math
from itertools import combinations

from .pass_interception import merge_corroborated_pass_events
from .shot_rebound import reconcile_shot_events


def finalize_ball_events(semantic_events, fused_events, shot_timeline, player_tracks,
                         *, fps=30, ball_tracks=None, court_keypoints=None,
                         discontinuity_frames=None):
    candidates = merge_corroborated_pass_events(
        semantic_events, fused_events, player_tracks,
        duplicate_window_frames=max(3, round(fps * 0.25)),
    )
    events = reconcile_shot_events(candidates, shot_timeline)
    retained = []
    cuts = set(discontinuity_frames or [])
    for event in events:
        release = int(event.get("release_frame", event["frame_index"]))
        end = int(event.get("catch_frame", event["frame_index"]))
        if any(release < cut <= end for cut in cuts):
            reason = "scene_discontinuity"
        elif _is_throw_in(event, player_tracks, ball_tracks, court_keypoints,
                          max(3, round(fps * 0.1))):
            reason = "throw_in"
        else:
            retained.append(event)
            continue
        shot_timeline.arbitration.append({
            "event": dict(event), "decision": "excluded_from_public_totals",
            "reason": reason,
        })
    return retained


def _is_throw_in(event, players, balls, keypoints, support_frames):
    """Require a held ball outside an observed endline, then an inside catch.

    This does not infer a score/outcome, does not use fallback homographies, and
    abstains when court geometry or visible ball support is unavailable.
    """
    if event.get("type") != "pass" or keypoints is None or balls is None:
        return False
    release = event.get("release_frame")
    catch = event.get("catch_frame", event.get("frame_index"))
    if release is None or catch is None or release < support_frames - 1 or catch <= release:
        return False
    if catch >= min(len(players), len(balls), len(keypoints)):
        return False
    for side in ("left", "right"):
        observed_ball_frames = 0
        for frame in range(release - support_frames + 1, release + 1):
            source = players[frame].get(event.get("from_player_id"), {}).get("bbox")
            line = _endline(keypoints[frame], side)
            if source is None or line is None:
                break
            height = source[3] - source[1]
            foot = ((source[0] + source[2]) / 2, source[3])
            if height <= 0 or _inside_distance(foot, line) >= -max(3, height * 0.04):
                break
            ball = balls[frame].get(1, {})
            box = ball.get("bbox")
            if box is not None and not ball.get("interpolated", False):
                x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                margin = height * 0.15
                if not (source[0] - margin <= x <= source[2] + margin
                        and source[1] - margin <= y <= source[3]):
                    break
                observed_ball_frames += 1
        else:
            receiver = players[catch].get(event.get("to_player_id"), {}).get("bbox")
            line = _endline(keypoints[catch], side)
            if receiver is None or line is None or observed_ball_frames < 2:
                continue
            foot = ((receiver[0] + receiver[2]) / 2, receiver[3])
            if _inside_distance(foot, line) > max(3, (receiver[3] - receiver[1]) * 0.04):
                return True
    return False


def _endline(keypoints, side):
    if isinstance(keypoints, dict):
        points = keypoints.get("points", [])
        confidences = keypoints.get("confidence", [])
    elif keypoints is not None and getattr(keypoints, "conf", None) is not None:
        point_batches = keypoints.xy.cpu().tolist()
        confidence_batches = keypoints.conf.cpu().tolist()
        if not point_batches or not confidence_batches:
            return None
        points = point_batches[0]
        confidences = confidence_batches[0]
    else:
        return None
    valid = {i: point for i, point in enumerate(points)
             if i < len(confidences) and confidences[i] >= 0.65
             and len(point) >= 2 and all(math.isfinite(float(v)) and v > 0 for v in point[:2])}
    boundary = range(6) if side == "left" else range(10, 16)
    interior = (8, 9, 6, 7) if side == "left" else (16, 17, 6, 7)
    pairs = [(valid[i], valid[j]) for i, j in combinations(boundary, 2)
             if i in valid and j in valid]
    reference = next((valid[i] for i in interior if i in valid), None)
    if not pairs or reference is None:
        return None
    a, b = max(pairs, key=lambda pair: math.dist(*pair))
    if math.dist(a, b) < 50:
        return None
    cross = (b[0] - a[0]) * (reference[1] - a[1]) - (b[1] - a[1]) * (reference[0] - a[0])
    if abs(cross) / math.dist(a, b) < 15:
        return None
    return a, b, 1 if cross > 0 else -1


def _inside_distance(point, line):
    a, b, direction = line
    return direction * ((b[0] - a[0]) * (point[1] - a[1])
                        - (b[1] - a[1]) * (point[0] - a[0])) / math.dist(a, b)

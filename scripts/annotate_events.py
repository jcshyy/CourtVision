import argparse
import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "benchmarks" / "courtvision_v1"
EVENT_KEYS = {
    ord("p"): "pass",
    ord("t"): "steal",
    ord("i"): "interception",
    ord("o"): "offensive_rebound",
    ord("e"): "defensive_rebound",
    ord("s"): "shot",
    ord("f"): "deflection",
    ord("k"): "dead_ball",
    ord("u"): "unknown_change",
}


def _load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _save_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def event_is_complete(event):
    event_type = event.get("event_type")
    if event_type in {"pass", "steal", "interception"}:
        return (
            event.get("release_frame") is not None
            and event.get("catch_frame") is not None
            and event["release_frame"] <= event["catch_frame"]
        )
    if event_type == "shot":
        return (
            event.get("release_frame") is not None
            and event.get("catch_frame") is None
            and event.get("to_team") is None
        )
    if event_type in {"offensive_rebound", "defensive_rebound"}:
        return (
            event.get("release_frame") is None
            and event.get("catch_frame") is not None
            and event.get("from_team") is None
        )
    return True


def set_event_type(event, event_type):
    event["event_type"] = event_type
    if event_type == "shot":
        event["catch_frame"] = None
        event["to_team"] = None
    elif event_type in {"offensive_rebound", "defensive_rebound"}:
        event["release_frame"] = None
        event["from_team"] = None


def next_pending_index(events, current_index):
    order = list(range(current_index + 1, len(events)))
    order.extend(range(0, current_index))
    return next(
        (index for index in order if events[index].get("review_status") == "pending"),
        len(events),
    )


def _read_frame(capture, frame_index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Could not read video frame {frame_index}")
    return frame


def _draw_header(frame, event, index, total, frame_index, teams):
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 86), (0, 0, 0), -1)
    first = (
        f"event {index + 1}/{total} {event['video_id']} frame={frame_index} "
        f"range={event['start_frame']}-{event['end_frame']} status={event['review_status']}"
    )
    second = (
        f"type={event['event_type']} release={event.get('release_frame')} "
        f"catch={event.get('catch_frame')} from={event.get('from_team')} "
        f"to={event.get('to_team')}"
    )
    third = (
        f"1/2 from: {teams['team_a']['name']} / {teams['team_b']['name']}   "
        f"3/4 to: {teams['team_a']['name']} / {teams['team_b']['name']}"
    )
    for y, text in ((24, first), (50, second), (76, third)):
        cv2.putText(
            output, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    return output


def annotate(benchmark_dir):
    manifest = json.loads((benchmark_dir / "manifest.json").read_text(encoding="utf-8"))
    videos = {video["id"]: video for video in manifest["videos"]}
    event_path = benchmark_dir / "events.jsonl"
    events = _load_jsonl(event_path)
    if not events:
        print("No event candidates found.")
        return
    index = next(
        (i for i, event in enumerate(events) if event.get("review_status") == "pending"),
        0,
    )
    capture = None
    active_video_id = None
    frame_count = 0
    frame_index = 0
    window = "CourtVision event review"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while 0 <= index < len(events):
        event = events[index]
        video = videos[event["video_id"]]
        if active_video_id != event["video_id"]:
            if capture is not None:
                capture.release()
            capture = cv2.VideoCapture(str(ROOT / video["path"]))
            if not capture.isOpened():
                raise ValueError(f"Could not open {video['path']}")
            active_video_id = event["video_id"]
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = min(
            frame_count - 1,
            max(0, event.get("release_frame") or event["start_frame"]),
        )
        while True:
            frame = _read_frame(capture, frame_index)
            cv2.imshow(
                window,
                _draw_header(frame, event, index, len(events), frame_index, video["teams"]),
            )
            key = cv2.waitKey(0) & 0xFF
            if key == ord("a"):
                frame_index = max(0, frame_index - 1)
            elif key == ord("d"):
                frame_index = min(frame_count - 1, frame_index + 1)
            elif key == ord("r"):
                event["release_frame"] = frame_index
            elif key == ord("R"):
                event["release_frame"] = None
            elif key == ord("c"):
                event["catch_frame"] = frame_index
            elif key == ord("C"):
                event["catch_frame"] = None
            elif key in EVENT_KEYS:
                set_event_type(event, EVENT_KEYS[key])
            elif key == ord("1"):
                event["from_team"] = "team_a"
            elif key == ord("2"):
                event["from_team"] = "team_b"
            elif key == ord("3"):
                event["to_team"] = "team_a"
            elif key == ord("4"):
                event["to_team"] = "team_b"
            elif key == ord("0"):
                event["from_team"] = None
                event["to_team"] = None
            elif key == ord("v"):
                if not event_is_complete(event):
                    print(
                        "Event is incomplete: passes/steals/interceptions need release "
                        "and catch; shots need release only; rebounds need catch only."
                    )
                    continue
                event["review_status"] = "verified"
                _save_jsonl(event_path, events)
                index = next_pending_index(events, index)
                break
            elif key == ord("j"):
                event["review_status"] = "rejected"
                _save_jsonl(event_path, events)
                index = next_pending_index(events, index)
                break
            elif key == ord("["):
                _save_jsonl(event_path, events)
                index = max(0, index - 1)
                break
            elif key == ord("]"):
                _save_jsonl(event_path, events)
                index = min(len(events) - 1, index + 1)
                break
            elif key == ord("q"):
                _save_jsonl(event_path, events)
                capture.release()
                cv2.destroyAllWindows()
                return
    _save_jsonl(event_path, events)
    if capture is not None:
        capture.release()
    cv2.destroyAllWindows()
    print("All event candidates reviewed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review CourtVision event candidates.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    annotate(parser.parse_args().benchmark_dir.resolve())

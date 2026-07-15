import argparse
import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "benchmarks" / "courtvision_v1"


def _load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _save(path, records):
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def annotate(benchmark_dir):
    path = benchmark_dir / "annotations.jsonl"
    records = _load(path)
    index = next((i for i, item in enumerate(records) if item["review_status"] != "verified"), 0)
    window = "CourtVision benchmark"
    click = {"point": None}

    def on_mouse(event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["point"] = [int(x), int(y)]

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while 0 <= index < len(records):
        record = records[index]
        image = cv2.imread(str(benchmark_dir / record["image_path"]))
        if image is None:
            raise ValueError(f"Could not read {record['image_path']}")
        click["point"] = record["ball"]["center_px"]
        while True:
            display = image.copy()
            if click["point"] is not None:
                cv2.drawMarker(display, tuple(click["point"]), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            label = (
                f"{index + 1}/{len(records)} {record['video_id']} f{record['frame_index']} "
                f"ball={record['ball']['visibility']} possession={record['possession']['state']} "
                f"team={record['possession']['team']} status={record['review_status']}"
            )
            cv2.rectangle(display, (0, 0), (display.shape[1], 38), (0, 0, 0), -1)
            cv2.putText(display, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, display)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("v"):
                record["ball"] = {"visibility": "visible", "center_px": click["point"]}
            elif key == ord("o"):
                record["ball"] = {"visibility": "occluded", "center_px": None}
            elif key == ord("x"):
                record["ball"] = {"visibility": "out_of_frame", "center_px": None}
            elif key == ord("u"):
                record["ball"] = {"visibility": "uncertain", "center_px": None}
            elif key == ord("1"):
                record["possession"] = {"state": "controlled", "team": "team_a"}
            elif key == ord("2"):
                record["possession"] = {"state": "controlled", "team": "team_b"}
            elif key in map(ord, "lfsdn"):
                states = {ord("l"): "loose", ord("f"): "in_flight", ord("s"): "shot", ord("d"): "dead", ord("n"): "unknown"}
                record["possession"] = {"state": states[key], "team": None}
            elif key in (10, 13):
                if record["ball"]["visibility"] == "visible" and click["point"] is None:
                    print("Click the ball before verifying a visible label.")
                    continue
                record["ball"]["center_px"] = click["point"] if record["ball"]["visibility"] == "visible" else None
                record["review_status"] = "verified"
                _save(path, records)
                index += 1
                break
            elif key == ord("b"):
                _save(path, records)
                index = max(0, index - 1)
                break
            elif key == ord("q"):
                _save(path, records)
                cv2.destroyAllWindows()
                return
    _save(path, records)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively label benchmark frames.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    annotate(parser.parse_args().benchmark_dir.resolve())

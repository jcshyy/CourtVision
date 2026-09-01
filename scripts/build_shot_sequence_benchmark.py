import ast
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "shot_sequences_v1"

SELECTED = {
    "bkn-vs-hou-0022300469_366.mp4": "calibration",
    "bkn-vs-hou-0022300469_377.mp4": "holdout",
    "bkn-vs-hou-0022300469_382.mp4": "holdout",
    "bos-vs-ind-0022300493_366.mp4": "holdout",
    "bos-vs-ind-0022300507_351.mp4": "holdout",
    "det-vs-okc-0022401108_24.mp4": "calibration",
    "det-vs-okc-0022401108_28.mp4": "holdout",
    "det-vs-okc-0022401108_16.mp4": "holdout",
    "gsw-vs-mem-0022401100_9.mp4": "calibration",
    "gsw-vs-mem-0022401100_17.mp4": "calibration",
    "gsw-vs-mem-0022401100_21.mp4": "holdout",
    "cha-vs-ind-0022401104_29.mp4": "holdout",
    "cha-vs-ind-0022401104_9.mp4": "calibration",
    "cha-vs-ind-0022401104_5.mp4": "calibration",
    "cha-vs-ind-0022401104_1.mp4": "holdout",
    "mem-vs-det-0022401129_25.mp4": "calibration",
    "mia-vs-bos-0022401106_8.mp4": "calibration",
    "mia-vs-bos-0022401106_24.mp4": "holdout",
    "min-vs-bkn-0022401113_29.mp4": "calibration",
}

INPUT_ALIASES = {
    "bkn-vs-hou-0022300469_366.mp4": "bard_bkn_hou_366",
    "bos-vs-ind-0022300493_366.mp4": "bard_bos_ind_366",
    "det-vs-okc-0022401108_24.mp4": "bard_det_okc_24",
    "gsw-vs-mem-0022401100_9.mp4": "bard_gsw_mem_9",
    "gsw-vs-mem-0022401100_17.mp4": "bard_gsw_mem_17",
    "gsw-vs-mem-0022401100_21.mp4": "bard_gsw_mem_21",
}

# Visual review labels refine BARD's 2PT/3PT annotation into the trajectory
# families used for threshold reporting. Block and putback take precedence.
TWO_POINT_STYLE = {
    "bkn-vs-hou-0022300469_377.mp4": "jump_shot",
    "bkn-vs-hou-0022300469_382.mp4": "layup",
    "bos-vs-ind-0022300493_366.mp4": "layup",
    "det-vs-okc-0022401108_16.mp4": "layup",
    "cha-vs-ind-0022401104_9.mp4": "layup",
    "cha-vs-ind-0022401104_1.mp4": "layup",
    "mia-vs-bos-0022401106_8.mp4": "layup",
}


def _color(action):
    return action.get("color") or action.get("jersey_color")


def _source_rows():
    sources = [
        ROOT / "holdout_sources/BARD_repo/validation/2024/benchmark.csv",
        ROOT
        / "holdout_sources/BARD_repo/validation/2025/multi/validazione_multilabelling.csv",
    ]
    rows = {}
    for source in sources:
        with source.open(encoding="utf-8-sig", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                name = Path(row["files"]).name
                if name not in SELECTED:
                    continue
                rows[name] = {
                    "actions": ast.literal_eval(row["actions_name"]),
                    "label_source": source.relative_to(ROOT).as_posix(),
                    "source_video": (
                        source.parent / Path(row["files"]).name
                        if source.parent.name == "multi"
                        else source.parent / "multi" / Path(row["files"]).name
                    ),
                }
    return rows


def _sequences(name, row):
    actions = row["actions"]
    shots = [index for index, action in enumerate(actions) if "Shot" in action["action"]]
    previous_rebound = None
    for sequence_index, action_index in enumerate(shots, start=1):
        action = actions[action_index]
        next_index = shots[sequence_index] if sequence_index < len(shots) else len(actions)
        following = actions[action_index + 1 : next_index]
        block = next((item for item in following if item["action"] == "Block"), None)
        rebound = next((item for item in following if item["action"] == "Rebound"), None)
        putback = (
            previous_rebound is not None
            and _color(previous_rebound) == _color(action)
        )
        if putback:
            style = "putback"
        elif block is not None:
            style = "blocked_shot"
        elif action["action"] == "3PT Shot":
            style = "jump_shot"
        else:
            style = TWO_POINT_STYLE.get(name, "close_range_or_midrange")
        rebound_type = None
        if rebound is not None:
            rebound_type = (
                "offensive_rebound"
                if _color(rebound) == _color(action)
                else "defensive_rebound"
            )
        alias = INPUT_ALIASES.get(name, Path(name).stem.replace("-", "_"))
        yield {
            "sequence_id": f"{Path(name).stem}-s{sequence_index}",
            "video_id": alias,
            "source_video": row["source_video"].relative_to(ROOT).as_posix(),
            "sequence_index": sequence_index,
            "split": SELECTED[name],
            "shot_value": 3 if action["action"] == "3PT Shot" else 2,
            "shot_style": style,
            "outcome": "made" if action["result"] else "missed",
            "blocked": block is not None,
            "rebound_type": rebound_type,
            "shooter_label": action.get("player"),
            "shooter_color": _color(action),
            "rebounder_label": rebound.get("player") if rebound else None,
            "rebounder_color": _color(rebound) if rebound else None,
            "release_frame": None,
            "rim_frame": None,
            "resolution_frame": None,
            "temporal_status": "clip_level_ordered",
            "review_status": "verified_bard_label_and_visual_style",
            "label_source": row["label_source"],
        }
        previous_rebound = rebound


def main():
    rows = _source_rows()
    missing = sorted(set(SELECTED) - set(rows))
    if missing:
        raise ValueError(f"Missing BARD rows: {missing}")
    records = [
        sequence
        for name in SELECTED
        for sequence in _sequences(name, rows[name])
    ]
    if not 20 <= len(records) <= 30:
        raise ValueError(f"Expected 20-30 sequences, found {len(records)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    annotation_path = OUTPUT / "sequences.jsonl"
    annotation_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "name": "CourtVision shot sequences v1",
        "sequence_count": len(records),
        "clip_count": len(SELECTED),
        "splits": {
            split: sum(record["split"] == split for record in records)
            for split in ("calibration", "holdout")
        },
        "styles": {
            style: sum(record["shot_style"] == style for record in records)
            for style in sorted({record["shot_style"] for record in records})
        },
        "annotations": "sequences.jsonl",
        "note": (
            "BARD provides ordered clip-level action labels, not event frames. "
            "Temporal fields remain null until frame-by-frame review."
        ),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

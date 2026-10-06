"""Replay the local BARD inputs with fixed settings, then score shot counts.

Run before and after event changes into different output directories. Detector
caches are reused by main.py; all event logic is executed afresh. This is not a
temporal shot/pass benchmark (see evaluate_event_sequences.py for that).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--video-id", action="append")
    parser.add_argument("--frozen-team-cache-tag", help="Replay a specific existing team-cache fingerprint to isolate event logic")
    args = parser.parse_args()
    import main as pipeline
    from scripts.evaluate_shot_sequences import evaluate, DEFAULT_BENCHMARK
    if args.frozen_team_cache_tag:
        from backend.app.utils.cache import load_cache
        tag = args.frozen_team_cache_tag
        if not tag.isalnum():
            parser.error("The cache tag must be an alphanumeric fingerprint")

        def frozen_teams(assigner, frames, player_tracks, *, cache_path, **kwargs):
            directory = Path(cache_path).parent
            matches = list(directory.glob(f"player_assignment_*_automatic_{tag}.pkl"))
            if len(matches) != 1:
                raise ValueError(f"Expected one frozen team cache in {directory}, found {len(matches)}")
            metadata = matches[0].with_name(matches[0].name.replace("player_assignment_", "team_assignment_")).with_suffix(".json")
            assigner.assignment_metadata = json.loads(metadata.read_text(encoding="utf-8"))
            assignments = load_cache(matches[0])
            if len(assignments) != len(frames):
                raise ValueError("Frozen team cache frame count does not match the video")
            print(f"Frozen team input: {matches[0]}", flush=True)
            return assignments

        pipeline.TeamAssigner.get_player_teams_across_frames = frozen_teams

    records = [json.loads(line) for line in
               (DEFAULT_BENCHMARK / "sequences.jsonl").read_text().splitlines()
               if line.strip()]
    video_ids = args.video_id or sorted({record["video_id"] for record in records
                                        if (ROOT / "input_videos" / (record["video_id"] + ".mp4")).exists()})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for video_id in video_ids:
        sys.argv = ["main.py", str(ROOT / "input_videos" / (video_id + ".mp4")),
                    "--analysis-only", "--allow-uncertain-teams",
                    "--output-analysis", str(args.output_dir / (video_id + "_analysis.json"))]
        print(f"Replaying {video_id}", flush=True)
        pipeline.main()
    report = evaluate(args.output_dir.resolve(), DEFAULT_BENCHMARK)
    report["frozen_team_cache_tag"] = args.frozen_team_cache_tag
    (args.output_dir / "shot_count_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["micro_event"], indent=2), flush=True)


if __name__ == "__main__":
    main()

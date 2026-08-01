# CourtVision

CourtVision is an offline basketball-video analysis pipeline. It detects and
tracks players and the ball, assigns teams from jersey evidence, estimates
possession and holder-change events, maps players to a tactical court, and
renders an annotated video.

## Current product status

Treat the output as **experimental analysis**, not verified game statistics.
Team assignment now rejects short, mixed, distant, and referee-contaminated
tracks, but ball acquisition remains sensitive to false ball detections and
nearby defenders. Unknown assignments and missing possession are intentional;
the pipeline no longer invents a team when evidence is weak.

## Local setup

Use Python 3.14 and install the pinned runtime:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Place these untracked model files in `backend/models/`:

- `player_detector.pt`
- `ball_detector_model.pt`
- `court_keypoint_detector.pt`

Verify the environment:

```powershell
.\.venv\Scripts\python.exe scripts\check_runtime.py --check-models
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Analyze a video

```powershell
.\.venv\Scripts\python.exe main.py input_videos\video_1.mp4 `
  --output-video output_videos\video_1.mp4
```

If automatic jersey discovery is uncertain, the process exits with status 2
and a `needs_team_colors` JSON result. Retry with two distinct primary jersey
colors; they guide prototypes but do not bypass crop rejection or unknowns:

```powershell
.\.venv\Scripts\python.exe main.py game.mp4 `
  --team-1-color "#FFFFFF" --team-2-color "#C8102E"
```

For longer uploads, use bounded jobs with unique outputs:

```powershell
.\.venv\Scripts\python.exe main.py game.mp4 `
  --start-seconds 60 --duration-seconds 15 `
  --target-fps 15 --max-width 960 `
  --output-video output_videos\game_s60_d15.mp4
```

## Deployment

The included container is a batch-worker image. Model weights are excluded and
must be mounted at `/app/backend/models`. Mount separate input, output, and cache
volumes for each worker. Do not expose `main.py` directly as a multi-user web
service: job queuing, authentication, upload scanning, retention, and resource
isolation belong in the hosting layer.

The in-memory decoder rejects selections estimated above 2 GiB. Production
jobs should still enforce platform-level limits for upload size, duration,
concurrency, CPU/GPU time, disk, and cache retention.

For a guarded single-job AWS worker, including a 30-second input limit, GPU
verification, a 90-minute hard timeout, and optional automatic instance stop,
follow [the EC2 worker guide](deploy/ec2/README.md).

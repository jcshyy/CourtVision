# CourtVision

CourtVision is an offline basketball-video analysis pipeline. It detects and
tracks players and the ball, estimates teams and possession, maps player
positions to a tactical court, identifies timecoded pass and interception
candidates, and renders an annotated video.

The repository also includes the private-beta web application: passwordless
email access for known users, direct temporary uploads, managed batch jobs, and
an evidence-first review room built around annotated video, tactical court
context, and an event rundown.

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

### Private beta web app

The AWS reference stack is in [`deploy/aws`](deploy/aws/README.md). It keeps the
initial limits (30 seconds, 15 FPS, 960px, 24-hour retention) in configuration,
not application structure, so they can change later without redesigning the
workflow. The deployment requires an AWS Batch queue and job definition with
the CourtVision models, a verified SES sender, and an allowlist entry for each
beta user.

To inspect the UI locally without AWS credentials:

```powershell
python -m http.server 8765 -d web
```

Open `http://127.0.0.1:8765/` for the landing page or
`http://127.0.0.1:8765/demo.html` for the permanent preprocessed sample analysis.
The real authenticated client lives at `app.html`; local-only state inspection
is available with `app.html?demo=signin`, `upload`, `processing`, `colors`,
`error`, or `review`. Those local-only interface states use synthetic fixtures;
authenticated review sessions use artifacts generated from the uploaded clip.

The API contract also has a Flask control-plane adapter. It preserves the
Lambda routes while keeping uploads in S3 and inference in AWS Batch; see
[`deploy/flask`](deploy/flask/README.md) for local and container verification.

### Batch worker

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

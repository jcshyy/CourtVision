# CourtVision local upload demo

This mode runs the real upload-to-review browser workflow without AWS. Flask
serves the existing application, stores jobs under `runs/local_demo`, and runs
one analysis subprocess at a time on this computer.

## Start

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\check_runtime.py --check-models
.\.venv\Scripts\python.exe -m backend.app.local_demo
```

The browser opens at `http://127.0.0.1:8080/`. Local mode skips email sign-in
and is visibly labeled `Local analysis`. Choose a video of 30 seconds or less,
then keep the terminal running while the job processes.

Use a different port or data directory when needed:

```powershell
.\.venv\Scripts\python.exe -m backend.app.local_demo `
  --port 8081 `
  --data-dir C:\path\to\courtvision-local-jobs
```

## Runtime behavior

- Only `127.0.0.1` is exposed by default.
- One worker runs at a time; additional jobs remain queued.
- The real `main.py` pipeline and the local `.pt` weights are used.
- Uploaded sources, annotated videos, manifests, job metadata, and failure
  reports remain under the local data directory. Reusable inference caches stay
  under `backend/stubs`; both locations are excluded from Git.
- Browser access to a job expires after 24 hours by default. Local files are
  not automatically deleted; remove `runs/local_demo` when they are no longer
  needed.
- The safety timeout is 90 minutes. Override it with
  `COURTVISION_LOCAL_TIMEOUT_SECONDS` if a CPU-only run needs longer.

The first run may be slow without an NVIDIA CUDA device. Watch the terminal for
pipeline progress or an actionable model/runtime error. If jersey discovery is
uncertain, the browser asks for two primary jersey colors and reuses the same
uploaded job and cache.

## Stop

Press `Ctrl+C` in the server terminal. A job already running is terminated when
the Python server process exits.

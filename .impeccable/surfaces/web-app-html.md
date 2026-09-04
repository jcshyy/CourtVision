---
version: 1
slug: "web-app-html"
primary_target: "web/app.html"
related_targets: ["web/app.js","web/styles.css","backend/app/web_api.py","backend/app/batch_job.py"]
---

# Initial CourtVision authenticated web application

- **Scope and mode:** Operate surface covering self-service email-and-password authentication, bounded upload-to-review sessions, 24-hour recent-analysis recovery, processing status, immersive result review, download, and structured failure reporting.
- **Audience and job:** Coaches, analysts, and video researchers inspect experimental CourtVision estimates and candidates against a real basketball clip.
- **Primary flow:** Create account → confirm email → sign in → Analyze one bounded clip (maximum 30 seconds; 30 FPS; 1280px) or open Profile to recover an unexpired job → queued/processing status or retained result → synchronized annotated video, tactical court, and Event Rundown → optional Evidence & Unknowns → download or timestamp-scoped issue report.
- **Direction:** Replay Rundown, immersive variant. Approved comp: `.impeccable/mocks/replay-rundown-immersive.png`.
- **Memorable moment:** Selecting a Pass candidate or Interception candidate moves the video playhead and tactical court together.
- **States:** Signed out, account creation, email confirmation, invalid/expired confirmation code, forgot/reset password, upload validation, uploading, queued, processing stages, team colors required, complete, no reliable events, partial tactical data, failed, expired result, expired session, and report success/error.
- **Constraints:** Self-service accounts require confirmed email ownership before analysis. Recent analyses are private to the signed-in email, limited to unexpired jobs, and disappear automatically after the retention window. All outputs remain candidate, estimate, unknown, or experimental and require video review. The app may expose only capabilities implemented in the repository.

## Approved-comp fidelity inventory

| Visible ingredient | Implementation medium | Commitment |
| --- | --- | --- |
| Compact status rail | Semantic HTML/CSS | Service state, deletion deadline, download, report, and sign-out actions. |
| Dominant annotated replay | Native HTML video using the generated result | Primary evidence surface. |
| Tactical court dock | Existing court raster plus semantic markers | Markers use internal `T`-prefixed track IDs, never implied jersey numbers. |
| Persistent Event Rundown | Semantic ordered list | Pass candidate, Interception candidate, and unknown states only. |
| Timecode ruler and playhead | Native range control plus CSS | Synchronizes the selected candidate across replay and tactical view. |
| Evidence & Unknowns | Native disclosure | No calibrated confidence; states explain candidate/unknown review requirements. |
| Account access | Semantic forms | Email-and-password signup, confirmation, sign-in, and password recovery use Cognito-managed email delivery. |
| Profile and Recent analyses | Dedicated compact Analyze/Profile switcher plus semantic ordered list with refresh action | Profile owns the private Recent analyses list and signed-in account/retention context; its compact Analyze shortcut returns to upload, while each unexpired job exposes status, clip identity, submitted time, deletion countdown, and the appropriate progress, issue, or result-recovery action. |
| Upload and processing | Semantic form and live region | Bounded profile and explicit recovery states. |
| Failure report | Accessible dialog/form | Attaches notes to the current job, candidate, and timecode. |

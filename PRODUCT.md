# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

CourtVision serves basketball coaches, analysts, and video researchers who need to inspect game footage and extract structured observations without reviewing every frame manually.

## Product Purpose

CourtVision turns basketball video into an annotated analysis artifact. It detects and tracks players and the ball, assigns teams from jersey evidence, estimates possession and holder-change events, maps player movement onto a tactical court, calculates movement metrics, and renders the results over the source video.

Success currently means producing useful, inspectable experimental analysis while making uncertainty visible. The product must prefer an unknown or missing result over an unsupported statistic.

## Positioning

CourtVision is an evidence-first basketball video analysis system. Its defining mechanism is confidence-aware interpretation across frames: it rejects weak or contaminated observations, preserves unknown assignments, and exposes analysis that researchers and practitioners can inspect rather than presenting every model output as verified fact.

## Operating Context

Users supply basketball footage for bounded analysis jobs and review the rendered video, tactical court view, team assignments, possession estimates, passes or interceptions, speed, and distance outputs. Analysts and researchers may also use diagnostic galleries, benchmark reports, annotation tools, and cached intermediate results to validate behavior.

The current implementation combines a Python batch pipeline with a private-beta web control plane: a static browser client, passwordless allowlisted access, direct temporary uploads, a Lambda job API, an AWS Batch worker entrypoint, and an evidence-first review surface. The reference AWS stack provisions the web and API resources, while the GPU compute environment, Batch queue and job definition, model-weight provisioning, alarms, and capacity limits remain deployment responsibilities. Upload scanning and stronger per-user concurrency controls are still required before access expands beyond a tightly controlled beta.

## Capabilities and Constraints

- Detects and tracks basketball players and the ball from video.
- Assigns teams from multi-frame jersey evidence and permits assignments to remain unknown when evidence is weak.
- Estimates ball possession, passes, and interceptions; these outputs remain experimental and are not verified game statistics.
- Detects court keypoints, projects player locations onto a tactical court, and estimates player speed and distance.
- Renders an annotated output video and preserves bounded-job processing options for clip duration, frame rate, and width.
- Requires local model weights that are excluded from the repository.
- Currently materializes selected and rendered frames in memory; longer clips require stateful chunking before job limits can be raised safely.
- Includes a reference private-beta web layer and AWS control-plane stack; the GPU worker infrastructure and a proven end-to-end staging deployment remain outstanding.
- Commercial use must account for rights to uploaded and benchmark video footage.

## Brand Commitments

The product name is CourtVision. Product language should be direct, analytical, and explicit about confidence, uncertainty, and experimental status. It must not imply that unverified model outputs are official or production-grade statistics.

## Evidence on Hand

- The current product behavior and operating limits are documented in `README.md`.
- A dated real-video validation summary, known failure modes, and remaining release blocker are documented in `docs/validation-report.md`.
- Reproducible labeled development benchmarks and baseline reports live under `benchmarks/courtvision_v1/`.
- A sealed, NBA-weighted holdout protocol and annotation workflow live under `benchmarks/courtvision_nba_holdout_v1/`.
- Automated tests cover the main pipeline, tracking, team assignment, ball events, tactical projection, speed and distance, caching, benchmarks, and video handling under `tests/`.
- No verified customer claims, testimonials, production accuracy claims, or commercial deployment evidence are currently on hand; future product surfaces must not fabricate them.

## Product Principles

1. Prefer explicit uncertainty to false precision.
2. Make every important result inspectable against the source footage.
3. Separate experimental analysis from verified game statistics.
4. Validate changes on reproducible benchmarks without tuning against sealed holdouts.
5. Bound compute, memory, and operational risk before expanding job size or access.

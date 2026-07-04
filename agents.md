# AGENTS.md

Project: CourtVision
Repository: https://github.com/jcshyy/CourtVision
Reference implementation: https://github.com/abdullahtarek/basketball_analysis/tree/main

Investigation context:
- Use the repository history and source for architectural context, but treat the
  current local working tree as authoritative because it may contain uncommitted fixes.
- Reproduce visual pipeline bugs against the included videos in `input_videos/`.
- Treat `input_videos/spursknicksclip.mp4` as an exploratory real-world regression
  video. Review it for failures not exposed by videos 1-3, including team
  assignment, tracking, ball acquisition, possession, passes/interceptions,
  homography, speed/distance, overlays, runtime, and memory behavior. Do not
  assume its current output or statistics are correct.
- Use the reference implementation to understand the intended basketball-analysis
  flow and compare team-assignment behavior, but do not copy assumptions or code
  blindly. Preserve CourtVision's current interfaces and treat its local working
  tree and real-video evidence as authoritative.
- Before implementing a fix, inspect the relevant modules in
  `https://github.com/abdullahtarek/basketball_analysis/tree/main` and document how
  its team assignment, ball acquisition, pass detection, and interception logic
  compare with CourtVision. Identify which behavior is reusable in principle,
  which assumptions do not hold for the included videos, and why the chosen fix
  preserves CourtVision's interfaces. Do not claim to have consulted the reference
  repository unless the relevant source files were actually inspected.
- The current confirmed team-assignment regression is in `input_videos/video_3.mp4`:
  tracked player 22 is initially shown as the blue/white-jersey team and only
  changes after the player moves and a clearer jersey view appears.
- Captured outputs also show players wearing the same visible jersey assigned to
  different display teams in the same frame in `video_2.mp4` and `video_3.mp4`.
  This is not merely swapped team labels; it is inconsistent per-player
  classification and it corrupts possession, pass, and interception statistics.
- Inspect saved tracks, per-frame crops/features, distances to team prototypes,
  confidence/margin, votes, and final assignments for the failing track. Do not
  guess from synthetic colors alone.
- Inspect the rendered output video as well as caches. For `video_3.mp4`, the
  current output misclassifies red-jersey player 22 as Team 1 through frame 99,
  changes the assignment at frame 100, and consequently labels the red-to-white
  holder change at frame 129 as a Team 1 pass instead of an interception/turnover.
  The later holder changes at frames 182, 218, and 240 appear to be white-to-white
  passes and must not be broken while fixing the earlier event.

Rules:
- Keep changes small and focused.
- Do not delete or modify trained model files.
- Do not commit .pt model files.
- Do not remove working features.
- Prefer fixes that improve upload/deployment readiness.
- Use unique per-video or per-job cache/output paths.
- Add logging for pipeline stages when helpful.
- Run tests or basic validation after changes.
- Do not declare a visual classification issue fixed from unit tests alone.
- For team-assignment changes, run the real `video_3.mp4` pipeline or a focused
  diagnostic over its cached player tracks and verify player 22 from its first
  visible frames through later movement.
- If runtime dependencies prevent real-video validation, report the issue as
  unverified rather than marking it complete.
- Version or invalidate affected caches whenever cached-output semantics change.
- Treat the included videos as a small regression suite, not as color-label
  specifications. A fix must be based on crop quality, separability, temporal
  evidence, and track consistency; it must not special-case a filename, track ID,
  frame number, red hue, white brightness, display-team ID, or expected event count.
- Never assume red is Team 2, white is Team 1, or that either team ID has a fixed
  real-world color. Team IDs may be swapped between videos as long as assignments
  are internally consistent and downstream statistics use the same mapping.
- Do not tune a threshold solely until `video_3` passes. Explain the signal behind
  each threshold, test boundary cases, and run the same end-to-end analysis on
  `video_1.mp4`, `video_2.mp4`, and `video_3.mp4` (or report any video that cannot
  be validated).
- For every included video, inspect the source, rendered output, assignment
  metadata/diagnostics, team consistency over time, unknown rate, possession
  segments, team-possession percentages, player speed/distance overlays, and every
  emitted pass/interception. Record the relevant frame, from/to player IDs,
  assigned teams, and whether the footage supports the event.
- `spursknicksclip.mp4` is too large for the current in-memory pipeline as one
  job. Analyze multiple bounded segments using explicit `--start-seconds`,
  `--duration-seconds`, `--target-fps`, `--max-width`, and unique output paths.
  Include early, middle, and late segments plus focused segments around suspected
  possession changes or visible failures. Record the exact settings for each run.
- A fix is not validated if only `video_3` was rendered or if `video_1` and
  `video_2` were checked only through unit tests or aggregate counts.
- Before accepting a team-assignment fix, visually inspect representative early,
  crowded, transition, and late frames. Verify same-jersey players receive one
  team assignment, different jerseys remain separated, unknown evidence stays
  unknown, and player 22 is backfilled consistently from reliable track evidence.
- Recompute and inspect possession, pass, and interception events after any
  upstream assignment change. Confirm events against actual holder transitions;
  do not declare downstream counts fixed merely because a target number appears.
- Validate team possession against the source footage and possession timeline.
  Confirm that only valid teams contribute, unknown/no-holder frames are handled
  explicitly, percentages use a documented denominator, and displayed values
  agree with the underlying frame counts. A correct final percentage is not
  sufficient if intermediate possession segments are assigned to the wrong team.
- Validate speed and distance against player motion and timestamps. Check FPS,
  homography availability/fallback, court scale, displacement, elapsed time,
  stationary-player behavior, track gaps, and implausible spikes. Compare the
  numeric overlays with visible movement at representative slow, stationary,
  running, camera-motion, and homography-transition frames.
- Do not repair speed or possession with video-specific multipliers, caps, player
  IDs, frame ranges, expected percentages, or expected event totals. Fix the
  general upstream cause and add boundary/regression tests.
- Do not hard-code Spurs/Knicks names, jersey colors, player identities, court
  appearance, scoreboard state, timestamps, expected possession, or event totals.
  Any fix motivated by this video must use a general signal and be rerun against
  videos 1-3 to demonstrate that established behavior is preserved.
- Compare before/after results for all three videos and investigate every changed
  assignment transition, possession segment, team-possession percentage,
  speed/distance segment, pass, or interception. Preserve correct existing
  behavior even when the aggregate count happens to remain equal.

Known problem areas:
- Court keypoint/homography reliability.
- Shared stub/cache filenames across videos.
- Manual team descriptions.
- Team discovery can incorrectly group visually distinct red and white jerseys
  when raw torso-average colors include skin, court, background, shadows, or
  jersey-number pixels.
- Prefer robust HSV/Lab jersey-region sampling, pixel filtering, and multiple
  observations per tracked player over a single raw BGR mean crop.
- Team assignment currently locks uncertain early observations too aggressively.
  Occluded, edge-of-frame, small, blurred, or mostly-background torso crops need
  an explicit quality/confidence check and should not count as normal evidence.
- A player should remain unknown or inherit a cautiously computed track-level
  result until enough high-quality observations agree. Do not silently default
  missing/invalid jersey features to team 1.
- Automatic discovery must expose a video/job-level confidence result. If two
  sufficiently distinct and stable jersey prototypes cannot be established,
  do not continue with guessed teams. Support an explicit pre-analysis fallback
  that asks the user for both teams' primary jersey colors (prefer color swatches
  or a color picker, with optional team labels).
- Treat user-provided jersey colors as prototype guidance, not guaranteed truth:
  retain crop-quality rejection, unknown assignments, distance margins, and
  temporal/track-level aggregation. Validate that the two colors are distinct.
- Record assignment mode (`automatic` or `user_colors`) and normalized color
  inputs in job metadata and the assignment-cache identity. Changing colors or
  mode must invalidate only the affected team-assignment cache.
- Never silently switch to hard-coded jersey descriptions or default colors when
  automatic discovery is uncertain. Return an actionable status that the UI/CLI
  can use to request jersey colors before analysis continues.
- Team assignment stability and cache invalidation when assignment logic changes.
- Ball possession, passes, and interceptions depending on upstream accuracy.
- Team-possession percentages depending on ball acquisition, team assignment,
  unknown-frame policy, and the percentage denominator.
- Speed/distance accuracy depending on FPS, homography validity, court scale,
  coordinate continuity, track gaps, and camera motion.
- Deployment asset handling for large models.
- Runtime/dependency consistency.

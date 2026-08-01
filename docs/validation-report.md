# CourtVision validation report

Date: 2026-07-08

## Scope and settings

The local working tree was treated as authoritative. The current player cache
uses `v3_referee_track_filter`; team assignment uses algorithm `v14`. Videos
1–3 were decoded at their native 30 FPS and 1280×720 resolution. The exploratory
clip was run as three unique jobs:

| Segment | Settings |
| --- | --- |
| Early | `--start-seconds 0 --duration-seconds 4 --target-fps 15 --max-width 960` |
| Middle | `--start-seconds 4 --duration-seconds 4 --target-fps 15 --max-width 960` |
| Late | `--start-seconds 8 --duration-seconds 4 --target-fps 15 --max-width 960` |

Generated videos, crop galleries, source/render contact sheets, and JSON audit
reports are stored under the ignored `output_videos/validation/` directory.

## Reference implementation comparison

The relevant reference source was inspected before implementation:

- Team assignment runs FashionCLIP on one full player crop using two fixed text
  descriptions, caches the result by player ID, and clears the cache every 50
  frames. The reusable principle is stable assignment by track. Its fixed jersey
  descriptions, single-view classification, and periodic forgetting do not hold
  for the supplied videos.
- Ball acquisition uses bbox containment, distance to player-box key points, and
  11 consecutive frames. CourtVision retains those public interfaces and the
  general spatial evidence, fixes the farthest-player containment bug, backfills
  confirmed runs, and expresses confirmation duration in seconds so sampled
  clips behave consistently. The raw ball signal remains too noisy for a final
  accuracy claim.
- Pass and interception detection compare the previous and current confirmed
  holder teams. CourtVision retains this behavior, keeps unknown teams explicit,
  records release/catch/gap metadata, and rejects transitions across excessive
  no-holder gaps.

## Team-assignment result

The real-video crop galleries show consistent display teams for clearly visible
matching jerseys in videos 1–3. Short, mixed, occluded, and referee-contaminated
tracks now remain unknown instead of defaulting to a team.

For `video_3`:

- Track 12, corresponding to the long red track that exposed the early-view
  regression, is Team 2 for its complete cached timeline. It has 102 accepted
  observations; 22 of 24 temporally selected observations vote Team 2, with
  93.23% weighted support. Early evidence is therefore backfilled from the full
  track rather than locked from the first crop.
- Current track 22 is also consistently red/Team 2: all 11 observations from
  frames 109–119 vote Team 2.
- The two-frame referee-contaminated track 37 remains unknown.
- Team IDs are display identities only and are allowed to swap between videos.

## End-to-end summary

Possession percentages use only frames with both a confirmed holder and a valid
team; unknown/no-holder frames are excluded from the denominator.

| Video | Frames | Unknown assignment labels | Known possession frames | Team possession | Events | Homography | Maximum reported speed |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| video 1 | 117 | 109 | 55 | 100.00% / 0.00% | none | 2 unavailable frames | 28.80 km/h |
| video 2 | 174 | 346 | 91 | 100.00% / 0.00% | 4 passes | 7 temporal discontinuities | 25.09 km/h |
| video 3 | 243 | 359 | 82 | 75.61% / 24.39% | 3 passes | 26 unavailable frames; 7 discontinuities | 29.39 km/h |

Rendered outputs have exactly the same frame count and FPS as their sources.
Speed is calculated with source FPS, continuous track segments, explicit
homography discontinuities, a time-scaled half-second window, and median-smoothed
tactical coordinates. No video-specific multiplier or speed cap is used.

The three later `video_3` white-to-white transfers remain represented. Offline
backfilling moves their catch frames from the old delayed 182/218/240 markers to
172/208/230, matching the start of each confirmed holder run.

## Exploratory segment findings

- Early: consistent white/dark separation with two unknown tracks; no possession
  run was reliable enough to confirm.
- Middle: team IDs are swapped relative to the early job but internally
  consistent. Twelve possession frames were confirmed, split 50%/50%, with one
  holder change classified as an interception. It requires manual play review
  before being treated as a statistic.
- Late: internally consistent team separation with five unknown tracks; no
  possession run was reliable enough to confirm.
- All three segments had zero speeds above 40 km/h. The middle segment used two
  bounded homography fallback frames; early and late had no homography failures.

## Remaining release blocker

Team assignment is substantially safer, but possession and event accuracy are
not production-ready. In `video_3`, raw ball proximity alternates among nearby
players during a continuous dribble. The red-to-white change visible around
frames 119–129 does not become a verified interception because the previous red
holder evidence is separated by a long interval of unstable candidates. Lowering
the confirmation threshold produces an earlier false turnover around frames
66–75, so threshold tuning is not an acceptable fix.

The next required model change is a confidence-bearing ball trajectory and
holder state model that combines ball detection confidence, motion continuity,
player containment, candidate distance, and hysteresis. Until that is validated,
deploy the application only as an experimental batch analyzer and label
possession/pass/interception output accordingly.

# Calibration review checklist

All labels in this batch are drafts. Review the rendered contact sheets against
the raw clips before changing any `review_status` value to `verified`.

## `nba_015` — events verified

- Same-team pass: release frame 165, catch frame 181.
- Made three-point attempt: release frame 222.
- Cleveland inbound: release frame 450, catch frame 465.
- Hard cut: frame 478.
- Re-check the revised ball-center markers; prior body-locked markers were
  removed or re-labeled from the raw full-resolution frames.

## `ncaa_m_001`

- Confirm that frames 69, 387, 426, 612, and 693 start new scenes.
- Verified correction: frames 249-256 are shooting motion, not a pass. A
  possible catch/gather begins near frame 244 and the made shot releases at
  frame 264.
- Verified missed free throw: release frame 517.
- Verified defensive rebound secure-control frame: 585.
- Re-check the final pass catch boundary. It releases at frame 742, remains in
  flight at frame 751, and is provisionally caught at frame 755.

## `fixed_001`

- Verified team 1: red, pink, and blue.
- Verified team 2: black, orange, and neon yellow/green.
- Re-check the provisional endpoints of the blue-to-pink pass: release frame
  100, catch frame 114.
- Re-check the provisional endpoints of the pink-to-red pass: release frame
  142, catch frame 158.
- Re-check the provisional endpoints of the red-to-blue pass: release frame
  170, catch frame 186. The ball is still in flight at frame 178.
- Verified shot attempt: release frame 204. The clip ends before its outcome is
  clear, so `unknown` remains the conservative label.
- Re-check the revised small-ball markers. Ambiguous body-locked points were
  removed instead of being guessed.

Do not resolve a low-confidence item by guessing. Leave it `draft`, or mark the
ball `uncertain`, until the raw video supports a definitive label.

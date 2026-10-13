---
version: 1
slug: "web-index-html"
primary_target: "web/index.html"
related_targets: ["web/landing.css","web/landing.js","web/demo.html","web/app.html"]
---

# CourtVision public landing page

- **Scope and mode:** Persuade surface for coaches, analysts, and video researchers evaluating the public experimental preview.
- **Visitor understanding and action:** Understand that CourtVision analyzes short basketball clips into an annotated replay, tactical court, possession estimates, and timecoded pass/interception candidates; open the public upload-preview desk or inspect the permanent fully interactive sample.
- **Proof:** A visible Analyze video action, a large live synthetic interface demonstration, an explicit capacity-status band, a 00:00–00:30 cue ruler, and three supported capability bands: Annotated Replay, Tactical Court View, and Evidence & Unknowns.
- **Direction:** Supported-Capabilities Live Film Room in the established Replay Rundown world. Approved comp: `.impeccable/mocks/landing-supported-capabilities.png`.
- **Memorable moment:** Selecting a Pass candidate or Interception candidate seeks the synthetic replay to the same timecode and updates the tactical court.
- **Constraints:** Use the exact approved hero copy. Every result is experimental and requires video review. While `publicPreview=true` and `analysisAvailable=false`, `/app.html` validates format, size, and duration locally, stops before network upload, shows capacity pending, and explicitly says the video was not uploaded. No evidence ledger, audit record, multi-angle, official verification, fouls/contact/screens/drives/feet-set/shots/points/end-of-possession events, jersey numbers, calibrated confidence, accuracy, customer counts, testimonials, or authoritative decisions.
- **Routes:** `/` is the official public site with a visible Analyze video action. `/app.html` is the public upload-preview state while analysis capacity is unavailable; runtime configuration can restore its authenticated live-analysis path when GPU capacity becomes available. `/demo.html` remains the permanent API-independent, fully interactive preprocessed sample.
- **Hosting:** Cloudflare Pages is the intended static host. Hosting mechanics are implementation context, not a new visual-system concept.

## Approved-comp fidelity inventory

| Visible ingredient | Implementation medium | Commitment |
| --- | --- | --- |
| Compact public navigation | Semantic HTML/CSS | CourtVision, supported-capabilities anchors, and a persistent Analyze video action without dashboard chrome. |
| Direct hero | Semantic HTML/CSS | Exact user-provided headline and paragraph, Analyze video first, sample analysis second, and no unsupported claim. |
| Faint court geometry | Authored CSS/SVG | Court lines organize negative space; they never imply tracked data. |
| Dominant interface demo | Live iframe to `demo.html` | Real interactive synthetic UI, not a raster screenshot; visibly labeled synthetic. |
| Capacity-status band | Semantic HTML/CSS | State plainly that public local preview is open, live processing is pending, and selected video is not uploaded. |
| 30-second cue ruler | Semantic HTML/CSS | Structural spine for the initial bounded profile; candidate and unknown marks only. |
| Three proof bands | Semantic HTML/CSS plus existing synthetic assets | Annotated replay, tactical player positions, and Evidence & Unknowns only. |
| Closing analysis invitation | Semantic HTML/CSS | Experimental-status reminder plus public upload-preview and permanent-sample actions. |

The approved comp is a north star for hierarchy and material treatment. Its core UI must remain semantic and interactive. Production copy is governed by the user-approved capability contract rather than any incidental generated text.

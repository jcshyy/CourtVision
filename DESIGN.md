---
name: CourtVision
description: Evidence-first basketball replay review for inspectable experimental analysis.
colors:
  monitor: "#111417"
  monitor-deep: "#090b0d"
  monitor-raised: "#1a1f24"
  ink: "#0f1419"
  paper: "#e8ecef"
  paper-bright: "#f8fafb"
  paper-muted: "#cbd2d8"
  rule: "#313a42"
  rule-light: "#aeb7bf"
  cobalt: "#2d6bff"
  cobalt-strong: "#1e55d6"
  cobalt-soft: "#dfe8ff"
  amber: "#f3b51b"
  amber-ink: "#382600"
  amber-soft: "#fff0bf"
  danger: "#d74b4b"
  danger-soft: "#ffe3e3"
  success: "#2e8b65"
  success-soft: "#dff4ea"
  text: "#f5f7f8"
  text-muted: "#aeb8c1"
  paper-text-muted: "#53616d"
  pure-white: "#ffffff"
typography:
  display:
    fontFamily: "Barlow Condensed, sans-serif"
    fontSize: "clamp(2.75rem, 7vw, 5.75rem)"
    fontWeight: 700
    lineHeight: 0.92
    letterSpacing: "-0.03em"
  landing-display:
    fontFamily: "Barlow Condensed, sans-serif"
    fontSize: "clamp(3.6rem, 7.4vw, 6rem)"
    fontWeight: 700
    lineHeight: 0.92
    letterSpacing: "-0.03em"
  headline:
    fontFamily: "Barlow Condensed, sans-serif"
    fontSize: "clamp(1.65rem, 3vw, 2.55rem)"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "-0.02em"
  title:
    fontFamily: "Barlow Condensed, sans-serif"
    fontSize: "1.15rem"
    fontWeight: 700
  body:
    fontFamily: "Barlow, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Barlow Condensed, sans-serif"
    fontSize: "0.86rem"
    fontWeight: 700
    letterSpacing: "0.055em"
  timecode:
    fontFamily: "Barlow Condensed, sans-serif"
    fontSize: "1.08rem"
    fontWeight: 600
rounded:
  small: "9px"
  regular: "14px"
  pill: "999px"
spacing:
  page-pad: "clamp(1rem, 2.4vw, 2.25rem)"
  landing-page: "clamp(1rem, 3vw, 3rem)"
components:
  button-primary:
    backgroundColor: "{colors.cobalt}"
    textColor: "{colors.pure-white}"
    rounded: "{rounded.small}"
    padding: "0.65rem 1rem"
  button-primary-hover:
    backgroundColor: "{colors.cobalt-strong}"
    textColor: "{colors.pure-white}"
    rounded: "{rounded.small}"
    padding: "0.65rem 1rem"
  button-secondary:
    backgroundColor: "transparent"
    textColor: "{colors.text}"
    rounded: "{rounded.small}"
    padding: "0.65rem 1rem"
  button-quiet:
    backgroundColor: "transparent"
    textColor: "{colors.text-muted}"
    rounded: "{rounded.small}"
    padding: "0.45rem 0.7rem"
  input-field:
    backgroundColor: "{colors.paper-bright}"
    textColor: "{colors.ink}"
    rounded: "{rounded.small}"
    padding: "0.72rem 0.85rem"
  beta-chip:
    backgroundColor: "{colors.amber}"
    textColor: "{colors.amber-ink}"
    rounded: "{rounded.pill}"
    padding: "0.25rem 0.55rem"
  event-row-selected:
    backgroundColor: "{colors.cobalt-soft}"
    textColor: "{colors.ink}"
    padding: "0.75rem 1rem"
  tactical-dock:
    backgroundColor: "{colors.monitor}"
    textColor: "{colors.text}"
    rounded: "{rounded.small}"
  landing-button-primary:
    backgroundColor: "{colors.cobalt}"
    textColor: "{colors.pure-white}"
    rounded: "{rounded.small}"
    padding: "0.72rem 1rem"
  landing-button-secondary:
    backgroundColor: "transparent"
    textColor: "{colors.text}"
    rounded: "{rounded.small}"
    padding: "0.72rem 1rem"
---

# Design System: CourtVision

## Overview

**Creative North Star: "Replay Rundown / Immersive Replay"**

CourtVision is a live production desk for interrogating a model result against the footage that produced it. Matte monitor black lets the annotated replay dominate; cool-gray rundown stock gives events, forms, capability bands, and recovery steps the familiar order of a broadcast control sheet. Cobalt behaves like a digital grease pencil—an active annotation, synchronized position, or available action—not ambient brand decoration.

The system is dense, exact, and deliberately non-authoritative across three separated surfaces: the official public site, a public upload-preview desk whose live-analysis path is runtime-controlled, and a permanent API-independent synthetic demo. Amber makes preview status, capacity limits, and unresolved evidence conspicuous, while tabular timecodes make every cue recoverable. Narrow broadcast lettering gives the interface urgency without imitating a scoreboard or turning experimental output into a settled game record.

**Key Characteristics:**

- Immersive replay on a matte monitor-black working field.
- Cool-gray rundown stock for ordered controls and evidence lists.
- Cobalt grease-pencil marks for actions, selection, and synchronization.
- Amber uncertainty that remains visible instead of being averaged away.
- Narrow broadcast lettering with exact, tabular timecodes.
- Evidence-first language that labels model output as candidate, unknown, or experimental.
- Public education, local upload preview, permanent synthetic demonstration, and live real-clip analysis remain visibly and technically explicit.

## Colors

The palette separates the replay desk from its paper rundown, then reserves chroma for action, uncertainty, and bounded feedback.

### Primary

- **Grease-Pencil Cobalt** (`cobalt`): Primary actions, active timeline marks, selected team markers, drag state, and event candidates.
- **Pressed Cobalt** (`cobalt-strong`): Hover state for primary actions.
- **Selection Wash** (`cobalt-soft`): Selected and hovered event rows on paper stock.

### Secondary

- **Uncertainty Amber** (`amber`): Public-preview capacity state, focus outlines, unknown markers, and possible-holder emphasis.
- **Uncertainty Ink** (`amber-ink`): Legible text and strokes against amber fills.
- **Uncertainty Wash** (`amber-soft`): Unknown-player surfaces on the tactical court.

### Tertiary

- **Failure Red** (`danger`, `danger-soft`): Destructive action and error feedback.
- **Completion Green** (`success`, `success-soft`): Completed processing stages and successful feedback.

### Neutral

- **Monitor Black** (`monitor`): The main working field and replay console.
- **Monitor Deep** (`monitor-deep`): Top bar, video well, processing desk, and deepest evidence cells.
- **Monitor Raised** (`monitor-raised`): Quiet hover feedback on the dark field.
- **Rundown Ink** (`ink`): Primary copy on paper stock.
- **Rundown Paper** (`paper`): Event rail, upload rundown, processing sheet, forms, and dialogs.
- **Bright Paper** (`paper-bright`): Inputs and light tactical markers.
- **Muted Paper** (`paper-muted`): Subordinate marks on dark surfaces.
- **Dark Rule / Light Rule** (`rule`, `rule-light`): Structural dividers for monitor and paper materials respectively.
- **Monitor Text / Muted Monitor Text** (`text`, `text-muted`): Primary and supporting copy on the replay desk.
- **Muted Paper Text** (`paper-text-muted`): Secondary instructions and evidence qualifiers on rundown stock.
- **Pure White** (`pure-white`): High-contrast text on cobalt and semantic fills.

### Named Rules

**The Grease-Pencil Economy Rule.** Cobalt marks an action, selection, synchronized cue, or tracked subject; it does not wash large passive surfaces.

**The Amber Means Uncertainty Rule.** Amber always signals preview status, unknown state, possible possession, or keyboard focus. Never use it as generic decoration.

**The Two Stocks Rule.** Monitor neutrals hold the replay and its instrumentation; paper neutrals hold ordered lists, forms, recovery steps, and disclosures.

## Typography

**Display Font:** Barlow Condensed (with sans-serif fallback)

**Body Font:** Barlow (with sans-serif fallback)
**Label/Timecode Font:** Barlow Condensed (with sans-serif fallback)

**Character:** Barlow Condensed supplies the narrow, urgent cadence of a broadcast rundown; Barlow keeps instructions and evidence readable at working density. Both families are locally hosted and used without ornamental alternates.

### Hierarchy

- **Display** (700, responsive display scale, 0.92 line-height): High-impact state headlines and upload/recovery prompts, balanced and usually held to 10–14 characters per line.
- **Landing Display** (700, enlarged responsive display scale, 0.92 line-height): The uppercase public hero and major capability-section statements.
- **Headline** (700, responsive headline scale, 1 line-height): Panel and form headings.
- **Title** (700, compact fixed scale): Local group and empty-state headings.
- **Body** (400, base scale, 1.5 line-height): Instructions, evidence explanations, and operational copy; paragraphs stop at 70 characters.
- **Label** (700, compact scale, 0.055em tracking, uppercase): Field labels, panel titles, and operational metadata.
- **Timecode** (600, compact fixed scale, tabular numerals): Event cues, replay position, duration, and retention-related timing.

### Named Rules

**The Timecode Rule.** A precise cue is always `MM:SS.t`; bounded duration and ruler labels use compact `MM:SS`. Keep leading zeroes and tabular numerals, and never extend the first-profile ruler beyond the real clip duration or 30 seconds.

**The Broadcast Narrow Rule.** Use Barlow Condensed for headlines, labels, brand lettering, status chips, panel titles, and timecodes; use regular-width Barlow for explanatory prose and controls.

## Layout

The official public site at `/` uses a direct split hero: exact evidence-first copy on the left and faint authored court geometry on the right. The geometry is compositional structure, never tracked data. `Analyze video` is the visible primary action and opens `/app.html`; the permanent synthetic interface demo begins immediately below the hero and remains the dominant proof object, followed by a capacity-status band, 30-second cue spine, three capability bands, a bounded three-step workflow, and a closing invitation.

The three surfaces stay separate by route and responsibility. `/` is the official public site and explains the implemented capability contract. `/app.html` is a public local-validation preview while `publicPreview=true` and `analysisAvailable=false`: it checks format, size, and duration without sending the selected video, stops before network upload, and returns a capacity-pending recovery state that explicitly says the video was not uploaded. `/demo.html` remains the permanent, fully interactive, preprocessed sample independent of the analysis API. When live GPU capacity becomes available, runtime configuration may restore the authenticated live-analysis path without changing this visual world.

The desktop review is an immersive two-column desk: the replay takes the fluid column and the persistent rundown rail clamps between 19rem and 27rem. A 4.25rem status rail keeps identity, service state, deletion deadline, download, report, and sign-out controls visible without becoming global navigation. The tactical court docks over the replay at up to 40% of its width, while the timeline and Evidence & Unknowns stay attached below the monitor.

Upload and processing reuse the same desk grammar: a dominant monitor field paired with a narrower paper rundown. Page padding is fluid through the `page-pad` token. Major paired desks use a regular 14px corner and clip their contrasting stocks into one bounded artifact.

At 1080px, rails and docks narrow. At 820px, paired desks stack, the tactical court leaves picture-in-picture mode for a full-width companion panel, and the event list receives a bounded scroll region. At 560px, actions become icon-forward, the limit strip becomes one column, evidence cells stack, and tracking overlays yield to the underlying video. No essential action depends on hover.

Landing behavior follows its own observed thresholds. At 980px, navigation anchors collapse, the hero court becomes a faint edge field, and the embedded demo receives a fixed working height. At 760px, capability bands and workflow steps stack, court copy precedes its visual, and the hero limits become a two-column strip. At 520px, actions become full-width, the demo frame runs edge-to-edge, capability typography tightens, and the Event Rundown sample collapses to a two-column cue without losing visible state text.

**The Replay Leads Rule.** On a review surface, the annotated video remains the largest evidence region; the tactical court, timeline, Event Rundown, and Evidence & Unknowns explain it rather than competing with it.

**The Surface Separation Rule.** Public explanation, local upload preview, permanent synthetic demonstration, and live real-clip analysis never blur their service state or imply that a local or synthetic video was uploaded or analyzed.

## Elevation & Depth

CourtVision is flat by default and builds depth through tonal stock changes, hairline rules, and clipping. The shared desk shadow (`shadow`) lifts major upload and processing artifacts from the monitor field. Tactical docks and toasts use a tighter dark shadow; dialogs use the strongest overlay shadow; ordinary buttons gain only a small hover lift.

### Shadow Vocabulary

- **Desk Lift** (`0 18px 44px rgba(0, 0, 0, 0.28)`): Major upload and processing desks.
- **Dock Lift** (`0 14px 34px rgba(0, 0, 0, 0.36)`): Tactical dock and toast notices.
- **Dialog Lift** (`0 30px 80px rgba(0, 0, 0, 0.5)`): Modal failure reporting over a darkened backdrop.
- **Action Lift** (`0 8px 20px rgba(0, 0, 0, 0.22)`): Hover response for ordinary buttons.

### Named Rules

**The Flat Evidence Rule.** Rows, rulers, and evidence cells stay flat and separated by rules; shadows are reserved for whole desks, overlays, and transient lift.

## Shapes

Large composite desks and dialogs use gently rounded regular corners (14px). Controls, fields, tactical docks, notices, and toasts use the tighter small corner (9px). Status chips are fully pill-shaped, while timecode badges remain square-edged to read like burned-in monitor data.

Shape carries evidence semantics on the tactical court: display team one is circular, display team two is squared, and unknowns use dashed amber treatment. Timeline candidates are circular; `Unknown` cues are square. This redundancy preserves meaning without relying on color alone.

**The Shape Carries State Rule.** Candidate and unknown states must differ by silhouette or border style as well as color.

## Components

### Buttons

Buttons are compact, tactile controls that lift by one pixel on hover while preserving the production-desk density.

- **Shape:** Tight small corner (9px), 2.75rem minimum height, and 0.65rem by 1rem padding.
- **Primary:** Cobalt fill with pure-white text; hover moves to pressed cobalt.
- **Secondary:** Transparent on dark stock with a medium-gray border and monitor text; paper contexts switch to rundown ink and a light-gray hover fill.
- **Quiet:** Transparent, muted text, reduced height, and no hover shadow; used for sign-out and low-priority changes.
- **Focus:** A 3px amber outline with a 3px offset. Disabled controls retain structure at 52% opacity.

### Status Chips

The operational-status chip is a compact amber pill with uppercase condensed type and a leading current-color dot. It communicates public preview, capacity, or uncertainty state, never marketing promotion.

### Cards / Containers

- **Corner Style:** Regular corners (14px) for paired desks and dialogs; small corners (9px) for docks and notices.
- **Background:** Monitor/deep-monitor layers for replay and processing; paper for rundowns, forms, and dialogs.
- **Shadow Strategy:** Whole artifacts use the bounded vocabulary in Elevation & Depth; internal rows remain flat.
- **Border:** One-pixel dark rules on monitor surfaces and light rules on paper surfaces.
- **Internal Padding:** Fluid at desk level; compact and repeated within lists and fields.

### Inputs / Fields

- **Style:** Bright paper fill, rundown ink, a one-pixel medium-gray stroke, small corners (9px), and 0.72rem by 0.85rem padding.
- **Focus:** Border changes to cobalt while the universal amber focus outline remains visible.
- **Specialized Input:** The six-digit access code uses 2rem condensed bold type, 0.32em tracking, centered alignment, and tabular numerals.
- **Error / Disabled:** Errors use failure-red ink on a soft-red field; disabled inputs retain layout at 52% opacity.

### Navigation

The app top status rail is utility navigation, not a dashboard shell. CourtVision and the current operational-status chip sit left, timing or service status centers, and available session actions sit right. Below 560px, action labels become visually hidden while icons preserve the same accessible names.

The public landing uses a separate sticky 4.5rem navigation rail with brand, section anchors, experimental-analysis state, and a visible `Analyze video` action to `/app.html`. Section anchors collapse below 980px; the analysis action remains available.

### Public Landing Hero

The public hero uses the exact headline **“Review every candidate against the play.”** and the exact supporting paragraph **“CourtVision analyzes short basketball clips and produces an annotated replay, tactical court view, possession estimates, and timecoded pass and interception candidates. Results are experimental and require video review.”** `Analyze video` is the primary action and `Explore sample analysis` is the secondary action. A compact availability line states that the public preview is open, live processing is waiting on GPU capacity, and the sample remains fully interactive.

Authored one-pixel court lines, rings, lane geometry, and a dashed arc organize the hero’s negative space. They are faint, unlabeled, and `aria-hidden`; never render track markers or claim that this geometry is an analysis result.

### Public Upload Preview

While public preview is enabled and analysis is unavailable, the upload desk keeps its existing paper-and-monitor composition but replaces authenticated-processing assumptions with explicit service truth. It validates the selected video's format, size, and duration locally, shows an amber **Analysis capacity pending** notice, and stops before any network upload. The terminal recovery message must include **“This video was not uploaded.”** and point to the working sample; a successful local validation must never be styled or worded as completed analysis.

When runtime configuration enables analysis again, the same route may restore authentication, upload, processing, and review. This is a state change within the established system, not a separate visual identity.

### Permanent Interface Demo

The landing embeds `/demo.html` as the dominant live interface artifact inside a monitor-black browser frame. The demo is API-independent, visibly labeled **Synthetic interface demo**, **No video is processed**, and **Experimental results**, and uses only synthetic footage and internal data. Selecting a `Pass candidate` or `Interception candidate` synchronizes the synthetic replay time, tactical view, and Evidence & Unknowns.

### Capability Bands

Three paper-stock bands state only implemented capabilities: **Annotated replay** for player and ball tracks plus team and possession estimates; **Tactical court view** for estimated player positions labeled with internal `T`-prefixed track IDs; and **Evidence & Unknowns** for pass/interception candidates and unresolved states. The bands alternate copy and visual regions without expanding the capability contract.

### Event Rundown

Each event row is a full-width four-part cue: exact timecode, state symbol, event label, and explicit state text. Event labels are only `Pass candidate` or `Interception candidate`; visible state text is only `Candidate` or `Unknown`. Selected and hovered rows use the cobalt selection wash; the current row gains a one-pixel inset cobalt rule. Selecting a cue synchronizes replay time, tactical position, Evidence & Unknowns, and reporting context.

### Timeline

The app timeline is a one-pixel light rule with cobalt candidate dots, square amber unknown markers, a cobalt two-pixel playhead, and three tabular labels. The invisible range input covers the full scale so the native control remains keyboard and pointer operable. The public 30-second cue spine uses the same candidate/unknown shapes at `00:00`, `00:15`, and `00:30`; it describes the bounded profile rather than pretending to replay data.

### Tactical Court

The tactical dock uses a monitor frame, condensed uppercase header, court raster, semantic player silhouettes, and an explicit legend. Every player label is an internal `T`-prefixed track ID such as `T1`; never use jersey-looking bare numbers. A possible holder gains an amber ring and small lift; unavailable frames render a centered amber-bordered notice instead of fabricated positions.

### Evidence & Unknowns

Evidence & Unknowns is a native disclosure attached below the timeline. Its dark cells state the selected candidate, assignment state, review requirement, and tactical availability; every result remains a candidate or estimate that requires source-play review. The disclosure remains subordinate to the replay and Event Rundown scan path. `Analysis Details` is the only acceptable alternate label when space or context requires it.

## Do's and Don'ts

### Do:

- **Do** keep replay footage dominant and make every candidate recoverable by an exact timecode.
- **Do** describe every result as a candidate or estimate that requires video review; use `Candidate` or `Unknown` as Event Rundown state text.
- **Do** keep the official public site, local upload preview, permanent API-independent synthetic demo, and live analysis states visibly and technically explicit.
- **Do** validate preview format, size, and duration on-device, stop before network upload while capacity is pending, and say plainly that the video was not uploaded.
- **Do** use authored court geometry as quiet composition on the landing and the live synthetic interface as its dominant demonstration.
- **Do** keep landing capability bands to annotated player/ball tracking, team/possession estimates, tactical player positions, annotated video, and pass/interception candidates.
- **Do** label all tactical players with internal `T`-prefixed track IDs.
- **Do** preserve shape and border redundancy for team and uncertainty states.
- **Do** use cobalt for actions and synchronized evidence, amber for uncertainty and focus, and neutral stocks for structure.
- **Do** preserve keyboard-visible amber focus and reduced-motion behavior on every interactive state.

### Don't:

- **Don't** turn CourtVision into an authoritative KPI dashboard or present experimental analysis as a settled game record.
- **Don't** fabricate events, player positions, tactical data, or analysis detail when the pipeline supplies none.
- **Don't** add trust indicators, decision claims, or analysis categories beyond the repository capability contract.
- **Don't** add any Event Rundown label beyond `Pass candidate` and `Interception candidate`.
- **Don't** use jersey-looking bare numbers for player markers.
- **Don't** imply that local validation uploaded, processed, or analyzed the selected video.
- **Don't** use amber as a decorative accent or cobalt as a broad passive background.
- **Don't** hide uncertainty inside Evidence & Unknowns; the Event Rundown itself must say when a cue is `Unknown`.
- **Don't** wrap the authenticated bounded upload-to-review flow in global product navigation.

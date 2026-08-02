(() => {
  "use strict";

  const config = Object.assign(
    {
      apiBaseUrl: "/api",
      maxDurationSeconds: 30,
      maxUploadBytes: 500 * 1024 * 1024,
      targetFps: 15,
      maxWidth: 960,
      resultRetentionHours: 24,
      pollIntervalMs: 4000,
    },
    window.COURTVISION_CONFIG || {},
  );

  const app = document.querySelector("#app");
  const query = new URLSearchParams(window.location.search);
  const localHost = ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
  const permanentDemo = document.body.hasAttribute("data-courtvision-demo");
  const embeddedDemo = permanentDemo && query.get("embedded") === "1";
  const demoMode = permanentDemo ? query.get("state") || "review" : localHost ? query.get("demo") : null;
  const activeJobKey = "courtvision.activeJob";
  const permanentDemoAssets = {
    videoUrl: "assets/courtvision-demo-tactical-fixed.webm",
    analysisUrl: "assets/courtvision-demo-analysis.json",
    posterUrl: "assets/courtvision-demo-poster.webp",
  };

  const state = {
    view: "loading",
    authStep: "email",
    email: "",
    session: null,
    csrfToken: null,
    selectedFile: null,
    selectedFileDuration: 0,
    uploadProgress: 0,
    busy: false,
    message: null,
    job: null,
    analysis: null,
    downloads: null,
    selectedEventId: null,
    currentTime: 0,
    pollTimer: null,
    toastTimer: null,
    syncAnimationTimer: null,
  };

  const iconPaths = {
    upload: '<path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v5h14v-5"/>',
    download: '<path d="M12 4v12m0 0l4.5-4.5M12 16l-4.5-4.5M5 20h14"/>',
    flag: '<path d="M6 21V4m0 1h10l-1.5 4L16 13H6"/>',
    signout: '<path d="M10 5H5v14h5m3-4l4-3-4-3m4 3H9"/>',
    mail: '<path d="M4 6h16v12H4zM4 7l8 6 8-6"/>',
    lock: '<rect x="5" y="10" width="14" height="10" rx="2"/><path d="M8 10V7a4 4 0 018 0v3"/>',
    arrow: '<path d="M5 12h14m-5-5l5 5-5 5"/>',
    play: '<path d="M8 5l11 7-11 7z"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v6l4 2"/>',
    check: '<path d="M5 12.5l4 4L19 7"/>',
    alert: '<path d="M12 4L3.5 20h17L12 4zM12 9v5m0 3h.01"/>',
    close: '<path d="M6 6l12 12M18 6L6 18"/>',
    refresh: '<path d="M19 8V4l-3 3a8 8 0 10.5 10.5M19 4v5h-5"/>',
    folder: '<path d="M3 7h7l2 2h9v10H3z"/>',
    evidence: '<path d="M5 4h14v16H5zM8 8h8M8 12h8M8 16h5"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-9h.01"/>',
    chevron: '<path d="M8 10l4 4 4-4"/>',
  };

  function icon(name, label = "") {
    const aria = label ? `role="img" aria-label="${escapeHtml(label)}"` : 'aria-hidden="true"';
    return `<svg class="icon" viewBox="0 0 24 24" ${aria}>${iconPaths[name] || iconPaths.info}</svg>`;
  }

  const demoEvents = [
    { id: "event-1", type: "pass", timeSeconds: 4.2, status: "candidate", fromTeamId: 1, toTeamId: 1 },
    { id: "event-2", type: "pass", timeSeconds: 8.4, status: "candidate", fromTeamId: 1, toTeamId: 1 },
    { id: "event-3", type: "interception", timeSeconds: 14.7, status: "unknown", fromTeamId: 2, toTeamId: null },
    { id: "event-4", type: "pass", timeSeconds: 22.3, status: "unknown", fromTeamId: null, toTeamId: null },
    { id: "event-5", type: "interception", timeSeconds: 29.1, status: "candidate", fromTeamId: 1, toTeamId: 2 },
  ];

  const demoAnalysis = {
    schemaVersion: 1,
    beta: true,
    disclaimer:
      "Synthetic local preview. Production shows only the uploaded clip and its experimental CourtVision analysis.",
    source: { fps: 15, frameCount: 450, durationSeconds: 30 },
    court: { width: 300, height: 161 },
    events: demoEvents,
    frames: [0, 4.2, 8.4, 14.7, 22.3, 29.1].map((time, index) => makeDemoFrame(time, index)),
    diagnostics: {
      tacticalView: { fallback_used: [126, 127] },
      teamAssignment: { discovery_confidence: null },
    },
  };

  function makeDemoFrame(time, offset) {
    const seeds = [
      [45, 70, 1],
      [83, 42, 1],
      [118, 92, 1],
      [156, 58, 1],
      [204, 104, 1],
      [72, 112, 2],
      [134, 32, 2],
      [178, 82, 2],
      [236, 48, 2],
      [266, 116, null],
    ];
    return {
      frameIndex: Math.round(time * 15),
      timeSeconds: time,
      players: seeds.map(([x, y, teamId], index) => ({
        id: index + 1,
        x: Math.max(8, Math.min(292, x + ((offset * (index % 3)) % 17) - 6)),
        y: Math.max(8, Math.min(153, y + ((offset * (index % 4)) % 13) - 5)),
        teamId,
        isHolder: index === (offset + 2) % 9,
      })),
    };
  }

  async function loadPermanentDemoAnalysis() {
    const response = await fetch(permanentDemoAssets.analysisUrl, { cache: "no-cache" });
    if (!response.ok) throw new Error("The sample analysis could not be loaded.");
    const analysis = await response.json();
    if (
      analysis?.schemaVersion !== 1 ||
      !Array.isArray(analysis.events) ||
      !Array.isArray(analysis.frames) ||
      !Number.isFinite(Number(analysis.source?.durationSeconds))
    ) {
      throw new Error("The sample analysis file is invalid.");
    }
    return analysis;
  }

  async function boot() {
    app.setAttribute("aria-busy", "true");
    if (demoMode) {
      state.session = { email: "analyst@example.com", expiresAt: new Date(Date.now() + 8 * 3600e3).toISOString() };
      state.csrfToken = "local-demo";
      if (demoMode === "signin") {
        state.session = null;
        state.view = "auth";
      } else if (demoMode === "processing") {
        state.view = "processing";
        state.job = demoJob("processing", "Analyzing players, ball, and court");
      } else if (demoMode === "colors") {
        state.view = "colors";
        state.job = demoJob("needs_team_colors", "Team colors required");
      } else if (demoMode === "error") {
        state.view = "failure";
        state.job = demoJob("failed", "Analysis failed");
        state.job.errorMessage = "The worker stopped before it could render the result.";
      } else if (demoMode === "review") {
        state.view = "review";
        state.job = demoJob("complete", "Review ready");
        if (permanentDemo) {
          try {
            state.analysis = await loadPermanentDemoAnalysis();
            state.downloads = {
              videoUrl: permanentDemoAssets.videoUrl,
              playbackUrl: permanentDemoAssets.videoUrl,
              analysisUrl: permanentDemoAssets.analysisUrl,
            };
            state.job.durationSeconds = state.analysis.source.durationSeconds;
          } catch (error) {
            state.view = "failure";
            state.job = demoJob("failed", "Sample analysis unavailable");
            state.job.errorMessage = error.message;
          }
        } else {
          state.analysis = demoAnalysis;
          state.downloads = { videoUrl: "", playbackUrl: "", analysisUrl: "" };
        }
        const firstEvent = state.analysis?.events?.[0] || null;
        state.selectedEventId = firstEvent?.id || null;
        state.currentTime = Number(firstEvent?.timeSeconds || 0);
      } else {
        state.view = "upload";
      }
      render();
      return;
    }

    try {
      const session = await api("/auth/session", { method: "GET", allowUnauthorized: true });
      if (!session || !session.authenticated) {
        state.view = "auth";
      } else {
        state.session = session;
        state.csrfToken = session.csrfToken;
        const activeJob = window.localStorage.getItem(activeJobKey);
        if (activeJob) {
          const resumed = await loadJob(activeJob, true);
          if (!resumed) state.view = "upload";
        } else {
          state.view = "upload";
        }
      }
    } catch (error) {
      state.view = "auth";
      if (error.status && error.status !== 401) {
        state.message = { type: "error", text: "CourtVision could not reach the beta service. Try again." };
      }
    }
    render();
  }

  function demoJob(status, stage) {
    const now = Date.now();
    return {
      id: "local-demo-job",
      status,
      stage,
      filename: "synthetic-beta-clip.mp4",
      durationSeconds: 30,
      createdAt: new Date(now - 8 * 60e3).toISOString(),
      updatedAt: new Date(now - 20e3).toISOString(),
      expiresAt: new Date(now + 24 * 3600e3).toISOString(),
      teamColorReason: "Automatic jersey discovery needs two distinct primary colors.",
    };
  }

  function render() {
    clearPoll();
    app.setAttribute("aria-busy", state.busy ? "true" : "false");
    if (state.view === "loading") {
      app.innerHTML = `<div class="loading-screen"><img class="loading-mark" src="assets/mark.svg" alt="CourtVision is loading" /></div>`;
      return;
    }
    if (state.view === "auth") renderAuth();
    else if (state.view === "upload") renderUpload();
    else if (state.view === "processing") renderProcessing();
    else if (state.view === "colors") renderTeamColors();
    else if (state.view === "review") renderReview();
    else if (state.view === "failure") renderFailure();
    else renderUpload();
    app.removeAttribute("aria-busy");
    requestAnimationFrame(() => {
      const heading = app.querySelector("h1");
      if (heading && !heading.hasAttribute("tabindex")) heading.setAttribute("tabindex", "-1");
    });
  }

  function renderAuth() {
    const codeStep = state.authStep === "code";
    app.innerHTML = `
      <main class="auth-view view">
        <section class="auth-scene" aria-labelledby="auth-title">
          <a class="auth-brand" href="./" aria-label="CourtVision home">
            <img src="assets/mark.svg" alt="" />
            <span class="brand">CourtVision</span>
            <span class="status-chip">Private beta</span>
          </a>
          <div class="auth-copy">
            <h1 id="auth-title">Review the play. Keep the uncertainty.</h1>
            <p>CourtVision turns one basketball clip into an annotated replay, tactical court, and timecoded event rundown built for evidence—not automatic decisions.</p>
          </div>
          <ul class="auth-proof" aria-label="Beta boundaries">
            <li>Invite-only access</li>
            <li>30-second clip limit</li>
            <li>Automatic deletion after 24 hours</li>
          </ul>
        </section>
        <section class="auth-panel" aria-labelledby="signin-heading">
          <div class="auth-form-wrap">
            <h2 id="signin-heading">${codeStep ? "Check your email" : "Enter your email"}</h2>
            <p>${
              codeStep
                ? `We sent a six-digit sign-in code to <strong>${escapeHtml(state.email)}</strong>.`
                : "Use the address approved for the CourtVision beta. There is no public signup."
            }</p>
            ${messageMarkup()}
            ${codeStep ? codeForm() : emailForm()}
          </div>
        </section>
      </main>
      ${toastRegion()}
    `;
    if (codeStep) {
      const form = app.querySelector("#code-form");
      form.addEventListener("submit", verifyCode);
      app.querySelector("#change-email").addEventListener("click", () => {
        state.authStep = "email";
        state.message = null;
        render();
      });
      app.querySelector("#resend-code").addEventListener("click", requestCode);
      app.querySelector("#code").focus();
    } else {
      app.querySelector("#email-form").addEventListener("submit", requestCode);
      app.querySelector("#email").focus();
    }
  }

  function emailForm() {
    return `
      <form id="email-form" class="form-stack" novalidate>
        <div class="field">
          <label class="field-label" for="email">Approved email</label>
          <input class="input" id="email" name="email" type="email" inputmode="email" autocomplete="email" required maxlength="254" value="${escapeHtml(state.email)}" placeholder="you@example.com" />
        </div>
        <button class="button button-primary" type="submit" ${state.busy ? "disabled" : ""}>
          ${icon("mail")}<span>${state.busy ? "Sending code…" : "Email me a code"}</span>
        </button>
        <p class="field-hint">For privacy, CourtVision gives the same response whether or not an address is allowlisted.</p>
      </form>
    `;
  }

  function codeForm() {
    return `
      <form id="code-form" class="form-stack" novalidate>
        <div class="field">
          <label class="field-label" for="code">Six-digit code</label>
          <input class="input code-input" id="code" name="code" type="text" inputmode="numeric" autocomplete="one-time-code" pattern="[0-9]{6}" minlength="6" maxlength="6" required aria-describedby="code-hint" />
          <p class="field-hint" id="code-hint">Codes expire after 10 minutes and work once.</p>
        </div>
        <button class="button button-primary" type="submit" ${state.busy ? "disabled" : ""}>
          ${icon("lock")}<span>${state.busy ? "Checking code…" : "Enter CourtVision"}</span>
        </button>
        <div class="form-actions">
          <button class="button button-secondary" id="resend-code" type="button" ${state.busy ? "disabled" : ""}>Send another code</button>
          <button class="button button-quiet" id="change-email" type="button">Use a different email</button>
        </div>
      </form>
    `;
  }

  async function requestCode(event) {
    event.preventDefault();
    const emailInput = app.querySelector("#email");
    if (emailInput) {
      if (!emailInput.reportValidity()) return;
      state.email = emailInput.value.trim().toLowerCase();
    }
    state.busy = true;
    state.message = null;
    renderAuth();
    try {
      if (demoMode) {
        state.authStep = "code";
        state.message = { type: "success", text: "Local preview: use any six-digit code." };
        return;
      }
      const result = await api("/auth/request-code", {
        method: "POST",
        body: { email: state.email },
        publicRequest: true,
      });
      state.authStep = "code";
      state.message = { type: "success", text: result.message };
    } catch (error) {
      state.message = { type: "error", text: error.message };
    } finally {
      state.busy = false;
      render();
    }
  }

  async function verifyCode(event) {
    event.preventDefault();
    const input = app.querySelector("#code");
    if (!input.reportValidity()) return;
    state.busy = true;
    state.message = null;
    renderAuth();
    try {
      if (demoMode) {
        state.session = { email: state.email || "coach@example.com" };
        state.csrfToken = "local-demo";
        state.view = "upload";
        return;
      }
      const session = await api("/auth/verify-code", {
        method: "POST",
        body: { email: state.email, code: input.value },
        publicRequest: true,
      });
      state.session = session;
      state.csrfToken = session.csrfToken;
      const retainedJob = window.localStorage.getItem(activeJobKey);
      if (retainedJob) {
        const resumed = await loadJob(retainedJob, true);
        if (!resumed) state.view = "upload";
      } else {
        state.view = "upload";
      }
    } catch (error) {
      state.message = { type: "error", text: error.message };
    } finally {
      state.busy = false;
      render();
    }
  }

  function renderUpload() {
    app.innerHTML = `
      <div class="app-shell">
        ${topbar()}
        <main class="workspace-view view" aria-labelledby="upload-title">
          <header class="workspace-heading">
            <div>
              <h1 id="upload-title">Put one clip on the desk.</h1>
              <p>Use a bounded game segment. CourtVision will preserve unknowns when evidence is weak.</p>
            </div>
            ${limitStrip()}
          </header>
          ${messageMarkup()}
          <form id="upload-form" class="upload-desk" novalidate>
            <section class="upload-monitor" aria-labelledby="drop-title">
              <div class="drop-zone" id="drop-zone" data-dragging="false">
                <div class="drop-zone-inner">
                  <div class="drop-symbol">${icon("upload")}</div>
                  <h2 id="drop-title">Choose a basketball clip</h2>
                  <p>MP4, MOV, or WebM. CourtVision reads the first bounded selection and rejects clips outside the beta profile.</p>
                  <label class="button button-primary" for="video-file">${icon("folder")}<span>Choose video</span></label>
                  <input class="file-input" id="video-file" name="video" type="file" accept="video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm" required />
                </div>
              </div>
            </section>
            <aside class="upload-rundown" aria-labelledby="upload-rundown-title">
              <h2 class="panel-title" id="upload-rundown-title"><span>Job rundown</span><span class="timecode">00:30 max</span></h2>
              ${selectedFileMarkup()}
              <button class="button button-primary" id="analyze-button" type="submit" ${!state.selectedFile || state.busy ? "disabled" : ""}>
                ${icon("arrow")}<span>${state.busy ? "Preparing upload…" : "Analyze clip"}</span>
              </button>
              <p class="field-hint">Beta analytics are experimental. Review every result against the source play.</p>
            </aside>
          </form>
        </main>
        ${toastRegion()}
      </div>
    `;
    bindGlobalActions();
    const input = app.querySelector("#video-file");
    input.addEventListener("change", () => selectFile(input.files[0]));
    const drop = app.querySelector("#drop-zone");
    ["dragenter", "dragover"].forEach((name) =>
      drop.addEventListener(name, (event) => {
        event.preventDefault();
        drop.dataset.dragging = "true";
      }),
    );
    ["dragleave", "drop"].forEach((name) =>
      drop.addEventListener(name, (event) => {
        event.preventDefault();
        drop.dataset.dragging = "false";
      }),
    );
    drop.addEventListener("drop", (event) => {
      const file = event.dataTransfer.files[0];
      if (file) selectFile(file);
    });
    app.querySelector("#upload-form").addEventListener("submit", createAndUploadJob);
  }

  function limitStrip() {
    return `
      <dl class="limit-strip" aria-label="Current beta processing limits">
        <div><dt>Clip</dt><dd>≤ ${config.maxDurationSeconds}s</dd></div>
        <div><dt>Analysis</dt><dd>${config.targetFps} FPS</dd></div>
        <div><dt>Width</dt><dd>${config.maxWidth}px</dd></div>
      </dl>
    `;
  }

  function selectedFileMarkup() {
    if (!state.selectedFile) {
      return `
        <div class="selected-file">
          <p>No clip selected.</p>
          <p class="field-hint">The filename and limits will appear here before anything uploads.</p>
        </div>
      `;
    }
    return `
      <div class="selected-file">
        <h3>${escapeHtml(state.selectedFile.name)}</h3>
        <dl class="file-facts">
          <dt>Duration</dt><dd>${formatTime(state.selectedFileDuration, true)}</dd>
          <dt>Size</dt><dd>${formatBytes(state.selectedFile.size)}</dd>
          <dt>Retention</dt><dd>${config.resultRetentionHours} hours</dd>
        </dl>
        ${
          state.busy
            ? `<div class="upload-progress"><label for="upload-progress">Uploading ${Math.round(state.uploadProgress)}%</label><progress id="upload-progress" max="100" value="${state.uploadProgress}">${state.uploadProgress}%</progress></div>`
            : ""
        }
      </div>
    `;
  }

  async function selectFile(file) {
    state.message = null;
    if (!file) return;
    if (!isAllowedVideo(file)) {
      state.selectedFile = null;
      state.message = { type: "error", text: "Choose an MP4, MOV, or WebM video." };
      render();
      return;
    }
    if (file.size > config.maxUploadBytes) {
      state.selectedFile = null;
      state.message = { type: "error", text: `This video exceeds the ${formatBytes(config.maxUploadBytes)} beta limit.` };
      render();
      return;
    }
    try {
      const duration = await readVideoDuration(file);
      if (!Number.isFinite(duration) || duration <= 0) throw new Error("CourtVision could not read this video's duration.");
      if (duration > config.maxDurationSeconds + 0.05) {
        throw new Error(`Choose a clip no longer than ${config.maxDurationSeconds} seconds.`);
      }
      state.selectedFile = file;
      state.selectedFileDuration = duration;
    } catch (error) {
      state.selectedFile = null;
      state.selectedFileDuration = 0;
      state.message = { type: "error", text: error.message };
    }
    render();
  }

  function readVideoDuration(file) {
    return new Promise((resolve, reject) => {
      const video = document.createElement("video");
      const url = URL.createObjectURL(file);
      video.preload = "metadata";
      video.onloadedmetadata = () => {
        const duration = video.duration;
        URL.revokeObjectURL(url);
        resolve(duration);
      };
      video.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("CourtVision could not read this video. Try a standard MP4 export."));
      };
      video.src = url;
    });
  }

  async function createAndUploadJob(event) {
    event.preventDefault();
    if (!state.selectedFile || state.busy) return;
    state.busy = true;
    state.uploadProgress = 0;
    state.message = null;
    renderUpload();
    try {
      if (demoMode) {
        await delay(450);
        state.job = demoJob("processing", "Queued for analysis");
      } else {
        const response = await api("/jobs", {
          method: "POST",
          body: {
            filename: state.selectedFile.name,
            contentType: state.selectedFile.type || contentTypeFromName(state.selectedFile.name),
            sizeBytes: state.selectedFile.size,
            durationSeconds: state.selectedFileDuration,
          },
        });
        state.job = response.job;
        window.localStorage.setItem(activeJobKey, state.job.id);
        await uploadToS3(response.upload, state.selectedFile, (progress) => {
          state.uploadProgress = progress;
          const progressElement = app.querySelector("#upload-progress");
          if (progressElement) progressElement.value = progress;
          const label = progressElement?.previousElementSibling;
          if (label) label.textContent = `Uploading ${Math.round(progress)}%`;
        });
        const started = await api(`/jobs/${state.job.id}/start`, { method: "POST", body: {} });
        state.job = started.job;
      }
      state.view = "processing";
      state.selectedFile = null;
      state.selectedFileDuration = 0;
    } catch (error) {
      state.message = { type: "error", text: error.message };
    } finally {
      state.busy = false;
      render();
    }
  }

  function uploadToS3(upload, file, onProgress) {
    return new Promise((resolve, reject) => {
      const form = new FormData();
      Object.entries(upload.fields).forEach(([key, value]) => form.append(key, value));
      form.append("file", file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", upload.url, true);
      xhr.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable) onProgress((event.loaded / event.total) * 100);
      });
      xhr.addEventListener("load", () => {
        if (xhr.status >= 200 && xhr.status < 300) resolve();
        else reject(new Error("The upload did not complete. Check your connection and try again."));
      });
      xhr.addEventListener("error", () => reject(new Error("The upload lost its connection. Try again.")));
      xhr.send(form);
    });
  }

  function renderProcessing() {
    const stages = processingStages(state.job);
    app.innerHTML = `
      <div class="app-shell">
        ${topbar()}
        <main class="workspace-view view" aria-labelledby="processing-title">
          <div class="processing-desk">
            <section class="processing-monitor" aria-live="polite">
              <h1 id="processing-title">${escapeHtml(state.job?.stage || "Preparing analysis")}</h1>
              <p>${escapeHtml(state.job?.filename || "Uploaded clip")} · results remain available for ${config.resultRetentionHours} hours.</p>
              <div class="processing-ruler" aria-hidden="true"><span class="processing-beam"></span></div>
            </section>
            <aside class="processing-sheet" aria-labelledby="stage-title">
              <h2 class="panel-title" id="stage-title"><span>Processing rundown</span><span class="timecode">${formatTime(state.job?.durationSeconds || 30, true)}</span></h2>
              <ol class="stage-list">
                ${stages
                  .map(
                    (stage, index) => `
                      <li data-state="${stage.state}">
                        <span class="stage-marker" aria-hidden="true">${stage.state === "complete" ? icon("check") : index + 1}</span>
                        <div><strong>${stage.name}</strong><span>${stage.description}</span></div>
                      </li>`,
                  )
                  .join("")}
              </ol>
              <p class="field-hint">You can close this tab. CourtVision will recover the active session after you return and sign in.</p>
              ${demoMode ? '<button class="button button-primary" id="demo-complete" type="button">Open synthetic review</button>' : ""}
            </aside>
          </div>
        </main>
        ${toastRegion()}
      </div>
    `;
    bindGlobalActions();
    if (demoMode) {
      app.querySelector("#demo-complete").addEventListener("click", () => {
        state.job = demoJob("complete", "Review ready");
        state.analysis = demoAnalysis;
        state.downloads = { videoUrl: "#", playbackUrl: "", analysisUrl: "" };
        state.selectedEventId = demoEvents[1].id;
        state.currentTime = demoEvents[1].timeSeconds;
        state.view = "review";
        render();
      });
    } else {
      state.pollTimer = window.setTimeout(() => loadJob(state.job.id), config.pollIntervalMs);
    }
  }

  function processingStages(job) {
    const status = job?.status || "queued";
    const stageText = String(job?.stage || "").toLowerCase();
    const activeIndex = status === "queued" ? 1 : stageText.includes("final") ? 3 : status === "complete" ? 4 : 2;
    const definitions = [
      ["Upload received", "The source clip is stored in the private job prefix."],
      ["Worker queued", "A bounded GPU worker is reserved for this analysis."],
      ["Evidence pass", "Players, ball, teams, events, and court position are evaluated."],
      ["Review render", "The annotated video and machine-readable evidence are finalized."],
      ["Ready", "Download and structured failure reporting become available."],
    ];
    return definitions.map(([name, description], index) => ({
      name,
      description,
      state: index < activeIndex ? "complete" : index === activeIndex ? "active" : "pending",
    }));
  }

  async function loadJob(jobId, silent = false) {
    try {
      const response = await api(`/jobs/${jobId}`, { method: "GET" });
      state.job = response.job;
      if (state.job.status === "complete") {
        await loadReviewArtifacts();
      } else if (state.job.status === "needs_team_colors") {
        state.view = "colors";
      } else if (state.job.status === "failed") {
        state.view = "failure";
      } else {
        state.view = "processing";
      }
      if (!silent) render();
      return true;
    } catch (error) {
      if (error.status === 401) return false;
      if (error.status === 404 || error.status === 410) {
        window.localStorage.removeItem(activeJobKey);
        state.job = null;
        state.view = "upload";
        state.message = { type: "error", text: "The previous analysis session is no longer available. Upload a new clip." };
        if (!silent) render();
        return false;
      }
      if (!silent) {
        state.message = { type: "error", text: "CourtVision could not refresh the job. It will try again." };
        render();
      }
      return true;
    }
  }

  async function loadReviewArtifacts() {
    const downloads = await api(`/jobs/${state.job.id}/download`, { method: "GET" });
    const response = await fetch(downloads.analysisUrl, { cache: "no-store" });
    if (!response.ok) throw new Error("The analysis manifest could not be loaded.");
    state.analysis = await response.json();
    state.downloads = downloads;
    state.selectedEventId = state.analysis.events?.[0]?.id || null;
    state.currentTime = state.analysis.events?.[0]?.timeSeconds || 0;
    state.view = "review";
  }

  function renderTeamColors() {
    app.innerHTML = `
      <div class="app-shell">
        ${topbar()}
        <main class="workspace-view view" aria-labelledby="colors-title">
          <header class="workspace-heading">
            <div>
              <h1 id="colors-title">The jerseys need one human cue.</h1>
              <p>${escapeHtml(state.job?.teamColorReason || "Automatic discovery could not separate two reliable jersey groups.")}</p>
            </div>
            ${limitStrip()}
          </header>
          ${messageMarkup()}
          <div class="upload-desk">
            <section class="upload-monitor">
              <div class="drop-zone">
                <div class="drop-zone-inner">
                  <div class="drop-symbol">${icon("evidence")}</div>
                  <h2>Choose primary jersey colors</h2>
                  <p>These colors guide team prototypes. They do not bypass crop rejection, referee filtering, or unknown assignments.</p>
                </div>
              </div>
            </section>
            <aside class="upload-rundown paper-surface">
              <h2 class="panel-title"><span>Team cue</span><span class="timecode">Required</span></h2>
              <form id="color-form" class="color-form">
                <div class="color-fields">
                  <div class="field"><label class="field-label" for="team-1-color">Team one</label><input class="color-input" id="team-1-color" name="team1Color" type="color" value="#F4F5F7" /></div>
                  <div class="field"><label class="field-label" for="team-2-color">Team two</label><input class="color-input" id="team-2-color" name="team2Color" type="color" value="#1E55D6" /></div>
                </div>
                <button class="button button-primary" type="submit" ${state.busy ? "disabled" : ""}>${icon("refresh")}<span>${state.busy ? "Re-queuing…" : "Continue analysis"}</span></button>
              </form>
            </aside>
          </div>
        </main>
        ${toastRegion()}
      </div>
    `;
    bindGlobalActions();
    app.querySelector("#color-form").addEventListener("submit", submitTeamColors);
  }

  async function submitTeamColors(event) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const team1Color = String(data.get("team1Color")).toUpperCase();
    const team2Color = String(data.get("team2Color")).toUpperCase();
    if (team1Color === team2Color) {
      state.message = { type: "error", text: "Choose two distinct primary jersey colors." };
      render();
      return;
    }
    state.busy = true;
    renderTeamColors();
    try {
      if (demoMode) state.job = demoJob("processing", "Queued with team colors");
      else {
        const response = await api(`/jobs/${state.job.id}/team-colors`, {
          method: "POST",
          body: { team1Color, team2Color },
        });
        state.job = response.job;
      }
      state.view = "processing";
    } catch (error) {
      state.message = { type: "error", text: error.message };
    } finally {
      state.busy = false;
      render();
    }
  }

  function renderFailure() {
    app.innerHTML = `
      <div class="app-shell">
        ${topbar()}
        <main class="workspace-view view" aria-labelledby="failure-title">
          <div class="processing-desk">
            <section class="processing-monitor">
              <h1 id="failure-title">This run stopped before review.</h1>
              <p>${escapeHtml(state.job?.errorMessage || "CourtVision could not complete the bounded analysis job.")}</p>
              <div class="form-actions">
                <button class="button button-primary" id="retry-job" type="button">${icon("refresh")}<span>Retry analysis</span></button>
                <button class="button button-secondary" id="new-upload" type="button">Choose another clip</button>
              </div>
            </section>
            <aside class="processing-sheet">
              <h2 class="panel-title"><span>Recovery</span><span>Beta</span></h2>
              <p>The uploaded clip remains private until its deletion deadline. Retrying reuses the same bounded source.</p>
              <p class="field-hint">If the failure repeats, choose another clip and include the processing failure in your beta feedback.</p>
            </aside>
          </div>
        </main>
        ${toastRegion()}
      </div>
    `;
    bindGlobalActions();
    app.querySelector("#new-upload").addEventListener("click", resetJob);
    app.querySelector("#retry-job").addEventListener("click", retryJob);
  }

  async function retryJob() {
    state.busy = true;
    try {
      if (demoMode) state.job = demoJob("processing", "Queued for another attempt");
      else {
        const response = await api(`/jobs/${state.job.id}/start`, { method: "POST", body: {} });
        state.job = response.job;
      }
      state.view = "processing";
    } catch (error) {
      state.message = { type: "error", text: error.message };
    } finally {
      state.busy = false;
      render();
    }
  }

  function renderReview() {
    const analysis = state.analysis || demoAnalysis;
    const duration = analysis.source?.durationSeconds || state.job?.durationSeconds || 30;
    const events = analysis.events || [];
    const selected = events.find((event) => event.id === state.selectedEventId) || events[0] || null;
    if (selected && !state.selectedEventId) state.selectedEventId = selected.id;
    app.innerHTML = `
      <div class="app-shell">
        ${topbar(true)}
        <main class="review-view view" aria-label="CourtVision beta review workspace">
          <div class="review-desk">
            <section class="replay-column" aria-label="Annotated replay and tactical court">
              <div class="video-monitor">
                ${
                  state.downloads?.playbackUrl
                    ? `<video id="result-video" controls playsinline preload="${permanentDemo ? "auto" : "metadata"}" poster="${escapeHtml(permanentDemo ? permanentDemoAssets.posterUrl : "")}" src="${escapeHtml(state.downloads.playbackUrl)}"></video>`
                    : `<div class="demo-frame" id="demo-video" role="img" aria-label="Synthetic local basketball frame for interface review"></div><span class="synthetic-label">Synthetic local preview</span>`
                }
                ${demoMode && !state.downloads?.playbackUrl ? '<div class="tracking-overlay" aria-hidden="true"><span class="tracking-mark mark-one"></span><span class="tracking-mark mark-two"></span></div>' : ""}
                <span class="timecode-badge" id="current-timecode">${formatTime(state.currentTime)}</span>
                ${tacticalDockMarkup(analysis)}
              </div>
              ${timelineMarkup(events, duration)}
              ${evidenceMarkup(selected, analysis)}
            </section>
            <aside class="rundown-rail" aria-labelledby="rundown-title">
              <header class="rundown-header">
                <h2 id="rundown-title"><span>Event rundown</span><span class="timecode">${formatTime(duration, true)}</span></h2>
                <p>Select a cue to inspect the same moment in the replay and tactical court.</p>
              </header>
              ${eventListMarkup(events, selected)}
              <footer class="rundown-footer">
                <div class="rundown-legend"><span><i class="legend-shape"></i>Candidate</span><span><i class="legend-shape unknown"></i>Unknown or tentative</span></div>
                <span>${escapeHtml(analysis.disclaimer || "Experimental beta analysis. Review every result against the source play.")}</span>
              </footer>
            </aside>
          </div>
          ${permanentDemo ? "" : reportDialogMarkup(selected)}
          <p class="sr-only" id="sync-announcement" aria-live="polite" aria-atomic="true"></p>
        </main>
        ${toastRegion()}
      </div>
    `;
    bindGlobalActions();
    bindReviewActions(duration);
    updateCourtAndTimeline();
  }

  function topbar(review = false) {
    if (embeddedDemo) return "";
    if (permanentDemo) {
      return `
        <header class="topbar">
          <div class="brand-lockup">
            <a class="brand" href="./" aria-label="CourtVision home">CourtVision</a>
            <span class="brand-divider" aria-hidden="true"></span>
            <span class="status-chip">Beta analysis</span>
          </div>
          <div class="topbar-center">${icon("clock")} <span>Preprocessed sample analysis</span></div>
          <nav class="topbar-actions" aria-label="Demo actions">
            <a class="button button-secondary" href="./">About CourtVision</a>
          </nav>
        </header>
      `;
    }
    const expiration = state.job?.expiresAt
      ? `Deletes ${relativeExpiration(state.job.expiresAt)}`
      : `${config.resultRetentionHours}-hour retention`;
    const canDownload = review && state.downloads?.videoUrl;
    return `
      <header class="topbar">
        <div class="brand-lockup">
          <a class="brand" href="./" aria-label="CourtVision home">CourtVision</a>
          <span class="brand-divider" aria-hidden="true"></span>
          <span class="status-chip">Beta analysis</span>
        </div>
        <div class="topbar-center">${icon("clock")} <span>${escapeHtml(expiration)}</span></div>
        <nav class="topbar-actions" aria-label="Session actions">
          ${
            canDownload
              ? `<a class="button button-primary" href="${escapeHtml(state.downloads.videoUrl)}" download>${icon("download")}<span>Download video</span></a>`
              : ""
          }
          ${review ? `<button class="button button-secondary" id="report-issue" type="button">${icon("flag")}<span>Report issue</span></button>` : ""}
          <button class="button button-quiet" id="sign-out" type="button">${icon("signout")}<span>Sign out</span></button>
        </nav>
      </header>
    `;
  }

  function tacticalDockMarkup(analysis) {
    const frame = frameAtTime(analysis, state.currentTime);
    return `
      <section class="tactical-dock" aria-labelledby="court-title">
        <header class="dock-header"><span id="court-title">Tactical court</span><span id="court-time">${formatTime(state.currentTime)}</span></header>
        <div class="court-stage" id="court-stage" role="img" aria-label="Player positions on the tactical court at ${formatTime(state.currentTime)}">
          ${courtMarkersMarkup(frame, analysis)}
        </div>
        <ul class="court-legend" aria-label="Tactical court legend"><li><i class="legend-dot one"></i>Display team one</li><li><i class="legend-dot two"></i>Display team two</li><li><i class="legend-dot unknown"></i>Unknown</li></ul>
      </section>
    `;
  }

  function courtMarkersMarkup(frame, analysis) {
    const width = analysis.court?.width || 300;
    const height = analysis.court?.height || 161;
    if (!frame?.players?.length) return '<span class="court-unavailable">Tactical positions unavailable at this moment</span>';
    const frameIndex = Number(frame.frameIndex);
    const mirrorX = (analysis.court?.mirrorXFrameRanges || []).some(
      ([firstFrame, lastFrame]) => frameIndex >= Number(firstFrame) && frameIndex <= Number(lastFrame),
    );
    return frame.players
      .map((player) => {
        const displayX = mirrorX ? width - Number(player.x) : Number(player.x);
        const teamClass = player.teamId === 1 ? "team-1" : player.teamId === 2 ? "team-2" : "unknown";
        const holderClass = player.isHolder ? "holder" : "";
        const label = `Internal track ${player.id}, ${player.teamId ? `estimated display team ${player.teamId}` : "unknown team"}${player.isHolder ? ", possible holder estimate" : ""}`;
        return `<span class="player-marker ${teamClass} ${holderClass}" style="left:${clamp((displayX / width) * 100, 1, 99)}%;top:${clamp((Number(player.y) / height) * 100, 2, 98)}%" title="${escapeHtml(label)}">T${escapeHtml(player.id)}</span>`;
      })
      .join("");
  }

  function timelineMarkup(events, duration) {
    const playhead = clamp((state.currentTime / duration) * 100, 0, 100);
    return `
      <section class="timeline-console" aria-label="Review timeline">
        <div class="timeline-scale">
          ${events
            .map(
              (event) => `<span class="timeline-event ${event.status === "unknown" ? "unknown" : ""}" style="left:${clamp((event.timeSeconds / duration) * 100, 0, 100)}%" aria-hidden="true"></span>`,
            )
            .join("")}
          <span class="timeline-playhead" id="timeline-playhead" style="left:${playhead}%" aria-hidden="true"></span>
          <input class="timeline-input" id="timeline-input" type="range" min="0" max="${duration}" step="0.05" value="${state.currentTime}" aria-label="Replay position" aria-valuetext="${formatTime(state.currentTime)}" />
          <div class="timeline-labels" aria-hidden="true"><span>00:00</span><span>${formatTime(duration / 2, true)}</span><span>${formatTime(duration, true)}</span></div>
        </div>
      </section>
    `;
  }

  function evidenceMarkup(event, analysis) {
    const unavailable = tacticalUnavailableCount(analysis);
    return `
      <details class="evidence-drawer">
        <summary>${icon("evidence")}<span>Evidence and unknowns</span>${icon("chevron")}</summary>
        <div class="evidence-content" id="evidence-content">${evidenceContentMarkup(event, unavailable)}</div>
      </details>
    `;
  }

  function evidenceContentMarkup(event, unavailable) {
    return `
      <div><strong>Selected cue</strong><span>${event ? `${escapeHtml(eventLabel(event))} at ${formatTime(event.timeSeconds)}` : "No candidate event at this moment"}</span></div>
      <div><strong>Review state</strong><span>${event?.status === "unknown" ? "Unknown—insufficient evidence" : "Candidate—requires video review"}</span></div>
      <div><strong>Evidence state</strong><span>Qualitative candidate or unknown state; requires video review</span></div>
      <div><strong>Tactical availability</strong><span>${unavailable ? `${unavailable} diagnostic frame references are present` : "No tactical fallback is reported at this cue"}</span></div>
    `;
  }

  function eventListMarkup(events, selected) {
    if (!events.length) {
      return '<div class="empty-rundown"><div><h3>No reliable events</h3><p>CourtVision kept the event list empty rather than inventing a transition.</p></div></div>';
    }
    return `
      <ol class="event-list">
        ${events
          .map(
            (event) => `
              <li>
                <button class="event-button" type="button" data-event-id="${escapeHtml(event.id)}" aria-current="${event.id === selected?.id ? "true" : "false"}">
                  <span class="event-time">${formatTime(event.timeSeconds)}</span>
                  <span class="event-symbol ${event.status === "unknown" ? "unknown" : ""}">${event.status === "unknown" ? icon("info") : icon("play")}</span>
                  <span class="event-name">${escapeHtml(eventLabel(event))}<small>${escapeHtml(eventDescription(event))}</small></span>
                  <span class="event-state">${event.status === "unknown" ? "Unknown" : "Candidate"}</span>
                </button>
              </li>`,
          )
          .join("")}
      </ol>
    `;
  }

  function reportDialogMarkup(selected) {
    return `
      <dialog id="report-dialog" aria-labelledby="report-title">
        <form id="report-form" method="dialog">
          <header class="dialog-header">
            <div><h2 id="report-title">Report a beta failure</h2><p class="field-hint">The report is attached to this job and timecode, not the retained video after deletion.</p></div>
            <button class="button button-quiet" id="close-report" type="button" aria-label="Close report dialog">${icon("close")}</button>
          </header>
          <div class="dialog-body form-stack">
            <div class="field"><label class="field-label" for="report-time">Timecode</label><input class="input" id="report-time" name="timeSeconds" type="number" min="0" max="${state.analysis?.source?.durationSeconds || 30}" step="0.1" value="${Number(state.currentTime).toFixed(1)}" required /></div>
            <div class="field"><label class="field-label" for="report-category">What failed?</label><select class="select" id="report-category" name="category" required><option value="">Choose a category</option><option value="ball_tracking">Ball tracking</option><option value="player_tracking">Player tracking</option><option value="team_assignment">Team assignment</option><option value="possession">Possession</option><option value="event_detection">Pass or interception</option><option value="tactical_view">Tactical court</option><option value="rendering">Rendered overlay</option><option value="processing">Processing reliability</option><option value="other">Other</option></select></div>
            <div class="field"><label class="field-label" for="report-notes">What should a reviewer see?</label><textarea class="textarea" id="report-notes" name="notes" rows="5" maxlength="2000" required placeholder="Describe the visible failure and the expected interpretation."></textarea><p class="field-hint">Do not include personal data or confidential team information.</p></div>
            <input id="report-event-id" type="hidden" name="eventId" value="${escapeHtml(selected?.id || "")}" />
          </div>
          <footer class="dialog-footer"><button class="button button-secondary" id="cancel-report" type="button">Cancel</button><button class="button button-primary" type="submit">${icon("flag")}<span>Save failure report</span></button></footer>
        </form>
      </dialog>
    `;
  }

  function bindReviewActions(duration) {
    const video = app.querySelector("#result-video");
    const timeline = app.querySelector("#timeline-input");
    if (video) {
      const initialVideoTime = clamp(state.currentTime, 0, duration);
      const syncInitialVideoTime = () => {
        video.currentTime = initialVideoTime;
      };
      if (video.readyState >= 1) syncInitialVideoTime();
      else video.addEventListener("loadedmetadata", syncInitialVideoTime, { once: true });
      video.addEventListener("timeupdate", () => {
        state.currentTime = video.currentTime;
        updateCourtAndTimeline();
      });
    }
    timeline.addEventListener("input", () => {
      state.currentTime = Number(timeline.value);
      if (video) video.currentTime = state.currentTime;
      updateCourtAndTimeline();
    });
    app.querySelectorAll("[data-event-id]").forEach((button) => {
      button.addEventListener("click", () => selectEvent(button.dataset.eventId, video));
    });
    const reportButton = app.querySelector("#report-issue");
    const dialog = app.querySelector("#report-dialog");
    if (reportButton && dialog) {
      reportButton.addEventListener("click", () => {
        const timeInput = app.querySelector("#report-time");
        timeInput.value = Number(state.currentTime).toFixed(1);
        app.querySelector("#report-event-id").value = state.selectedEventId || "";
        dialog.showModal();
      });
      app.querySelector("#close-report").addEventListener("click", () => dialog.close());
      app.querySelector("#cancel-report").addEventListener("click", () => dialog.close());
      app.querySelector("#report-form").addEventListener("submit", submitReport);
    }
  }

  function selectEvent(eventId, video) {
    const event = state.analysis.events.find((item) => item.id === eventId);
    if (!event) return;
    const previousCourtPositions = captureCourtMarkerPositions();
    state.selectedEventId = event.id;
    state.currentTime = Number(event.timeSeconds);
    if (video) video.currentTime = state.currentTime;
    app.querySelectorAll("[data-event-id]").forEach((button) =>
      button.setAttribute("aria-current", button.dataset.eventId === eventId ? "true" : "false"),
    );
    const evidence = app.querySelector("#evidence-content");
    const reportEvent = app.querySelector("#report-event-id");
    const announcement = app.querySelector("#sync-announcement");
    if (evidence) evidence.innerHTML = evidenceContentMarkup(event, tacticalUnavailableCount(state.analysis || demoAnalysis));
    if (reportEvent) reportEvent.value = event.id;
    if (announcement) announcement.textContent = `${eventLabel(event)} selected at ${formatTime(event.timeSeconds)}.`;
    updateCourtAndTimeline();
    animateCourtTransition(previousCourtPositions);
    runReviewSyncAnimation();
  }

  function captureCourtMarkerPositions() {
    return new Map(
      Array.from(app.querySelectorAll("#court-stage .player-marker")).map((marker) => [
        marker.textContent.trim(),
        marker.getBoundingClientRect(),
      ]),
    );
  }

  function animateCourtTransition(previousPositions) {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches || !previousPositions.size) return;
    requestAnimationFrame(() => {
      app.querySelectorAll("#court-stage .player-marker").forEach((marker) => {
        const previous = previousPositions.get(marker.textContent.trim());
        if (!previous) return;
        const current = marker.getBoundingClientRect();
        marker.animate(
          [
            {
              transform: `translate(calc(-50% + ${previous.left - current.left}px), calc(-50% + ${previous.top - current.top}px))`,
              opacity: 0.72,
            },
            { transform: "translate(-50%, -50%)", opacity: 1 },
          ],
          { duration: 340, easing: "cubic-bezier(0.16, 1, 0.3, 1)" },
        );
      });
    });
  }

  function runReviewSyncAnimation() {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const desk = app.querySelector(".review-desk");
    if (!desk) return;
    window.clearTimeout(state.syncAnimationTimer);
    desk.dataset.syncCount = String(Number(desk.dataset.syncCount || 0) + 1);
    desk.classList.remove("is-syncing");
    void desk.offsetWidth;
    desk.classList.add("is-syncing");
    state.syncAnimationTimer = window.setTimeout(() => desk.classList.remove("is-syncing"), 520);
  }

  function tacticalUnavailableCount(analysis) {
    return Object.values(analysis.diagnostics?.tacticalView || {})
      .filter(Array.isArray)
      .reduce((sum, entries) => sum + entries.length, 0);
  }

  function updateCourtAndTimeline() {
    const analysis = state.analysis || demoAnalysis;
    const duration = analysis.source?.durationSeconds || 30;
    const frame = frameAtTime(analysis, state.currentTime);
    const court = app.querySelector("#court-stage");
    if (court) {
      court.innerHTML = courtMarkersMarkup(frame, analysis);
      court.setAttribute(
        "aria-label",
        frame?.players?.length
          ? `Player positions on the tactical court at ${formatTime(state.currentTime)}`
          : `Tactical positions unavailable at ${formatTime(state.currentTime)}`,
      );
    }
    const time = app.querySelector("#court-time");
    const badge = app.querySelector("#current-timecode");
    const playhead = app.querySelector("#timeline-playhead");
    const input = app.querySelector("#timeline-input");
    if (time) time.textContent = formatTime(state.currentTime);
    if (badge) badge.textContent = formatTime(state.currentTime);
    if (playhead) playhead.style.left = `${clamp((state.currentTime / duration) * 100, 0, 100)}%`;
    if (input && document.activeElement !== input) input.value = state.currentTime;
    if (input) input.setAttribute("aria-valuetext", formatTime(state.currentTime));
  }

  async function submitReport(event) {
    event.preventDefault();
    const dialog = app.querySelector("#report-dialog");
    const form = event.currentTarget;
    if (!form.reportValidity()) return;
    const data = Object.fromEntries(new FormData(form).entries());
    data.timeSeconds = Number(data.timeSeconds);
    const submit = form.querySelector('button[type="submit"]');
    submit.disabled = true;
    try {
      if (!demoMode) {
        await api(`/jobs/${state.job.id}/reports`, { method: "POST", body: data });
      } else {
        await delay(250);
      }
      dialog.close();
      form.reset();
      showToast("Failure report saved for beta review.");
    } catch (error) {
      showToast(error.message, true);
    } finally {
      submit.disabled = false;
    }
  }

  function bindGlobalActions() {
    const signOut = app.querySelector("#sign-out");
    if (signOut) signOut.addEventListener("click", signOutUser);
  }

  async function signOutUser() {
    try {
      if (!demoMode) await api("/auth/sign-out", { method: "POST", body: {} });
    } catch (_error) {
      // The local session is still cleared when the service cannot respond.
    }
    clearPoll();
    state.session = null;
    state.csrfToken = null;
    state.job = null;
    state.analysis = null;
    state.downloads = null;
    state.view = "auth";
    state.authStep = "email";
    render();
  }

  function resetJob() {
    clearPoll();
    window.localStorage.removeItem(activeJobKey);
    state.job = null;
    state.analysis = null;
    state.downloads = null;
    state.selectedFile = null;
    state.message = null;
    state.view = "upload";
    render();
  }

  async function api(path, options = {}) {
    const method = options.method || "GET";
    const headers = { Accept: "application/json" };
    if (options.body !== undefined) headers["Content-Type"] = "application/json";
    if (!options.publicRequest && !["GET", "HEAD", "OPTIONS"].includes(method) && state.csrfToken) {
      headers["X-CourtVision-CSRF"] = state.csrfToken;
    }
    const response = await fetch(`${config.apiBaseUrl}${path}`, {
      method,
      headers,
      credentials: "same-origin",
      cache: "no-store",
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
    });
    let payload = null;
    try {
      payload = response.status === 204 ? null : await response.json();
    } catch (_error) {
      payload = null;
    }
    if (!response.ok) {
      const error = new Error(payload?.error?.message || "CourtVision could not complete the request.");
      error.status = response.status;
      error.code = payload?.error?.code;
      if (response.status === 401 && !options.allowUnauthorized) {
        clearPoll();
        state.session = null;
        state.csrfToken = null;
        state.view = "auth";
        state.authStep = "email";
        state.message = {
          type: "error",
          text: "Your sign-in expired. Sign in again to recover the retained analysis session.",
        };
        error.authenticationHandled = true;
        render();
      }
      throw error;
    }
    return payload;
  }

  function messageMarkup() {
    if (!state.message) return "";
    const error = state.message.type === "error";
    return `<p class="form-message ${error ? "form-message-error" : "form-message-success"}" role="${error ? "alert" : "status"}">${icon(error ? "alert" : "check")}<span>${escapeHtml(state.message.text)}</span></p>`;
  }

  function toastRegion() {
    return '<div class="toast-region" id="toast-region" aria-live="polite" aria-atomic="true"></div>';
  }

  function showToast(message, error = false) {
    const region = app.querySelector("#toast-region");
    if (!region) return;
    region.innerHTML = `<div class="toast" role="${error ? "alert" : "status"}">${icon(error ? "alert" : "check")}<span>${escapeHtml(message)}</span></div>`;
    window.clearTimeout(state.toastTimer);
    state.toastTimer = window.setTimeout(() => {
      if (region.isConnected) region.innerHTML = "";
    }, 5000);
  }

  function frameAtTime(analysis, time) {
    const frames = analysis.frames || [];
    if (!frames.length) return null;
    let best = frames[0];
    let distance = Math.abs(Number(best.timeSeconds) - time);
    for (const frame of frames) {
      const nextDistance = Math.abs(Number(frame.timeSeconds) - time);
      if (nextDistance < distance) {
        best = frame;
        distance = nextDistance;
      }
      if (Number(frame.timeSeconds) > time && nextDistance > distance) break;
    }
    return best;
  }

  function eventLabel(event) {
    const labels = {
      pass: "Pass candidate",
      interception: "Interception candidate",
    };
    return labels[event.type] || "Event candidate";
  }

  function eventDescription(event) {
    if (event.status === "unknown") return "Evidence remains unresolved";
    if (event.fromTeamId && event.toTeamId && event.fromTeamId !== event.toTeamId) {
      return `Display team ${event.fromTeamId} to display team ${event.toTeamId}`;
    }
    if (event.toTeamId) return `Display team ${event.toTeamId}`;
    return "Review against the source play";
  }

  function formatTime(seconds, compact = false) {
    const safe = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(safe / 60);
    const remainder = safe - minutes * 60;
    const whole = Math.floor(remainder);
    const tenths = Math.floor((remainder - whole) * 10 + 0.0001);
    if (compact) return `${String(minutes).padStart(2, "0")}:${String(whole).padStart(2, "0")}`;
    return `${String(minutes).padStart(2, "0")}:${String(whole).padStart(2, "0")}.${tenths}`;
  }

  function formatBytes(bytes) {
    const value = Number(bytes) || 0;
    if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GiB`;
    if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MiB`;
    return `${Math.max(1, Math.round(value / 1024))} KiB`;
  }

  function relativeExpiration(value) {
    const date = new Date(value);
    const hours = Math.max(0, Math.round((date.getTime() - Date.now()) / 3600000));
    if (hours < 1) return "within the hour";
    if (hours === 1) return "in 1 hour";
    return `in ${hours} hours`;
  }

  function isAllowedVideo(file) {
    return ["video/mp4", "video/quicktime", "video/webm"].includes(file.type) || /\.(mp4|mov|webm)$/i.test(file.name);
  }

  function contentTypeFromName(name) {
    if (/\.mov$/i.test(name)) return "video/quicktime";
    if (/\.webm$/i.test(name)) return "video/webm";
    return "video/mp4";
  }

  function titleCase(value) {
    return String(value)
      .replace(/[_-]+/g, " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function clearPoll() {
    if (state.pollTimer) window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }

  function delay(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
  }

  boot();
})();

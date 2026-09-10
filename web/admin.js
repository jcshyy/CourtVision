(() => {
  "use strict";

  const config = Object.assign({ apiBaseUrl: "/api" }, window.COURTVISION_CONFIG || {});
  const root = document.querySelector("#admin-app");
  const localDemo = ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname) && new URLSearchParams(window.location.search).get("demo") === "1";
  const state = { session: null, overview: null, query: "", error: null, loading: true };

  const icons = {
    refresh: '<path d="M19 8V4l-3 3a8 8 0 10.5 10.5M19 4v5h-5"/>',
    signout: '<path d="M10 5H5v14h5m3-4l4-3-4-3m4 3H9"/>',
    arrow: '<path d="M19 12H5m5-5l-5 5 5 5"/>',
    search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5L21 21"/>',
    users: '<circle cx="9" cy="8" r="3"/><circle cx="17" cy="9" r="2.5"/><path d="M3.5 20a5.5 5.5 0 0111 0m0-5a5 5 0 016 5"/>',
  };

  function icon(name) {
    return `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${icons[name]}</svg>`;
  }

  async function request(path, options = {}) {
    const response = await fetch(`${config.apiBaseUrl}${path}`, {
      method: options.method || "GET",
      credentials: "include",
      headers: Object.assign(
        { Accept: "application/json" },
        options.csrf ? { "Content-Type": "application/json", "X-CourtVision-CSRF": state.session?.csrfToken || "" } : {},
      ),
      body: options.body ? JSON.stringify(options.body) : undefined,
      cache: "no-store",
    });
    const payload = response.status === 204 ? null : await response.json().catch(() => null);
    if (!response.ok) {
      const error = new Error(payload?.error?.message || "CourtVision could not load the private dashboard.");
      error.status = response.status;
      throw error;
    }
    return payload;
  }

  async function boot() {
    try {
      if (localDemo) {
        state.session = { email: "owner@example.com", isAdmin: true, csrfToken: "local-demo" };
        state.overview = demoOverview();
      } else {
        state.session = await request("/auth/session");
        if (!state.session?.isAdmin) throw Object.assign(new Error("This page is restricted to the CourtVision administrator."), { status: 403 });
        state.overview = await request("/admin/overview");
      }
    } catch (error) {
      state.error = error;
    } finally {
      state.loading = false;
      render();
    }
  }

  function render() {
    root.removeAttribute("aria-busy");
    if (state.error) {
      const signedOut = state.error.status === 401;
      root.innerHTML = `
        <main class="admin-gate">
          <a class="brand" href="./" aria-label="CourtVision home">CourtVision</a>
          <section class="admin-gate-sheet" aria-labelledby="gate-title">
            <span class="admin-gate-mark">${icon("users")}</span>
            <h1 id="gate-title">${signedOut ? "Sign in first." : "Private access only."}</h1>
            <p>${escapeHtml(state.error.message)}</p>
            <a class="button button-primary" href="app.html">${icon("arrow")}<span>${signedOut ? "Go to sign in" : "Return to analysis"}</span></a>
          </section>
        </main>`;
      return;
    }

    const totals = state.overview?.totals || {};
    root.innerHTML = `
      <div class="app-shell admin-shell">
        <header class="topbar admin-topbar">
          <div class="brand-lockup"><a class="brand" href="./">CourtVision</a><span class="brand-divider" aria-hidden="true"></span><span class="status-chip">Private admin</span></div>
          <div class="topbar-center"><span>Signed in as <strong>${escapeHtml(state.session.email)}</strong></span></div>
          <nav class="topbar-actions" aria-label="Admin actions">
            <a class="button button-secondary" href="app.html">${icon("arrow")}<span>Analysis desk</span></a>
            <button class="button button-quiet" id="admin-sign-out" type="button">${icon("signout")}<span>Sign out</span></button>
          </nav>
        </header>
        <main class="admin-view" aria-labelledby="admin-title">
          <header class="admin-heading">
            <div><h1 id="admin-title">Account activity.</h1><p>Private signup and analysis usage across CourtVision. The usage ledger remains available after uploaded clips and results expire.</p></div>
            <p class="admin-updated">Updated <time datetime="${escapeHtml(state.overview.generatedAt)}">${formatDateTime(state.overview.generatedAt)}</time></p>
          </header>
          <dl class="admin-totals" aria-label="CourtVision activity totals">
            ${metric("Signups", totals.signups, "Cognito accounts")}
            ${metric("Confirmed", totals.confirmedUsers, "Email-confirmed accounts")}
            ${metric("Analysis users", totals.analysisUsers, "Unique account emails")}
            ${metric("Analyses", totals.analyses, "Jobs submitted")}
          </dl>
          <section class="admin-ledger" aria-labelledby="users-title">
            <header class="admin-ledger-header">
              <div><h2 id="users-title">People who signed up</h2><p id="user-count">${plural(state.overview.users.length, "account")}, newest first</p></div>
              <div class="admin-ledger-actions">
                <label class="admin-search">${icon("search")}<span class="sr-only">Search by email</span><input id="admin-search" type="search" placeholder="Search email" value="${escapeHtml(state.query)}" autocomplete="off" /></label>
                <button class="button button-paper" id="admin-refresh" type="button">${icon("refresh")}<span>Refresh</span></button>
              </div>
            </header>
            <div id="admin-table-region">${tableMarkup()}</div>
          </section>
        </main>
      </div>`;

    root.querySelector("#admin-search")?.addEventListener("input", (event) => {
      state.query = event.target.value;
      root.querySelector("#admin-table-region").innerHTML = tableMarkup();
    });
    root.querySelector("#admin-refresh")?.addEventListener("click", refresh);
    root.querySelector("#admin-sign-out")?.addEventListener("click", signOut);
  }

  function metric(label, value, note) {
    return `<div><dt>${label}</dt><dd>${Number(value || 0).toLocaleString()}</dd><span>${note}</span></div>`;
  }

  function tableMarkup() {
    const query = state.query.trim().toLowerCase();
    const users = (state.overview?.users || []).filter((user) => !query || user.email.toLowerCase().includes(query));
    if (!users.length) return `<div class="admin-empty"><h3>${query ? "No matching email." : "No signups yet."}</h3><p>${query ? "Try a different search." : "Confirmed and pending accounts will appear here after signup."}</p></div>`;
    return `
      <div class="admin-table-scroll">
        <table class="admin-table">
          <thead><tr><th scope="col">Email</th><th scope="col">Account</th><th scope="col">Signed up</th><th scope="col">Analyses</th><th scope="col">Last analysis</th></tr></thead>
          <tbody>${users.map(userRow).join("")}</tbody>
        </table>
      </div>`;
  }

  function userRow(user) {
    const confirmed = user.status === "CONFIRMED" && user.emailVerified;
    const stateLabel = !user.enabled ? "Disabled" : confirmed ? "Confirmed" : readableStatus(user.status);
    const stateClass = !user.enabled ? "disabled" : confirmed ? "confirmed" : "pending";
    return `<tr>
      <td data-label="Email"><strong>${escapeHtml(user.email || "Email unavailable")}</strong></td>
      <td data-label="Account"><span class="account-state account-state-${stateClass}"><i aria-hidden="true"></i>${escapeHtml(stateLabel)}</span></td>
      <td data-label="Signed up"><time datetime="${escapeHtml(user.signedUpAt || "")}">${formatDate(user.signedUpAt)}</time></td>
      <td data-label="Analyses"><strong class="analysis-count">${Number(user.analysisCount || 0).toLocaleString()}</strong></td>
      <td data-label="Last analysis">${user.lastAnalysisAt ? `<time datetime="${escapeHtml(user.lastAnalysisAt)}">${formatDateTime(user.lastAnalysisAt)}</time>` : '<span class="admin-none">No analysis yet</span>'}</td>
    </tr>`;
  }

  async function refresh() {
    const button = root.querySelector("#admin-refresh");
    if (button) { button.disabled = true; button.setAttribute("aria-busy", "true"); }
    try {
      state.overview = localDemo ? demoOverview() : await request("/admin/overview");
      render();
      requestAnimationFrame(() => root.querySelector("#admin-refresh")?.focus());
    } catch (error) {
      state.error = error;
      render();
    }
  }

  async function signOut() {
    try { if (!localDemo) await request("/auth/sign-out", { method: "POST", csrf: true, body: {} }); } catch (_error) { /* Clear the browser view either way. */ }
    window.location.assign("app.html");
  }

  function demoOverview() {
    const now = new Date();
    const users = [
      ["coach@northside.example", "CONFIRMED", true, 4, 2, 1],
      ["video@cityhoops.example", "UNCONFIRMED", false, 2, 0, null],
      ["analyst@falcons.example", "CONFIRMED", true, 9, 5, 3],
      ["owner@example.com", "CONFIRMED", true, 14, 7, 8],
    ].map(([email, status, verified, signupDays, analysisCount, lastDays]) => ({
      email, status, enabled: true, emailVerified: verified,
      signedUpAt: new Date(now.getTime() - signupDays * 864e5).toISOString(),
      analysisCount,
      lastAnalysisAt: lastDays === null ? null : new Date(now.getTime() - lastDays * 864e5).toISOString(),
    })).sort((a, b) => b.signedUpAt.localeCompare(a.signedUpAt));
    return { generatedAt: now.toISOString(), totals: { signups: 4, confirmedUsers: 3, analysisUsers: 3, analyses: 14 }, users };
  }

  function readableStatus(value) {
    return String(value || "Pending").toLowerCase().replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function formatDate(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "Unavailable" : date.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" });
  }

  function formatDateTime(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "Unavailable" : date.toLocaleString([], { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" });
  }

  function plural(value, noun) { return `${Number(value).toLocaleString()} ${noun}${Number(value) === 1 ? "" : "s"}`; }
  function escapeHtml(value) { return String(value ?? "").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]); }

  boot();
})();

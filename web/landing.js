(() => {
  "use strict";

  const configuredAppUrl = String(window.COURTVISION_CONFIG?.appUrl || "").trim();
  const appUrl = configuredAppUrl || "app.html";

  document.querySelectorAll("[data-analyze-link]").forEach((link) => {
    link.href = appUrl;
  });

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const demoBrowser = document.querySelector(".demo-browser");
  const landingRuler = document.querySelector(".landing-ruler");

  requestAnimationFrame(() => document.body.classList.add("motion-ready"));

  document.querySelectorAll('a[href="#interface-demo"]').forEach((link) => {
    link.addEventListener("click", () => {
      if (!demoBrowser || reducedMotion) return;
      demoBrowser.classList.remove("is-cued");
      requestAnimationFrame(() => demoBrowser.classList.add("is-cued"));
    });
  });

  if (landingRuler && !reducedMotion && "IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        landingRuler.classList.add("is-live");
        observer.disconnect();
      },
      { threshold: 0.45 },
    );
    observer.observe(landingRuler);
  }

  const year = document.querySelector("#current-year");
  if (year) year.textContent = String(new Date().getFullYear());
})();

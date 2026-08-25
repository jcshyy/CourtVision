# CourtVision web client

This is a dependency-free static client for the public CourtVision site, the
permanent sample analysis, and the bounded upload/review application. The public
site is designed to run on Cloudflare Pages while API Gateway and Lambda provide
the AWS control plane.

The checked-in `config.js` enables the public capacity preview. Visitors can
select a clip and validate its format, size, and duration locally, but the client
stops before upload while `analysisAvailable` is `false`. The working sample is
API-independent and remains fully interactive.

## Local visual review

```powershell
python -m http.server 8765 -d web
```

Use `http://127.0.0.1:8765/` for the public landing page,
`http://127.0.0.1:8765/demo.html` for the preprocessed `video_2` sample
analysis, and `http://127.0.0.1:8765/app.html` for the real authenticated
application. The sample video and analysis manifest are generated together so
the replay, Event Rundown, and tactical view remain synchronized. The sample
is API-independent and remains available after deployment.

For local state inspection, `app.html?demo=` supports `signin`, `upload`,
`processing`, `colors`, `error`, and `review`. Those local-only states continue
to use synthetic fixtures for interface development.

Runtime limits and public availability live in `config.js` and are also enforced
by the API and worker when live analysis is enabled. The AWS stack supplies the
authoritative API values after deployment.

## Cloudflare Pages

Deploy the `web` directory as the static output directory. No build command is
required. Attach `courtvision.video` as the production custom domain and keep the
Porkbun registration; only authoritative DNS moves to the Cloudflare nameservers.

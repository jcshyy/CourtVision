# CourtVision web client

This is a dependency-free static client for the private authenticated beta. In
AWS, CloudFront serves these files and routes `/api/*` to the Lambda API. Uploads
go directly to the private artifact bucket through short-lived presigned forms.

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

Runtime limits live in `config.js` and are also enforced by the API and worker.
The AWS template supplies the authoritative deployment values.

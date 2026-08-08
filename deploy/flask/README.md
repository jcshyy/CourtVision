# CourtVision Flask control plane

This adapter exposes the same `/api/*` contract as the existing Lambda handler.
It remains a control plane: uploads go directly to S3, job state lives in
DynamoDB, and inference runs in AWS Batch rather than inside an HTTP request.

## Local verification

Install the API-only dependencies and start Flask:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\api-requirements.txt
$env:SESSION_SECRET = "replace-with-a-long-local-development-secret"
$env:ENVIRONMENT = "development"
.\.venv\Scripts\python.exe -m backend.app.flask_api
```

The unauthenticated load-balancer health endpoint is available at
`http://127.0.0.1:8080/health`. Authenticated API routes remain under `/api`.

## Container

Build the lightweight control-plane image separately from the GPU worker:

```powershell
docker build -f Dockerfile.api -t courtvision-api:latest .
docker run --rm -p 8080:8080 `
  -e SESSION_SECRET="replace-with-a-long-local-development-secret" `
  -e ENVIRONMENT=development `
  courtvision-api:latest
```

Production uses the same environment variables currently supplied to the
Lambda function, including the DynamoDB table names, artifact bucket, SES
sender, Batch queue and job definition, and Secrets Manager secret ARN.

## Migration boundary

Do not route production traffic here until the Flask and Lambda contract tests
pass and a private Flask service can reach S3, DynamoDB, SES, Secrets Manager,
and AWS Batch through a least-privilege task or instance role. CloudFront can
then move `/api/*` from API Gateway to the Flask load balancer without changing
the browser client.

# CourtVision private beta on AWS

This stack deploys the authenticated web control plane. The GPU worker remains
an AWS Batch job definition and queue supplied as parameters because model
weights and GPU capacity are deployment-specific assets.

## Architecture

- CloudFront serves the public landing page, permanent synthetic interface demo,
  and authenticated beta app from a private S3 bucket. It forwards `/api/*` to
  an HTTP API Lambda on the same origin.
- An allowlisted email receives a six-digit SES code. The API issues an
  HttpOnly, Secure, SameSite=Strict session cookie after verification.
- The browser uploads directly to a private artifact bucket through a bounded
  presigned POST policy.
- Lambda records jobs in DynamoDB and submits the existing worker container to
  an AWS Batch GPU queue.
- The Batch entrypoint runs the bounded pipeline, uploads the annotated video
  and analysis manifest, and updates the job state.
- S3 objects and job records expire after 24 hours by default. Structured
  failure reports default to 90 days and never retain the source video.

## Prerequisites

1. AWS SAM CLI and AWS CLI authenticated to the target account.
2. A verified SES sender in the deployment region. Move SES out of sandbox or
   verify every invited recipient during the earliest beta.
3. An ECR image built from this repository with the required model weights
   supplied at runtime.
4. An AWS Batch GPU compute environment, job queue, and job definition whose
   command overrides the image default with `python -m backend.app.batch_job`.
5. Budget alarms and a bounded maximum vCPU policy for the Batch environment.

The Batch job role needs `s3:GetObject`, `s3:PutObject`, and
`dynamodb:UpdateItem` for the stack's artifact bucket and jobs table. Keep model
access restricted to its own S3 prefix or read-only mounted volume.

## Deploy

From the repository root:

```powershell
sam build --template-file deploy/aws/template.yaml
sam deploy --guided `
  --parameter-overrides `
    SesFromEmail=beta@example.com `
    BatchJobQueue=arn:aws:batch:REGION:ACCOUNT:job-queue/courtvision `
    BatchJobDefinition=arn:aws:batch:REGION:ACCOUNT:job-definition/courtvision:1
```

Read `WebBucketName` from the stack outputs, upload the static app, and
invalidate CloudFront:

```powershell
aws s3 sync web "s3://WEB_BUCKET" --delete
aws cloudformation describe-stacks --stack-name STACK_NAME
aws cloudfront create-invalidation --distribution-id DISTRIBUTION_ID --paths "/*"
```

After the same-origin authentication API has passed an end-to-end sign-in
check, set `authConnected: true` in `web/config.js` before the static upload.
The public landing page keeps its beta sign-in links hidden unless that explicit
flag is true on a non-local HTTPS origin.

Add a beta user to the `BetaUsersTableName` output:

```powershell
aws dynamodb put-item `
  --table-name TABLE_NAME `
  --item '{"email":{"S":"analyst@example.com"},"enabled":{"BOOL":true}}'
```

## Configurable limits

`MaxUploadBytes`, `MaxDurationSeconds`, `TargetFps`, `MaxWidth`,
`ResultRetentionSeconds`, `ArtifactRetentionDays`, and `ReportRetentionSeconds`
are stack parameters. Changing them does not require a UI redesign. Keep the
whole-day S3 lifecycle backstop aligned with the result-retention policy.

## Production checks

- Configure a custom domain and ACM certificate before inviting users.
- Replace the wildcard artifact-bucket CORS origin with the final application
  origin after the first deployment.
- Add beta users explicitly; there is no self-service signup path.
- Verify an end-to-end job with real mounted models before opening access.
- Confirm CloudWatch alarms for Lambda errors, Batch failures, queue age, and
  unexpected GPU runtime.
- Recheck the AWS Pricing Calculator and budget alarms before raising capacity.

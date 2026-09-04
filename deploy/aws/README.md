# CourtVision public analysis on AWS

This stack deploys the authenticated web control plane and its GPU compute
plane. The cost-optimized default runs the shared API request handler in AWS
Lambda behind API Gateway. A Flask adapter on ECS Fargate remains selectable
with `ApiRuntime=Flask`; both adapters expose the same tested API contract.

## Architecture

- The production hostname is `courtvision.video`. The browser and API stay on
  the same origin, with API requests routed under `https://courtvision.video/api/*`.
- CloudFront serves the public landing page, permanent synthetic interface demo,
  and authenticated analysis app from a private S3 bucket. It forwards `/api/*` to
  API Gateway and Lambda by default. Lambda invokes the same framework-neutral
  request handler used by the Flask adapter, without an always-running server.
- Users create an email-and-password account and confirm their address with a
  Cognito-delivered code. The API issues an
  HttpOnly, Secure, SameSite=Strict session cookie after verification.
- The browser uploads directly to a private artifact bucket through a bounded
  presigned POST policy.
- The API records jobs in DynamoDB and submits the worker container to an AWS
  Batch queue backed by managed GPU EC2 capacity.
- The Batch entrypoint runs the bounded pipeline, uploads the annotated video
  and analysis manifest, and updates the job state.
- Worker capacity has `MinvCpus=0`, so EC2 scales down when the queue is idle.
  Each new worker downloads the four private detector weights from the retained,
  versioned model bucket before inference.
- S3 objects and job records expire after 24 hours by default. Structured
  failure reports default to 90 days and never retain the source video.

## Prerequisites

1. AWS SAM CLI and AWS CLI authenticated to the target account.
2. Amazon Cognito's built-in email delivery is used for account confirmation
   and password recovery. Its default account quota is intended for low-volume
   low-volume public use.
3. One ECR worker image built from `Dockerfile`. The Lambda API is packaged by
   SAM and does not need an API container image.
4. Worker subnets with outbound access to ECR, S3, DynamoDB, and CloudWatch
   Logs. For this staging layout, public worker subnets must assign public IPv4
   addresses; production can move workers to private subnets with NAT or VPC
   endpoints.
5. An EC2 On-Demand vCPU quota and GPU capacity for the selected instance type.
   The default is one `g4dn.xlarge` worth of capacity (`BatchMaxVcpus=4`).
6. Budget alarms before raising `BatchMaxVcpus`.

The template creates the task/job roles and scopes model reads to `models/*`,
artifact reads/writes to `jobs/*`, and state updates to the jobs table.

## Build and push the worker image

Create the worker ECR repository, authenticate Docker, then build and push an
immutable tag. Replace the account and region placeholders:

```powershell
docker build -f Dockerfile -t ACCOUNT.dkr.ecr.REGION.amazonaws.com/courtvision-worker:COMMIT .
docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/courtvision-worker:COMMIT
```

Only an opt-in `ApiRuntime=Flask` deployment also needs an API image:

```powershell
docker build -f Dockerfile.api -t ACCOUNT.dkr.ecr.REGION.amazonaws.com/courtvision-api:COMMIT .
docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/courtvision-api:COMMIT
```

The worker image intentionally excludes `.pt` files. The stack creates a
private `ModelBucketName`; upload these exact keys after the first deploy:

```text
models/ebard_yolov8n.pt
models/yolo11n-pose.pt
models/wasb_basketball_torchscript.pt
models/court_keypoint_detector.pt
```

## Deploy

While CloudFront account verification is pending, keep the official static site
on GitHub Pages and deploy only the AWS control plane by passing
`EnableCloudFront=false`. This creates the Lambda/API Gateway endpoint, private
artifact and model storage, job tables, and the scale-to-zero Batch queue without
creating a second web host or CloudFront distribution.

From the repository root:

```powershell
sam build `
  --template-file deploy/aws/template.yaml `
  --build-dir "$env:TEMP\courtvision-sam-build"
sam deploy --guided `
  --template-file "$env:TEMP\courtvision-sam-build\template.yaml" `
  --parameter-overrides `
    ApiRuntime=Lambda `
    EnableCloudFront=false `
    WorkerImageUri=ACCOUNT.dkr.ecr.REGION.amazonaws.com/courtvision-worker:COMMIT `
    AllowedWebOrigin=https://courtvision.video `
    BallDetectorBackend=hybrid `
    VpcId=vpc-0123456789abcdef0 `
    WorkerSubnetIds=subnet-aaa,subnet-bbb
```

The checked-in `web/config.js` enables public account creation and authenticated
analysis against `https://api.courtvision.video/api`. Keep the API, Cognito,
private job bucket, and Batch worker healthy before publishing the static client.
If CourtVision moves fully behind CloudFront, update the stack with
`EnableCloudFront=true` plus the custom domain and certificate parameters.

To deploy the optional always-on Flask runtime, also pass `ApiImageUri`, at
least two `PublicSubnetIds`, and the regional `CloudFrontOriginPrefixListId`.
The template's Flask-only defaults are non-deployable sentinels because Lambda
does not consume them; they must be replaced for Flask. Find the managed
prefix-list ID with:

```powershell
aws ec2 describe-managed-prefix-lists `
  --filters Name=prefix-list-name,Values=com.amazonaws.global.cloudfront.origin-facing `
  --query "PrefixLists[0].PrefixListId" `
  --output text
```

The initial stack can become healthy before the model upload because `/health`
does not run inference. Before submitting a browser job, upload the local
weights to the `ModelBucketName` output:

```powershell
aws s3 cp backend/models/ebard_yolov8n.pt "s3://MODEL_BUCKET/models/ebard_yolov8n.pt"
aws s3 cp backend/models/yolo11n-pose.pt "s3://MODEL_BUCKET/models/yolo11n-pose.pt"
aws s3 cp backend/models/wasb_basketball_torchscript.pt "s3://MODEL_BUCKET/models/wasb_basketball_torchscript.pt"
aws s3 cp backend/models/court_keypoint_detector.pt "s3://MODEL_BUCKET/models/court_keypoint_detector.pt"
```

Read `WebBucketName` from the stack outputs, upload the static app, and
invalidate CloudFront:

```powershell
aws s3 sync web "s3://WEB_BUCKET" --delete
aws cloudformation describe-stacks --stack-name STACK_NAME
aws cloudfront create-invalidation --distribution-id DISTRIBUTION_ID --paths "/*"
```

The checked-in configuration sets `authConnected: true`, `publicPreview: false`,
and `analysisAvailable: true`. This exposes Cognito account flows and sends an
authenticated job request to the API, which returns a presigned policy for the
browser's direct upload into private S3 job storage. Set the client back to the
fail-closed preview state before a static upload if the API or worker is taken
out of service intentionally.

## Configurable limits

`MaxUploadBytes`, `MaxDurationSeconds`, `TargetFps`, `MaxWidth`,
`BallDetectorBackend`,
`ResultRetentionSeconds`, `ArtifactRetentionDays`, and `ReportRetentionSeconds`
are stack parameters. Changing them does not require a UI redesign. Keep the
whole-day S3 lifecycle backstop aligned with the result-retention policy.

## Production checks

- `courtvision.video` is registered at Porkbun. Use Porkbun DNS directly for
  ACM validation and the final CloudFront alias; no nameserver transfer or
  second CDN proxy is required.
- Request and validate an ACM certificate for `courtvision.video` in
  `us-east-1`, add the hostname and certificate to CloudFront, and redirect
  `www.courtvision.video` to the apex before inviting users.
- Keep the Lambda/API Gateway origin on HTTPS. If opting into Flask, add HTTPS
  from CloudFront to the ALB before treating it as a production boundary.
- Keep artifact-bucket CORS restricted to `https://courtvision.video`.
- Monitor Cognito confirmation delivery and account-creation abuse. Move to a
  dedicated transactional sender before the built-in daily quota becomes a
  product constraint.
- Verify an end-to-end job with real mounted models before opening access.
- Create CloudWatch alarms for Lambda errors/throttles, API Gateway 5xx
  responses, Batch failures, queue age, and unexpected GPU runtime. For an
  opt-in Flask deployment, monitor ECS task health and ALB 5xx responses too.
  The template creates log groups but does not currently provision these alarms.
- Recheck the AWS Pricing Calculator and budget alarms before raising capacity.

## Cost posture

The Lambda-first stack has no always-running API compute, load balancer, or API
public IPv4 addresses. At light preview traffic, non-GPU services should remain in
the low single-digit dollars per month before credits. The Batch environment has
`MinvCpus=0`; the `g4dn.xlarge` worker is charged only while EC2 capacity is
running. Selecting `ApiRuntime=Flask` adds the continuous Fargate, ALB, and
public IPv4 baseline and should be reserved for sustained traffic that justifies
it.

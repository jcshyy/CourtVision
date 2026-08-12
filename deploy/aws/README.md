# CourtVision private beta on AWS

This stack deploys the authenticated web control plane and its GPU compute
plane. Flask on ECS Fargate is the default API runtime. The previous Lambda
adapter remains selectable with `ApiRuntime=Lambda` as a rollback path, and
both adapters use the same tested request handler.

## Architecture

- CloudFront serves the public landing page, permanent synthetic interface demo,
  and authenticated beta app from a private S3 bucket. It forwards `/api/*` to
  a Flask container behind an Application Load Balancer. The ALB only accepts
  traffic from the AWS-managed CloudFront origin-facing prefix list.
- An allowlisted email receives a six-digit SES code. The API issues an
  HttpOnly, Secure, SameSite=Strict session cookie after verification.
- The browser uploads directly to a private artifact bucket through a bounded
  presigned POST policy.
- Flask records jobs in DynamoDB and submits the worker container to an AWS
  Batch queue backed by managed GPU EC2 capacity.
- The Batch entrypoint runs the bounded pipeline, uploads the annotated video
  and analysis manifest, and updates the job state.
- Worker capacity has `MinvCpus=0`, so EC2 scales down when the queue is idle.
  Each new worker downloads the five private detector weights from the retained,
  versioned model bucket before inference.
- S3 objects and job records expire after 24 hours by default. Structured
  failure reports default to 90 days and never retain the source video.

## Prerequisites

1. AWS SAM CLI and AWS CLI authenticated to the target account.
2. A verified SES sender in the deployment region. Move SES out of sandbox or
   verify every invited recipient during the earliest beta.
3. Two ECR images: `Dockerfile.api` for Flask and `Dockerfile` for inference.
4. A VPC with at least two public subnets for the ALB/Fargate service. The
   worker subnets must have outbound access to ECR, S3, DynamoDB, and CloudWatch
   Logs. For this staging layout, public worker subnets must assign public IPv4
   addresses; production can move both services to private subnets with NAT or
   VPC endpoints.
5. An EC2 On-Demand vCPU quota and GPU capacity for the selected instance type.
   The default is one `g4dn.xlarge` worth of capacity (`BatchMaxVcpus=4`).
6. Budget alarms before raising `BatchMaxVcpus`.

The template creates the task/job roles and scopes model reads to `models/*`,
artifact reads/writes to `jobs/*`, and state updates to the jobs table.

## Build and push the images

Create two ECR repositories, authenticate Docker, then build and push immutable
tags. Replace the account and region placeholders:

```powershell
docker build -f Dockerfile.api -t ACCOUNT.dkr.ecr.REGION.amazonaws.com/courtvision-api:COMMIT .
docker build -f Dockerfile -t ACCOUNT.dkr.ecr.REGION.amazonaws.com/courtvision-worker:COMMIT .
docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/courtvision-api:COMMIT
docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/courtvision-worker:COMMIT
```

The worker image intentionally excludes `.pt` files. The stack creates a
private `ModelBucketName`; upload these exact keys after the first deploy:

```text
models/player_detector.pt
models/yolo11n-pose.pt
models/ball_detector_model.pt
models/wasb_basketball_torchscript.pt
models/court_keypoint_detector.pt
```

## Deploy

From the repository root:

```powershell
sam build `
  --template-file deploy/aws/template.yaml `
  --build-dir "$env:TEMP\courtvision-sam-build"
sam deploy --guided `
  --template-file "$env:TEMP\courtvision-sam-build\template.yaml" `
  --parameter-overrides `
    SesFromEmail=beta@example.com `
    ApiRuntime=Flask `
    ApiImageUri=ACCOUNT.dkr.ecr.REGION.amazonaws.com/courtvision-api:COMMIT `
    WorkerImageUri=ACCOUNT.dkr.ecr.REGION.amazonaws.com/courtvision-worker:COMMIT `
    BallDetectorBackend=hybrid `
    VpcId=vpc-0123456789abcdef0 `
    PublicSubnetIds=subnet-aaa,subnet-bbb `
    WorkerSubnetIds=subnet-aaa,subnet-bbb `
    CloudFrontOriginPrefixListId=pl-0123456789abcdef0
```

Find the managed prefix-list ID in the deployment region with:

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
aws s3 cp backend/models/player_detector.pt "s3://MODEL_BUCKET/models/player_detector.pt"
aws s3 cp backend/models/yolo11n-pose.pt "s3://MODEL_BUCKET/models/yolo11n-pose.pt"
aws s3 cp backend/models/ball_detector_model.pt "s3://MODEL_BUCKET/models/ball_detector_model.pt"
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
`BallDetectorBackend`,
`ResultRetentionSeconds`, `ArtifactRetentionDays`, and `ReportRetentionSeconds`
are stack parameters. Changing them does not require a UI redesign. Keep the
whole-day S3 lifecycle backstop aligned with the result-retention policy.

## Production checks

- Configure a custom domain and ACM certificate before inviting users.
- Add HTTPS from CloudFront to the ALB before treating this staging stack as a
  production boundary. Viewer traffic is HTTPS now, while the restricted
  CloudFront-to-ALB hop uses HTTP.
- Replace the wildcard artifact-bucket CORS origin with the final application
  origin after the first deployment.
- Add beta users explicitly; there is no self-service signup path.
- Verify an end-to-end job with real mounted models before opening access.
- Confirm CloudWatch alarms for ECS task health, ALB 5xx responses, Batch
  failures, queue age, and unexpected GPU runtime.
- Recheck the AWS Pricing Calculator and budget alarms before raising capacity.

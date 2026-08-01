# Cost-controlled EC2 worker

This deployment runs CourtVision as a single-job batch worker. It is not a
public upload API. The defaults accept at most 30 seconds per invocation,
analyze at 15 FPS and 960-pixel width, cap the container at 14 GiB/4 vCPUs,
and kill a job after 90 minutes.

## Recommended starting instance

Use `g4dn.xlarge` in `us-east-1` for occasional inference. It has one NVIDIA
T4 GPU, 4 vCPUs, and 16 GiB RAM. Start with On-Demand while measuring real job
duration. Spot can be cheaper later, but an interruption can discard an active
job unless it is retried.

Do not leave this instance running as a web server. A continuously running GPU
instance is expensive; this worker is intended to start for queued work and
stop afterward.

## One-time instance setup

1. Create an AWS Budget with email alerts before launching compute. A $10
   monthly test budget with alerts at $5 and $8 is a reasonable starting guard.
2. Launch a `g4dn.xlarge` using a current AWS Deep Learning Base GPU AMI. Use a
   30 GiB gp3 root disk, encrypted, with delete-on-termination enabled.
3. Set **instance-initiated shutdown behavior** to **Stop**, not Terminate.
4. Do not open inbound application ports. Prefer AWS Systems Manager Session
   Manager; if SSH is necessary, restrict port 22 to your own IP.
5. Attach no broad AWS IAM permissions. An S3-backed workflow should use a role
   restricted to one input/output bucket prefix.
6. Install Git and Docker if the selected AMI does not already provide them,
   clone CourtVision into `/opt/courtvision/source`, then build:

   ```bash
   cd /opt/courtvision/source
   sudo docker build -t courtvision:latest .
   sudo mkdir -p /opt/courtvision/{models,input,output,cache}
   ```

7. Copy the three model files into `/opt/courtvision/models`. They are excluded
   from the Docker image and repository. Copy an input video into
   `/opt/courtvision/input`.

## Verify before processing

The runtime check reports whether PyTorch can see the NVIDIA GPU:

```bash
sudo docker run --rm --gpus all \
  --entrypoint python \
  -v /opt/courtvision/models:/app/backend/models:ro \
  courtvision:latest scripts/check_runtime.py --check-models --require-cuda
```

Do not proceed with the GPU instance if `cuda_available` is false; otherwise
you would pay GPU pricing while running CPU inference.

## Run one bounded job

```bash
cd /opt/courtvision/source
chmod +x deploy/ec2/run-job.sh
sudo -E SHUTDOWN_AFTER_JOB=1 deploy/ec2/run-job.sh clip.mp4
```

The result is written to `/opt/courtvision/output` under a unique UTC job ID.
With `SHUTDOWN_AFTER_JOB=1`, the script stops the instance whether the job
succeeds, fails, or hits the 90-minute timeout. Run once with
`SHUTDOWN_AFTER_JOB=0` while initially diagnosing the environment.

Override limits only deliberately:

```bash
sudo -E MAX_DURATION_SECONDS=30 TARGET_FPS=15 MAX_WIDTH=960 \
  MAX_JOB_MINUTES=90 SHUTDOWN_AFTER_JOB=1 \
  deploy/ec2/run-job.sh clip.mp4
```

## Cost guardrails

- A stopped instance does not incur EC2 compute charges, but its EBS disk still
  incurs storage charges.
- Delete unused output, caches, stopped test instances, EBS volumes, snapshots,
  Elastic IPs, and old container images.
- Tag the worker and its volume with `Project=CourtVision` for cost filtering.
- Use one worker and one concurrent job until runtime and accuracy are measured.
- Do not buy a Reserved Instance or Savings Plan during validation.
- Recheck the AWS Pricing Calculator for the selected region immediately before
  launch; prices and Spot capacity vary.

## Current limitation

CourtVision still materializes the selected frames and rendered frames in RAM.
The 30-second/15-FPS/960-width defaults are a bounded deployment profile, not a
streaming implementation. Longer clips require the planned stateful chunking
work before raising these limits safely.

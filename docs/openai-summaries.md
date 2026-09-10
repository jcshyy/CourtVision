# OpenAI summaries and tactical review

New AWS Batch jobs attach `gameSummary` to the private `analysis.json` artifact
after video processing. The review page displays it beneath Evidence & Unknowns.
The result shares the existing authenticated downloads and retention policy.
Opening a result does not make another OpenAI call. Existing results are unchanged.

CourtVision currently processes bounded clips, not complete games. Reports say
what footage they cover and must not invent a score, winner, shot result or team
strategy. Tactical insights are suggestions for film review, supported by
validated evidence references, not verified coaching conclusions. A model may
still misinterpret observations; review generated prose against the footage.

## Configuration

For AWS, create a Secrets Manager secret containing a plain OpenAI API key or
`{"OPENAI_API_KEY":"..."}`. Keep the value out of source control and browser config.
Deploy the rebuilt worker image and static client with these SAM parameters:

- `OpenAIApiKeySecretArn`: the secret ARN, in the worker's region.
- `OpenAISummaryModel`: defaults to `gpt-4.1-mini`; select a Responses model with
  Structured Outputs support that your OpenAI project can access.

An empty ARN disables generation. The worker receives permission only for that
secret. A customer-managed KMS key additionally needs a scoped decrypt grant.
Worker egress must reach `api.openai.com:443` and AWS Secrets Manager.

For local use, set `OPENAI_API_KEY` in the process environment, optionally set
`OPENAI_SUMMARY_MODEL`, and run:

```powershell
python -m backend.app.game_summary path/to/analysis.json path/to/summary.json
```

This writes a standalone report and exits nonzero when unavailable or disabled.
It does not change the source manifest. The local video demo does not generate
summaries automatically; automatic enrichment runs in the AWS Batch worker.

## Data and failure behavior

The worker uses the [Responses API](https://developers.openai.com/api/docs/guides/text)
with [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
and `store: false`. It sends clip duration, candidate counts, at most 200 timecoded
events, team IDs and aggregate holder/unknown-team observation counts. It sends no
video, file paths, account identifiers or arbitrary event evidence text. OpenAI's
API data policies still apply; `store: false` is not a zero-retention guarantee.

Each new completed processing run normally makes one provider request, capped at
2,200 output tokens and a 45-second network timeout. A generated response that
fails JSON or grounding validation receives one constrained correction attempt;
network and HTTP failures are not retried. Refusals, repeated invalid responses,
and unknown evidence references produce `status: unavailable`; the video job still
completes. Missing credentials produce `status: disabled` with no provider call.
Re-running video analysis can incur another generation charge. No live API call
is needed for the mocked unit tests.

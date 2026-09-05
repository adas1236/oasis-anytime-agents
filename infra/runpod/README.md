# Runpod evaluation workflow

This directory packages the existing evaluator for Runpod Pods. Nothing here
contacts Runpod by default: planning, launch, and termination each have a dry
mode, and API mutations require an explicit `--execute`.
After syncing the updated project, `oasis-runpod` is a shorthand for
`python -m oasis.runpod_experiments`.

## One-time setup

1. After committing and pushing these files, open the repository's **Actions**
   tab, choose **Build Runpod evaluation image**, select **Run workflow**, and
   leave the tag as `runpod-eval-v1`. The manual workflow publishes the exact
   image already named in `experiment.toml`; it does not need any added GitHub
   secrets. After its first successful build, make the GHCR package public.

   If you prefer to build locally instead, use:

   ```bash
   docker login ghcr.io -u adas1236
   docker build -f infra/runpod/Dockerfile \
     -t ghcr.io/adas1236/oasis-anytime-agents:runpod-eval-v1 .
   docker push ghcr.io/adas1236/oasis-anytime-agents:runpod-eval-v1
   ```

   If the GHCR package must remain private, create a Runpod container-registry
   credential and put its ID in `runpod.container_registry_auth_id` instead.

2. Create these three Runpod secrets. The matching references are already in
   `experiment.toml`:

   - `hf_token`: a Hugging Face token whose account has accepted the Gemma
     model terms.
   - `aws_access_key_id`: an AWS access key allowed to read/write your result
     prefix.
   - `aws_secret_access_key`: its AWS secret key.

3. Edit exactly two remaining placeholders in `experiment.toml`:

   ```toml
   AWS_DEFAULT_REGION = "YOUR_BUCKET_REGION"
   s3_uri = "s3://YOUR_BUCKET/oasis-anytime-agents"
   ```

4. Create a Runpod API key and export it only in the local shell used to launch
   and manage Pods:

   ```bash
   export RUNPOD_API_KEY=YOUR_RUNPOD_API_KEY
   ```

   The launch command refuses to proceed while any `REPLACE_...` placeholder
   remains. API keys and secret values are never written to a plan.

The supplied configuration uses S3 for artifact persistence. Other supported
options are:

   - Alternatively, set `runpod.network_volume_id`. Every job writes to a
     unique directory, so sharing results is safe. Preload the Hugging Face and
     OSRM caches there before a large run. A network volume constrains all Pods
     to its data center, which can reduce simultaneous GPU availability. If you
     choose this route, set `artifacts.s3_uri = ""` and remove the three AWS
     variables from `[runpod.env]`.
   - With neither S3 nor a network volume, each Pod gets its own volume disk. Do not terminate
     those Pods until their result directories have been downloaded.

Runpod resolves secret references in Pod environment variables. The planner
also rejects literal values for environment names that look sensitive, so keys
cannot accidentally be serialized into the JSON plan.

## Smoke test, timing benchmark, and full plan

The planner reads the datasets locally, performs the seeded shuffle once, and
stores both the ordered record IDs and their SHA-256 digest. Every budget job
receives its shard's digest. The evaluator reconstructs the slice inside the
container and fails before model loading if it differs.

Create a cheap three-job infrastructure smoke plan (two rows, only the
unlimited/unlimited condition):

```bash
uv run --no-sync python -m oasis.runpod_experiments plan \
  --config infra/runpod/experiment.toml \
  --output evaluation-output/runpod/smoke-plan.json \
  --rows 2 --time-budgets unlimited --token-budgets unlimited
```

Create a 20-row, 27-condition timing plan for one exact GPU type:

```bash
uv run --no-sync python -m oasis.runpod_experiments plan \
  --config infra/runpod/experiment.toml \
  --output evaluation-output/runpod/benchmark-5090.json \
  --rows 20 --gpu-type "NVIDIA GeForce RTX 5090"
```

Repeat with a different `--gpu-type` to compare hardware. Use `--gpu-count 2`
or another count for model-parallel testing; the evaluator sees all allocated
devices and Transformers/Accelerate chooses placement automatically. Extra GPUs
do not create data-parallel replicas. For throughput, `--shards N` creates N
independent Pods per dataset/budget condition and splits the same selected rows
without overlap.

For a cheaper raw-speed comparison before running all 27 conditions, create
three-job, unlimited-budget plans using the same 20 rows and seed:

```bash
uv run --no-sync python -m oasis.runpod_experiments plan \
  --config infra/runpod/experiment.toml \
  --output evaluation-output/runpod/speed-a5000.json \
  --rows 20 --time-budgets unlimited --token-budgets unlimited \
  --gpu-type "NVIDIA RTX A5000"

uv run --no-sync python -m oasis.runpod_experiments plan \
  --config infra/runpod/experiment.toml \
  --output evaluation-output/runpod/speed-4090.json \
  --rows 20 --time-budgets unlimited --token-budgets unlimited \
  --gpu-type "NVIDIA GeForce RTX 4090"

uv run --no-sync python -m oasis.runpod_experiments plan \
  --config infra/runpod/experiment.toml \
  --output evaluation-output/runpod/speed-5090.json \
  --rows 20 --time-budgets unlimited --token-budgets unlimited \
  --gpu-type "NVIDIA GeForce RTX 5090"
```

These plans select identical rows. Compare `job-status.json` for total Pod-job
time and `results.summary.json` for evaluator time and per-row token/timing
statistics. Then use the winning GPU in the 27-condition timing plan.

Create the full 500-row plan using the TOML defaults:

```bash
uv run --no-sync python -m oasis.runpod_experiments plan \
  --config infra/runpod/experiment.toml \
  --output evaluation-output/runpod/full-plan.json
```

## Launch and lifecycle

Preview the exact API payloads without contacting Runpod:

```bash
uv run --no-sync python -m oasis.runpod_experiments launch \
  --plan evaluation-output/runpod/smoke-plan.json --print-payloads
```

After inspection, export the API key locally and explicitly launch:

```bash
export RUNPOD_API_KEY=...
uv run --no-sync python -m oasis.runpod_experiments launch \
  --plan evaluation-output/runpod/smoke-plan.json --execute
```

The neighboring `*.state.json` is atomically updated after each Pod creation.
Rerunning the same launch command skips already recorded jobs, preventing
duplicates after a local interruption. `--job-id ID` selects individual jobs;
`--max-jobs N` limits a launch while testing.

Inspect lifecycle state and preview deletion:

```bash
uv run --no-sync python -m oasis.runpod_experiments status \
  --state evaluation-output/runpod/smoke-plan.state.json
uv run --no-sync python -m oasis.runpod_experiments terminate \
  --state evaluation-output/runpod/smoke-plan.state.json
```

Only add `--execute` to `terminate` after checking every `job-status.json` and
result summary in object storage or on a persistent volume. Pod deletion is
permanent for local volume data.

With S3 enabled, artifacts are stored below
`PREFIX/PLAN_ID/JOB_ID/`. Downloading that prefix produces the same directory
layout as `local_root`; for example:

```bash
aws s3 sync s3://BUCKET/PREFIX/PLAN_ID evaluation-output/runpod/results/PLAN_ID
```

Inside a Pod, one process runs one dataset/time/token/shard condition. The
existing evaluator fsyncs every completed row and atomically rewrites its
summary. The wrapper uploads all artifacts at the configured interval and on
exit, restores prior S3 checkpoints on retry, verifies the visible GPU count,
records the actual GPU/VRAM/compute capability and Torch/CUDA versions, and
forwards termination signals to the evaluator.

Official references: [Pod REST API](https://docs.runpod.io/api-reference/pods/POST/pods),
[Pod CLI and GPU identifiers](https://docs.runpod.io/runpodctl/reference/runpodctl-pod),
[Runpod secrets](https://docs.runpod.io/pods/templates/environment-variables), and
[storage behavior](https://docs.runpod.io/pods/storage/types).

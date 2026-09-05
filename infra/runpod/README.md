# Runpod evaluation workflow

This directory packages the existing evaluator for Runpod Pods. Nothing here
contacts Runpod by default: launch, stop, and termination each have a dry mode,
and API mutations require an explicit `--execute`.
After syncing the updated project, `oasis-runpod` is a shorthand for
`python -m oasis.runpod_experiments`.

## Registry evaluation restart

The current configuration uses `tool_mode = "registry"`: the application's full 21-tool catalog,
with dataset-backed provider adapters instead of the old two-tool solver shortcut. Models see only
the prompt, definitions, and tool results. The evaluator never fills in missing names or parameters.
See the repository README's mock-experiment section for the data boundary and scoring details.

Keep the previous Pods stopped. Their saved results use a different protocol; do not resume their
old plan or mix those rows into the new evaluation. Stopped Pods retain disks and may incur storage
charges until explicitly deleted. The new config uses a separate `oasis-registry-200-row-grid`
name and `runpod-eval-v5` image. No existing launch-state files are changed by generating a new plan.

Before building, run this offline, no-GPU check from the repository root (use a fresh output path
on subsequent runs):

```bash
PYTHONPATH=src .venv/bin/python -m oasis.registry_smoke \
  --output evaluation-output/registry-smoke
```

The Docker build repeats a one-row-per-dataset acceptance check. This verifies tool contracts,
restored artifact storage, source adapters, and scoring, not real-model performance. After the
image is published, first create a short pilot plan without launching anything:

```bash
uv run --no-sync python -m oasis.runpod_experiments plan \
  --config infra/runpod/experiment.toml \
  --output evaluation-output/runpod/registry-pilot.json \
  --rows 3 --time-budgets unlimited --token-budgets unlimited
```

Launch a pilot only when ready to incur GPU charges, using the lifecycle commands below and this
new plan path. Inspect all three datasets before launching the full grid. The config preserves
200 rows, seed 42, one RTX 5090, 27 conditions, and 30s/60s/unlimited wall budgets. It raises the
round cap to 20 and generation cap to 1536. The provisional aggregate token budgets are now
32k/256k/unlimited, because the full catalog is charged on every turn. Recalibrate token and wall
budgets against pilot traces; previous two-tool runtime and cost estimates are not transferable.

## One-time setup

1. After committing and pushing these files, open the repository's **Actions**
   tab, choose **Build Runpod evaluation image**, select **Run workflow**, and
   leave the tag as `runpod-eval-v5`. The manual workflow publishes the exact
   image already named in `experiment.toml`; it does not need any added GitHub
   secrets. The workflow summary records the immutable image digest and manifest.

   The image uses Runpod's host-cached PyTorch 2.8/CUDA 12.8 base. Its custom
   layer reuses that Torch installation, prunes Torch and torchvision from the
   exported application requirements, and keeps uv's download cache outside the
   final image. This avoids the duplicated CUDA stack that prevented the first
   image from finishing its pull in a reasonable time. The build also checks the
   Torch/torchvision versions and the Gemma 4 Transformers auto-model mapping,
   while the image includes the five OSRM matrices needed by the mock TSP data.

   If you prefer to build locally instead, use:

   ```bash
   docker login ghcr.io -u adas1236
   docker build --platform=linux/amd64 -f infra/runpod/Dockerfile \
     -t ghcr.io/adas1236/oasis-anytime-agents:runpod-eval-v5 .
   docker push ghcr.io/adas1236/oasis-anytime-agents:runpod-eval-v5
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

3. Verify the two deployment-specific values already filled into
   `experiment.toml`:

   ```toml
   AWS_DEFAULT_REGION = "us-east-2"
   s3_uri = "s3://oasis-anytime-agent/oasis-anytime-agents"
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
     optional runtime caches there before a large run. A network volume
     constrains all Pods to its data center, which can reduce simultaneous GPU
     availability. If you choose this route, set `artifacts.s3_uri = ""` and
     remove the three AWS variables from `[runpod.env]`.
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

First create a one-Pod infrastructure probe. It initializes Python, Torch/CUDA,
GPU discovery, and S3 upload, but deliberately does not download a model or run
an evaluation row:

```bash
uv run --no-sync python -m oasis.runpod_experiments plan \
  --config infra/runpod/experiment.toml \
  --output evaluation-output/runpod/image-probe.json \
  --rows 1 --time-budgets unlimited --token-budgets unlimited \
  --gpu-type "NVIDIA GeForce RTX 4090" --probe-only

uv run --no-sync python -m oasis.runpod_experiments launch \
  --plan evaluation-output/runpod/image-probe.json --max-jobs 1 --execute
```

The launch state records the Pod creation time and hourly price. The probe's
`job-status.json` records container start time, worker setup time, image tag,
Torch/CUDA versions, and GPU inventory. Together they separate image cold-start
cost from model and evaluator cost. After emitting `oasis_worker_finished`, the
container deliberately waits instead of exiting and being restarted by Runpod.
Stop the Pod immediately after that log appears or the probe status reaches
`complete`:

```bash
uv run --no-sync python -m oasis.runpod_experiments stop \
  --state evaluation-output/runpod/image-probe.state.json --execute
```

After the probe succeeds, create a three-job evaluation smoke plan (two rows,
only the unlimited/unlimited condition), then launch just its first job:

```bash
uv run --no-sync python -m oasis.runpod_experiments plan \
  --config infra/runpod/experiment.toml \
  --output evaluation-output/runpod/smoke-plan.json \
  --rows 2 --time-budgets unlimited --token-budgets unlimited

uv run --no-sync python -m oasis.runpod_experiments launch \
  --plan evaluation-output/runpod/smoke-plan.json --max-jobs 1 --execute
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

Create the full 200-row plan using the TOML defaults. The seeded ordered sample
is shared by all nine budget conditions for each dataset:

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
  --plan evaluation-output/runpod/smoke-plan.json --max-jobs 1 --execute
```

The neighboring `*.state.json` is atomically updated after each Pod creation.
Rerunning the same launch command skips already recorded jobs, preventing
duplicates after a local interruption. `--job-id ID` selects individual jobs;
`--max-jobs N` limits a launch while testing.

Inspect lifecycle state and preview deletion:

```bash
uv run --no-sync python -m oasis.runpod_experiments status \
  --state evaluation-output/runpod/smoke-plan.state.json
uv run --no-sync python -m oasis.runpod_experiments stop \
  --state evaluation-output/runpod/smoke-plan.state.json
uv run --no-sync python -m oasis.runpod_experiments terminate \
  --state evaluation-output/runpod/smoke-plan.state.json
```

Add `--execute` to `stop` as soon as a job completes; this releases compute but
keeps its Pod disk. Only add `--execute` to `terminate` after checking every
`job-status.json` and result summary in object storage or on a persistent
volume. Pod deletion is permanent for local volume data.

With S3 enabled, artifacts are stored below
`PREFIX/PLAN_ID/JOB_ID/`. Downloading that prefix produces the same directory
layout as `local_root`; for example:

```bash
aws s3 sync s3://BUCKET/PREFIX/PLAN_ID evaluation-output/runpod/results/PLAN_ID
```

Each job saves `results.jsonl` and `results.summary.json`, plus
`results.artifacts/RECORD_ID/BUDGET_ID/attempts/ATTEMPT_ID/trace.jsonl` and its sibling `artifacts/`
object store. Unique attempt directories prevent retries from overwriting old traces.
Trace lines include complete model turns, arguments/results, streamed candidates, and independently
scored incumbents—even before a row completes. Uploads recurse into these directories, skip
unchanged files and temporary writes, and occur every 15 seconds by default. This is a best-effort
sync interval, not a strict maximum loss guarantee if storage/network writes stall. Resume restores
completed-row checkpoints; an unfinished cell is re-executed, not resumed inside its model turn.

Once the matching launch state and result tree are local, produce a stage-by-stage
cost report:

```bash
uv run --no-sync python -m oasis.runpod_experiments cost \
  --state evaluation-output/runpod/image-probe.state.json \
  --results-root evaluation-output/runpod/results
```

The report separates image pull/container start, worker setup, workload runtime,
and completion-to-stop delay, then estimates compute cost from the price captured
at Pod creation. Use the same command with the smoke plan and later GPU benchmark
plans.

Inside a Pod, one process runs one dataset/time/token/shard condition. The
existing evaluator fsyncs every completed row and atomically rewrites its
summary. The wrapper uploads all artifacts at the configured interval and on
exit, restores prior S3 checkpoints on retry, verifies the visible GPU count,
records the actual GPU/VRAM/compute capability and Torch/CUDA versions, and
forwards termination signals to the evaluator. It treats a missing or incomplete
result summary as a failure, includes summary metrics or a bounded error tail in
its final log event, and waits for the lifecycle controller after every terminal
outcome so Runpod cannot start the workload a second time.

The launcher uses Runpod REST API v2 and writes every successful lifecycle
mutation to the neighboring state file before continuing.

The image intentionally runs the batch worker directly instead of chaining the
base image's `/start.sh`: these Pods expose no SSH or Jupyter service and persist
their outputs to S3 and the mounted Pod disk. Stop them after the terminal worker
event to release GPU billing; the completion hold is a restart guard, not an
automatic lifecycle controller.

Official references: [Pod REST API](https://docs.runpod.io/api-reference-v2/pods/create-pod),
[Pod CLI and GPU identifiers](https://docs.runpod.io/runpodctl/reference/runpodctl-pod),
[Runpod secrets](https://docs.runpod.io/pods/templates/environment-variables), and
[storage behavior](https://docs.runpod.io/pods/storage/types).

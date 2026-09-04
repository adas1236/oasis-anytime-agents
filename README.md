# OASIS Anytime GeoAI Agent

OASIS is a budget-aware GeoAI agent for public-health and spatial-equity planning. The Phase 12
demonstration release provides typed configuration, model-independent streaming chat, Gemma 4
profiles, immutable content-addressed artifacts, a validated async tool SDK, deterministic and
live/snapshotted evidence planes, and independently evaluated location-allocation optimization.
It also provides directed routing for mobile services and immutable demand, travel, and facility
scenario analysis. A framework-neutral anytime controller now owns immutable problem admission,
hard wall/token/tool budgets, atomic monotone incumbents, deterministic fallback, cancellation,
and append-only traces. A versioned asynchronous FastAPI service exposes model, tool, problem,
runtime, run, artifact, map, and replayable SSE contracts through `/api/v1`. A typed, hardware-
neutral runtime layer supports safe CPU planning, explicitly probed CUDA, memory-oriented
Accelerate dispatch across GPUs on one machine, and an authenticated remote model-worker
protocol. A manifest-driven evaluation harness now generates versioned synthetic instances,
builds independently scored exact or best-known references, runs paired controller comparisons,
and reports quality over wall time and model tokens. Gemma 4 uses its native
tool format; compatible non-Gemma chat models use a conservative tagged-JSON fallback. The fake
backend and frozen fixtures exercise the complete current workflow without model weights, network
access, or a GPU. A no-build, accessible web client now launches and follows runs exclusively
through `/api/v1`, including reconnect, cancellation, live verified maps, equity metrics, and
runtime resolution. A seven-problem Track B showcase now publishes independently scored absolute
health/equity metrics, improvement histories, immutable run hashes, resource profiles, and a
privacy/provenance audit. Real-hardware and real-model evaluation remain opt-in; this workspace
has also completed the documented RTX 5060 Ti/Gemma 4 E2B compatibility smoke.

## Setup

Install [uv](https://docs.astral.sh/uv/), then create the locked Python 3.12 environment:

```bash
uv sync
```

The repository configures a local `.uv-cache` so setup does not depend on a writable user cache.
The default `cpu` dependency group contains the CPU-side dependencies planned for the V1
implementation, including Torch and Accelerate from the official CPU wheel index. The mutually
exclusive `gpu` group uses the official CUDA 12.8 wheel index from the same `uv.lock`; do not use
`--all-groups`. Model weights are never downloaded by setup or by the default test suite.

Run the common quality gate with:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
git diff --check
```

The default tests are offline and CPU-only. Tests that require network, model weights, GPU,
multi-GPU, or long runtimes use explicit pytest markers and opt-in configuration.

## Track B demonstration release

Run or resume the complete CPU/fake showcase after `uv sync`:

```bash
uv run oasis demo --output evaluation-output/track-b-showcase-v1
```

The first run creates 21 independently evaluated controller runs: baseline, fixed deterministic
portfolio, and iterative fake-model portfolio for each of seven problems. A completed output is
warm-resumable; unchanged raw run records are reused. The command writes `release-report.json`,
the versioned manifest copy, per-run JSON, `runs.jsonl`, `summary.json`, and aggregate CSV/Parquet.
`release-report.json` is the concise demonstration artifact; it includes raw domain metrics, group
and scenario metrics, baseline gain, comparator histories, problem/evidence/policy hashes, and
hardware-validation status. It never uses assistant prose or solver-reported objectives as scores.

The package-owned snapshot is `oasis-track-b-synthetic-snapshot-v1`. It is generated from frozen
held-out seeds by generator `1.0.0`, contains no patient/person-level records, and is released as
CC0-1.0 with source URIs, generator versions, transformations, and immutable content hashes on
every artifact. It covers:

| Showcase | Authoritative metrics include |
|---|---|
| Cooling-center weighted coverage | overall and underserved-group coverage |
| Clinic average access | weighted average, maximum, and quantile access |
| Emergency/tail access | 90th-percentile, average, and maximum access |
| Capacity-limited allocation | served/unmet demand, coverage, access, and group allocation |
| Equity-first coverage | worst-group and overall coverage |
| One-facility-failure resilience | normal, one-failure, and redundant coverage |
| Mobile-service route coverage | served demand/value, coverage, and route time by scenario |

Monitor placement is deliberately not presented as a health-impact showcase: the current
`monitors` calibration fixture has no validated exposure, detection, or prediction-variance impact
model. A p-center proxy would not justify that claim.

The tested per-run default is 30 seconds, 4,096 aggregate model tokens, 1,024 generated tokens,
and eight tool calls. These limits admit the conservative 10-second `improve` p95 while retaining
the controller's finalization reserve. The `improve` p50 estimate is 125 ms, updated from the
offline release measurements; the p95 intentionally remains conservative for larger instances.
Each release report compares observed p50/p95 tool latency with declared p95, records per-action
input/generated-token growth, verifies the compact controller context stays below 16 KiB, totals
reachable artifact sizes, and reports deadline overshoot. Measurements are hardware/runtime
stratified and are not universal performance claims.

The reproducible snapshot is the default. Live refresh remains a separate, explicit external-read
workflow so mutable data cannot silently become benchmark truth:

```bash
OASIS_PROVIDER_USER_AGENT="my-oasis-test/0.1 contact@example.org" \
uv run oasis evidence live-smoke --confirm-live-network \
  --place "Cambridge, Massachusetts" \
  --source-url "https://example.org/public-data.geojson" \
  --source-format geojson \
  --source-license "CC-BY-4.0" \
  --source-units people \
  --source-crs EPSG:4326 \
  --route-coordinate=-71.1097,42.3736 \
  --route-coordinate=-71.0589,42.3601
```

The resulting evidence becomes a new immutable snapshot and therefore a new problem/run identity.
Provider failure returns an explicitly stale cached child snapshot when policy permits, otherwise
a typed failure; it never fabricates data. Endpoint licenses, acceptable-use rules, attribution,
freshness, field meanings, and spatial fitness remain the operator's responsibility.

Run the Phase 12 hardening scenarios directly with:

```bash
uv run pytest tests/e2e/test_phase12_release.py
```

The opt-in injectors in `oasis.failure_injection` reproduce provider outage, malformed model
output, unavailable model, model OOM/error, tool timeout, and mid-tool user cancellation. The
focused suite also exercises stale-cache warnings, model-token exhaustion followed by deterministic
search, routing interruption after baseline commitment, and evidence-hash isolation. The full
`uv run pytest` gate additionally covers custom compatible Hugging Face identity, policy-hash
isolation, and fake single-/multi-GPU planning. Every post-baseline failure or interruption retains
a plan that is independently revalidated and rescored.

For a complete local demonstration, launch the versioned API/UI separately after the offline
release gate:

```bash
uv run oasis serve --backend fake --serve-ui --ui-root ui \
  --artifact-root artifacts/ui-demo \
  --run-root artifacts/ui-demo-runs
```

The headless evaluation smoke suite remains available with:

```bash
uv run oasis evaluate src/oasis/evaluation/manifests/smoke.json \
  --output evaluation-output/offline-smoke-v1
uv run oasis summarize evaluation-output/offline-smoke-v1
```

Hardware validation status for this release is explicit: CPU/fake execution and fake one-/many-GPU
planning pass, and a real local RTX 5060 Ti/Gemma 4 E2B BF16/SDPA compatibility smoke passes.
Rented-host execution remains pending. The authenticated remote-worker protocol is covered with
local fake transport, not claimed as a measured provider deployment. The authoritative hardware
scope in `HARDWARE_CHANGES.md` excludes multi-node, SLURM-specific, container, DeepSpeed, vLLM,
and torchrun implementation.

## Deterministic fake chat

Use the fake backend for a complete offline multi-turn smoke test:

```bash
uv run oasis chat --backend fake \
  --prompt "What does this project do?" \
  --prompt "What model would a real run use?"
```

Run `uv run oasis chat --backend fake` without `--prompt` for an interactive session. Enter
`/quit` to stop. The fake backend deterministically echoes each user turn in streamed chunks and
reports aggregate token usage; it does not import Transformers or PyTorch.

## Model profiles and configuration

The default profile is `gemma4_e4b_it`. All named profiles use instruction-tuned Gemma 4 weights:

| Profile | Hugging Face model ID | Context limit |
|---|---|---:|
| `gemma4_e2b_it` | `google/gemma-4-E2B-it` | 128K |
| `gemma4_e4b_it` | `google/gemma-4-E4B-it` | 128K |
| `gemma4_12b_it` | `google/gemma-4-12B-it` | 256K |
| `gemma4_26b_a4b_it` | `google/gemma-4-26B-A4B-it` | 256K |
| `gemma4_31b_it` | `google/gemma-4-31B-it` | 256K |

Configuration precedence is explicit object overrides, CLI flags, `OASIS_*` environment values,
then safe defaults. Common values include `OASIS_MODEL_PROFILE`, `OASIS_MODEL_ID`,
`OASIS_MAX_GENERATED_TOKENS`, `OASIS_THINKING`, and `OASIS_DEVICE`. The device default is always
`cpu`; merely importing or configuring OASIS does not probe CUDA.

CLI flags switch profiles and settings without source changes:

```bash
uv run oasis chat --backend fake --profile gemma4_e2b_it --max-generated-tokens 128
uv run oasis chat --backend fake --model organization/compatible-chat-model
```

An explicit `--model`/`OASIS_MODEL_ID` overrides the named profile. Arbitrary Hugging Face IDs are
best-effort and must be generative causal models with a compatible chat template. Encoder-only
models, absent templates, inaccessible or gated weights, unsupported architectures, and models
requiring repository code fail with typed errors. `trust_remote_code` is false unless the
`--trust-remote-code` option is supplied deliberately.

## Opt-in real model chat

Gemma weights are distributed through the
[Gemma 4 model collection](https://huggingface.co/collections/google/gemma-4); access may require
accepting its license and authenticating. A real run may download many gigabytes and can be slow on
CPU. It is never part of `uv sync` or the default tests. Once access and sufficient disk/RAM are
available, the smallest profile can be exercised on CPU with:

```bash
HF_TOKEN=... uv run oasis chat \
  --backend transformers \
  --profile gemma4_e2b_it \
  --device cpu \
  --max-generated-tokens 64 \
  --prompt "Reply with one short sentence."
```

Use `--thinking` to enable Gemma 4's conversation-level thinking mode. Thought-channel text is
separated from public streamed text and is not printed by the CLI, while its tokens remain included
in usage. During a tool turn, thoughts remain available to the adapter until the tool response is
fed back, but they are never included in the public answer or required for correct execution.

The Transformers backend loads its processor and model lazily on the first request. It uses the
model's own chat template, `AutoProcessor` for Gemma 4, and `AutoModelForCausalLM` for raw text.
Explicit CPU placement is honored independently of profile selection. `auto` or `cuda` is an
explicit deployment choice and never causes OASIS to replace the requested model with a smaller
one.

## Portable inference runtimes

There are only three execution modes:

- local CPU;
- local CUDA, using one GPU when possible or Accelerate across the GPUs visible on the same
  machine;
- an authenticated remote model worker.

GPU visibility may come from a workstation, a rented host, or a scheduler allocation; OASIS treats
all of them identically and contains no SLURM-specific configuration. Multi-node inference is out
of scope. Most deployments need only `OASIS_DEVICE=cpu|cuda|auto`; remote deployments additionally
set `OASIS_RUNTIME_ENGINE=remote`, `OASIS_REMOTE_ENDPOINT`, and `OASIS_REMOTE_AUTH_TOKEN`.

Runtime policy and hardware discovery remain separate. `OASIS_DEVICE=cpu` is the safe default and
wins even when CUDA devices are visible. The planner considers only the supplied single-machine
inventory, preserves the exact model ID, reserves `OASIS_MEMORY_HEADROOM_FRACTION` (10% by
default), and returns a typed rejection when a requested dtype, quantization, attention backend,
or memory placement is not credible.

Safe CPU inspection never imports Torch or initializes CUDA:

```bash
uv run oasis hardware inspect
uv run oasis runtime plan --profile gemma4_e4b_it --device cpu
```

Named fixtures exercise deployment planning on any CPU-only machine. They are labeled
`discovery_mode: fake` and `dry_run: true` and make no claim about the current host:

```bash
uv run oasis hardware inspect --fixture local-5060ti-8gb
uv run oasis runtime plan --profile gemma4_e2b_it --device cuda \
  --fixture local-5060ti-16gb
uv run oasis runtime plan --profile gemma4_12b_it --device cuda \
  --fixture 2x24gb
uv run oasis runtime plan --profile gemma4_31b_it --device cuda \
  --fixture 4x80gb
```

The explicit probe observed 16,311 MiB on this local RTX 5060 Ti. The two named fixtures remain
fake dry-run scenarios: the full-checkpoint Gemma E2B estimate now rejects the 8 GiB fixture and
fits the 16 GiB fixture with the configured ten-percent headroom. Always use `--probe-cuda` for a
real host because current free memory, rather than the card name alone, controls admission.

The resolved `RuntimePlan` records model/revision, runtime kind, placement, dtype/quantization,
attention implementation, memory headroom and limits, offload policy, rationale, validation
status, startup time, peak device memory when observable, and generation throughput. Run records
also persist a sanitized `ComputeInventory`; device UUIDs are excluded from public results.
Evaluation code can use
`evaluation_group_key(plan, inventory)` so unlike hardware/runtime results are not pooled.

The base implementations are CPU Transformers, explicitly enabled single-GPU Transformers, and
Accelerate `device_map="auto"` dispatch. Accelerate dispatch is described only as a way to fit a
model across memory on one machine; it does not claim tensor-parallel speedup.

### Local CUDA workflow

CUDA inspection and model execution are never automatic. Install the locked CUDA runtime group,
inspect and dry-plan the exact visible allocation, then run the smoke command:

```bash
uv sync --frozen --no-group cpu --group gpu
uv run --no-sync oasis hardware inspect --probe-cuda
uv run --no-sync oasis runtime plan --probe-cuda --device cuda --profile gemma4_e2b_it
HF_TOKEN=... uv run --no-sync oasis chat --probe-cuda --backend transformers \
  --device cuda --profile gemma4_e2b_it --max-generated-tokens 32 \
  --prompt "Reply with one short sentence."
```

`--probe-cuda` is the explicit opt-in that imports Torch and queries only devices already visible to
the process. `--no-sync` is required after selecting the mutually exclusive GPU group; an ordinary
`uv run` restores the project's default CPU group. Model loading may download weights to `HF_HOME`.

Once the Gemma checkpoint is cached, run the opt-in GPU regression suite offline with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OASIS_RUN_GPU_TESTS=1 OASIS_GPU_TEST_MODEL=google/gemma-4-E2B-it \
uv run --no-sync pytest -m gpu tests/integration/test_gpu_runtime.py
```

On the validated host, Torch 2.11.0+cu128 reported CUDA 12.8 and compute capability 12.0. The suite
exercised inventory and driver metadata, BF16 matrix multiplication, SDPA, conservative placement,
normal and aborted streaming, and a complete anytime-controller run. The loaded checkpoint
contained 5.104 billion parameters and peaked at 10,236,558,336 allocator bytes in a short smoke
turn. Those observations establish compatibility for this machine; they are not general latency or
throughput claims.

For a single-node scheduler allocation, request resources outside OASIS and run the same three
commands above inside the allocation. `CUDA_VISIBLE_DEVICES` determines what the explicit probe can
see; scheduler names, ranks, and job IDs are neither parsed nor stored.

### Rented host or remote model worker

On an already provisioned GPU host, mount persistent Hugging Face cache and artifact/run
directories, clone the same revision, install `uv`, and use the normal commands directly. For the
complete service:

```bash
export HF_HOME=/workspace/hf-cache
export OASIS_ARTIFACT_ROOT=/workspace/oasis-artifacts
export OASIS_RUN_ROOT=/workspace/oasis-runs
uv sync --frozen --no-group cpu --group gpu
uv run --no-sync oasis serve --host 0.0.0.0 --probe-cuda --device cuda
```

For worker-only deployment, set a random bearer token in the environment—never a CLI argument—and
start the versioned streaming service:

```bash
export OASIS_MODEL_WORKER_AUTH_TOKEN=...
uv run --no-sync oasis model-worker serve --host 0.0.0.0 \
  --probe-cuda --device cuda --profile gemma4_e2b_it
```

Check it from the controller host, then configure the main service:

```bash
export OASIS_REMOTE_AUTH_TOKEN=...
uv run oasis model-worker health --endpoint https://worker.example
uv run oasis model-worker capabilities --endpoint https://worker.example

export OASIS_RUNTIME_ENGINE=remote
export OASIS_REMOTE_ENDPOINT=https://worker.example
uv run oasis serve
```

The client validates worker protocol version, model identity, and capabilities before generation.
Streaming generation, abort, terminal usage, and structured errors use `/api/v1`; authentication
failures and traces never echo the bearer token. Provider lifecycle and account automation remain
outside the application. Keep `OASIS_OFFLOAD_ROOT` on fast persistent storage when disk offload is
explicitly enabled, and synchronize `OASIS_ARTIFACT_ROOT` plus `OASIS_RUN_ROOT` to durable storage
before releasing an ephemeral host.

The project intentionally has no container image or provider-specific bootstrap layer. Those can
be added later if a concrete deployment needs them.

## Artifact store

`LocalArtifactStore` writes immutable bytes and complete metadata into a content-addressed
directory. Publication renames a prepared directory atomically, so readers never observe content
without metadata or metadata without content. Reads verify the SHA-256 identity by default, IDs
are parsed rather than treated as paths, and attempting to reuse bytes with different metadata is
rejected.

Large tool outputs must be stored as artifacts: the common `ToolResult` envelope limits the full
model-visible summary and exposes only compact artifact references. Artifact metadata includes
kind, media type, CRS and units, extents, counts, source/license/version, retrieval time, lineage,
quality warnings, and privacy classification. The storage protocol is independent of the local
filesystem implementation so an object store can replace it later.

Phase 3 adds deterministic codecs for the canonical evidence kinds:

- vectors use canonical GeoJSON with an explicit CRS;
- rasters use compressed GeoTIFF with its affine transform, CRS, nodata value, and band names;
- tables use Zstandard-compressed Parquet;
- directed, undirected, simple, and multigraphs use stable node-link JSON; and
- labeled matrices use a deterministic compressed NumPy archive.

Every codec requires explicit units and source/license provenance. Derived artifacts embed and
expose their parent IDs, operation version, and normalized parameters. Repeated operations over
the same immutable inputs therefore reproduce content and lineage hashes.

## Offline evidence demo

Run the complete Phase 3 public-library example through the CLI:

```bash
uv run oasis evidence demo --artifact-root artifacts/evidence-demo
```

The command creates fully synthetic projected population and candidate layers, retains population
need and group membership as separate dimensions, and emits a `DemandSpec`, `CandidateSpec`,
Euclidean access matrix, and binary-threshold service matrix. Its JSON output contains immutable
artifact IDs suitable for later problem compilation. Running it again with the same artifact root
reuses the same content-addressed artifacts.

The evidence plane also supports:

- schema/extent/missingness/duplicate/geometry/suppression profiling;
- CRS transformation, clipping, geometry repair, ID normalization, and declared unit conversion;
- spatial join, vector/raster zonal reduction, nearest feature, and raster sampling;
- counts, denominator-explicit rates, confidence intervals, direct age standardization, and
  small-cell suppression propagation;
- supplied-site and deterministic suitability-grid candidates;
- projected Euclidean, WGS84 geodesic, and directed graph-shortest-path matrices; and
- binary-threshold, piecewise-linear, and exponential-decay service responses.

Graph isochrones currently return auditable reachable-node sets. Polygonal isochrones are
explicitly deferred until a topology-safe polygonization contract is implemented.

## Open-data providers and snapshots

Phase 4 keeps live services behind four small protocols: ranked place resolution, catalog search,
source retrieval, and routed matrices. Built-in adapters support Nominatim-compatible geocoders,
STAC item search, generic HTTP CSV/GeoJSON retrieval, and OSRM-compatible table routing. Provider
objects are injected into tool contexts, so offline fixtures and alternate services produce the
same normalized tool and artifact schemas.

Place and catalog results are immediately stored as immutable JSON evidence artifacts. Their tool
summaries contain only counts, ambiguity state, and artifact IDs, keeping large or potentially
sensitive lookup details out of the compact model-visible trace.

All HTTP adapters share configured limits and resilience behavior:

- `OASIS_PROVIDER_USER_AGENT` identifies the installation; set a real contact string before using
  community services that require one.
- `OASIS_PROVIDER_TIMEOUT_SECONDS`, `OASIS_PROVIDER_MAX_ATTEMPTS`, and
  `OASIS_PROVIDER_BACKOFF_BASE_SECONDS` bound timeouts and exponential retry behavior.
- `OASIS_PROVIDER_MAX_RESPONSE_BYTES` and `OASIS_PROVIDER_MAX_PAGES` bound downloads and catalog
  pagination.
- `OASIS_PROVIDER_CACHE_ROOT` selects the local atomic snapshot index. Artifact bytes remain in
  `OASIS_ARTIFACT_ROOT` and are never mutated.

`snapshot_source` canonicalizes retrieved CSV as Parquet and GeoJSON as canonical GeoJSON before
returning it. A configured freshness window yields cache hits without network work. If refresh
fails, an acceptable cached snapshot is republished as an immutable child with an explicit stale
warning; an over-age/missing snapshot yields a typed provider failure. Optimization must consume
the returned snapshot artifact, never mutable response bytes.

Credentials belong in authentication hooks, not tool arguments. `BearerTokenAuthentication` and
`ApiKeyAuthentication` can inject headers or query values at the transport boundary; their
representations, structured logs, URLs, and provenance redact secret-like values. Avoid placing
sensitive record-level query values in provider URLs. Provider-specific response fields live only
inside opaque `provider_metadata` provenance, not canonical demand/access schemas. Artifact
metadata schema `1.1.0` adds this backwards-compatible envelope; readers accept earlier metadata
documents where it is absent.

Authenticated sources should also use a stable, non-secret `cache_partition` value so one
credential scope cannot reuse another scope's cache entry.

The live smoke workflow is deliberately opt-in and is not run by tests. It returns ranked place
candidates without choosing one, snapshots a public CSV/GeoJSON URL, and creates a small routed
duration matrix from explicitly supplied coordinates:

```bash
OASIS_PROVIDER_USER_AGENT="my-oasis-test/0.1 contact@example.org" \
uv run oasis evidence live-smoke \
  --confirm-live-network \
  --place "Cambridge, Massachusetts" \
  --source-url "https://example.org/public-data.geojson" \
  --source-format geojson \
  --source-license "CC-BY-4.0" \
  --source-units people \
  --source-crs EPSG:4326 \
  --route-coordinate=-71.1097,42.3736 \
  --route-coordinate=-71.0589,42.3601
```

Override `--nominatim-endpoint` or `--osrm-endpoint` for compatible deployments. Public demo
servers have their own acceptable-use, attribution, size, and rate policies; production use should
configure an appropriate endpoint. STAC assets discovered by `search_sources` are snapshotted with
the generic source adapter after their license, format, CRS, and units are declared.

## Location-allocation decision demo

Run the frozen cooling-center workflow entirely offline on CPU:

```bash
uv run oasis decision demo --artifact-root artifacts/decision-demo
```

The command creates synthetic population, group, facility, travel, and service evidence; compiles
an immutable maximum-weighted-coverage problem with a named group floor; commits a feasible greedy
baseline; runs resumable exact enumeration; independently rescores the retained plan; and writes a
summary plus GeoJSON and standalone SVG maps. Its JSON response reports the content-addressed
problem, baseline, best-plan, scorecard, summary, and map artifact IDs together with overall and
group metrics.

The location-allocation registry supports maximum weighted coverage, minimum-cost target coverage,
weighted p-median, p-center, configured quantile access, capacitated allocation with explicit unmet
demand, equity-floor and lexicographic max-min coverage, incremental siting around mandatory
existing facilities, and redundant/one-site-failure coverage. Every family uses the same immutable
`LocationAllocationProblem`, `Plan`, `ValidationReport`, and authoritative `Scorecard` contracts.
Problem and policy hashes lock all decision-relevant artifact identities, units, dimensions,
group/scenario definitions, constraints, and evaluator versions.

`compile_problem` rejects mismatched labels or units and policies for which it cannot commit a
feasible baseline. `improve` selects `add_swap`, bounded `multi_swap`, `local_assignment`,
`scenario_aware`, `exact_enumeration`, or `ortools_cp_sat` by one stable strategy value. Search
streams only candidates that pass independent validation and improve the fixed comparator; bounded
enumeration returns a resume token, while an uninterrupted complete enumeration and supported
CP-SAT coverage models emit checkable bound artifacts. CP-SAT currently accepts binary,
single-service-scenario coverage
families; fractional/multi-scenario coverage and access/capacity families use the deterministic
enumeration or local strategies. CP-SAT receives a hard remaining-time bound but cannot observe a
cooperative cancellation token during one native solver slice, so `improve` conservatively declares
that limitation in its tool contract. Tools never own or update a controller incumbent.

Location scenarios may independently replace demand weights or travel impedance and mark failed
facilities while retaining one immutable problem and comparator. Expected aggregation uses explicit
positive scenario weights; worst-case aggregation minimizes coverage and maximizes access cost.
Every scenario remains visible in the authoritative scorecard.

## Mobile-service routing demo

Run the frozen mobile-vaccination workflow entirely offline on CPU:

```bash
uv run oasis decision mobile-demo --artifact-root artifacts/mobile-routing-demo
```

The shared problem registry supports directed TSP calibration, prize-collecting orienteering, and
capacity- and time-window-constrained mobile-service routes. Route plans use the same neutral
`Plan`, `ValidationReport`, `Scorecard`, problem-plugin, compile, improve, and summary contracts as
location allocation. Validation independently enforces vehicle/depot structure, globally unique
service visits, reachability in every scenario, service windows, shift length, and capacity.

Nearest and prize-aware insertion construct deterministic baselines. `improve` selects `two_opt`,
`relocate`, `swap`, resumable `exact_enumeration`, or bounded `ortools_routing` through its existing
strategy field. Exact route enumeration currently supports one vehicle. The OR-Tools routing
strategy currently supports depot-return problems with one travel scenario and emits a bounded
solver record, not an optimality certificate. Candidate plans are always independently rescored.

`scenario_sweep` accepts complete, explicitly named policy alternatives. It publishes each as a
separate immutable problem with a distinct policy/problem hash and feasible baseline when one can
be constructed; it never mutates the source policy or compares scorecards across hashes.

## Anytime controller demo

Run a complete fake-model, real-tool, independently evaluated coverage search on CPU:

```bash
uv run oasis decision anytime-demo --artifact-root artifacts/anytime-demo
```

The command freezes the synthetic evidence, compiles an immutable problem and feasible baseline,
asks the scripted model for exactly one `improve` action, streams independently checked candidates
and a verified exact bound, and deterministically returns the retained incumbent. It persists
`run.json`, ordered `events.jsonl`, and `result.json` below the artifact root's `runs/` directory.
Repeated invocations use distinct opaque run IDs.

Budgets can be changed without source edits:

```bash
uv run oasis decision anytime-demo \
  --wall-time-ms 30000 \
  --max-total-model-tokens 2000 \
  --max-generated-tokens 256 \
  --max-tool-calls 2
```

`BudgetSpec` protects a configurable finalization reserve and enforces aggregate input/output,
generated-token, and tool-call limits. Its four execution tiers are baseline-only, deterministic
improvement, one-shot model selection, and iterative model control. If model tokens expire while
wall time and tool calls remain, the problem plugin's deterministic fallback ladder continues.
The model receives only a compact state containing the immutable problem hash, incumbent metrics,
recent action summaries, available tools, and remaining budget; full traces and large artifacts are
never replayed into its prompt.

All streamed or directly submitted plans are revalidated and rescored by the problem plugin before
an atomic comparator-based incumbent replacement. Tool/model failure, malformed or duplicate
actions, timeout, and user cancellation cannot erase that incumbent. Final rendering uses the
plugin directly and needs no model tokens. Tool prerequisites, privacy classification, JSON Schema,
latency estimates, subdeadlines, cancellation, and generation IDs are checked before or during each
admitted action; events from closed action generations are ignored.

The local run-store implementation remains behind a protocol; the service adds lifecycle and
streaming without changing controller semantics. The current in-process executor can cooperatively
cancel or task-cancel tools; process hard-kill is used only when a future isolated worker declares
it safe. The default acceptance suite is CPU/fake only; a real local Gemma/GPU end-to-end checkpoint
requires an environment with GPU access and model credentials.

## Versioned service API

Start the headless versioned service with the deterministic fake backend:

```bash
uv run oasis serve --backend fake \
  --artifact-root artifacts/service \
  --run-root artifacts/service-runs
```

The OpenAPI document is at `http://127.0.0.1:8000/api/v1/openapi.json` and interactive API docs
are at `http://127.0.0.1:8000/api/v1/docs`. The backend object belongs to the service lifecycle:
the fake backend is immediately ready, while a Transformers backend loads its processor and weights
lazily on the first model request and is reused across runs. Model startup metadata is reported
separately from each run's wall-time and token accounting.

The stable `/api/v1` surface is:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness without model or hardware probing |
| `GET` | `/models` | Model profiles and declared capabilities |
| `GET` | `/runtime` | Requested policy, known plan/capabilities, and sanitized known inventory |
| `GET` | `/tools` | Typed tool catalog |
| `GET` | `/problems` | Problem-plugin catalog |
| `POST` | `/chat` | Bounded raw chat through the service-owned backend |
| `POST` | `/runs` | Immediately accept a capacity-available asynchronous run |
| `GET` | `/runs/{run_id}` | Inspect current state or the persisted final result |
| `GET` | `/runs/{run_id}/events` | Replay and follow ordered Server-Sent Events |
| `POST` | `/runs/{run_id}/cancel` | Idempotently cancel and return the retained result |
| `GET` | `/artifacts/{artifact_id}` | Retrieve an allowlisted public artifact |
| `GET` | `/runs/{run_id}/map` | Render/retrieve a validated location-plan map |

Create a run by referring to problem and optional baseline artifacts already in the configured
artifact store:

```bash
curl -sS http://127.0.0.1:8000/api/v1/runs \
  -H 'Content-Type: application/json' \
  --data '{
    "source": {
      "kind": "artifact",
      "problem_artifact_id": "sha256-REPLACE_WITH_64_HEX_DIGITS",
      "baseline_plan_artifact_id": "sha256-REPLACE_WITH_64_HEX_DIGITS"
    },
    "budget": {
      "wall_time_ms": 30000,
      "max_total_model_tokens": 2000,
      "max_generated_tokens": 256,
      "max_tool_calls": 2
    }
  }'
```

`source.kind` may instead be `inline` with a complete compiled location/routing problem and an
optional baseline `Plan`, or `compile_problem` with the stable tool's structured evidence/policy
arguments. Structured compilation requires and consumes one declared tool call, and its time is
included in the request wall deadline. All accepted forms lock an immutable problem before
controller search.

The Phase 11 wire revision also accepts `source.kind: "example"` using an ID advertised by
`GET /api/v1/problems`. Example evidence is frozen, synthetic, and CC0; preparation is performed
through the same evidence and compilation tools and is charged against the declared tool-call and
wall budgets. `POST /runs` additionally accepts optional `model_profile`, `model_id`, and
`runtime_policy` fields. The service resolves and caches the selected lazy backend without changing
the requested model; omitting these fields retains the server policy.

SSE reconnect uses the normal `Last-Event-ID` header. The server replays every persisted event
after that sequence and then follows new events; subscribers receive wake-up signals only, so a
slow or disconnected client cannot backpressure the controller:

```bash
curl -N http://127.0.0.1:8000/api/v1/runs/RUN_ID/events
curl -N http://127.0.0.1:8000/api/v1/runs/RUN_ID/events -H 'Last-Event-ID: 5'
```

Completed results and traces survive service restart when the same `OASIS_RUN_ROOT` is mounted.
The local V1 service assumes one trusted local owner and exposes only artifacts marked `public`,
with explicit kind/content-type and byte-size allowlists. Request and artifact limits, concurrency,
paths, and SSE heartbeat are configurable through `OASIS_API_MAX_REQUEST_BYTES`,
`OASIS_API_MAX_ARTIFACT_RESPONSE_BYTES`, `OASIS_API_MAX_CONCURRENT_RUNS`, `OASIS_ARTIFACT_ROOT`,
`OASIS_RUN_ROOT`, and `OASIS_API_SSE_HEARTBEAT_SECONDS`. Capacity exhaustion returns a structured
`503` instead of queuing unbounded work. Cancellation waits at most
`OASIS_API_CANCEL_WAIT_SECONDS` for a final retained result before returning an acknowledgement;
API errors never include tracebacks or request records.

Static UI serving is disabled by default and is only deployment plumbing. Once a `ui/` directory
exists, enable it with `--serve-ui --ui-root ui` or `OASIS_SERVE_UI=true`; every API route remains
usable without those files.

## Independent web interface

Launch the complete offline UI demonstration with the fake backend:

```bash
uv run oasis serve --backend fake --serve-ui --ui-root ui \
  --artifact-root artifacts/ui-demo \
  --run-root artifacts/ui-demo-runs
```

Open `http://127.0.0.1:8000/`. There is no frontend installation or build command: `ui/index.html`,
CSS design tokens, and small native JavaScript modules are served as static files. The HTTP/SSE
client is isolated in `ui/src/api.js`; form, chart, state, and map code do not construct endpoint
URLs. Replacing the entire directory therefore requires no Python changes while `/api/v1` remains
compatible, and the backend and evaluator continue to work headlessly.

The form loads model profiles, launchable public-health examples, tools, problem plugins, and
runtime choices from the service. It exposes Gemma 4 profile and explicit Hugging Face ID
selection, wall/model-token/generated-token/tool-call budgets, server-advertised equity templates
and group floors, and advanced device/engine/dtype/quantization settings. The default advanced
choice preserves the server's deployment policy. Reasoning mode carries a visible warning because
reasoning tokens consume budget; private thought content is never shown.

During a run, the client keeps the last independently verified scorecard and map visible while new
work continues. It shows raw, group, and scenario metrics, baseline-relative comparator gain,
remaining budget, an incumbent quality curve, ordered action/tool events, requested versus
resolved runtime, sanitized hardware, evidence age, warnings, and termination status. SSE is read
through a reconnecting fetch stream so the client can send `Last-Event-ID`; compact UI state is
also kept in browser local storage, allowing a refresh or disconnect without discarding the
displayed incumbent. Cancellation retains that incumbent.

For a manual smoke check, launch each advertised example, confirm that a baseline map and metric
appear before completion, refresh during a run to exercise replay, cancel an active run, and check
failure/completed messages. Repeat once below 720 CSS pixels and once at a wide desktop viewport;
all controls and trace entries remain keyboard reachable, with shape and text reinforcing color.
The default test suite verifies static/API contract alignment and does not require a browser,
network, model weights, or Node.

## Reproducible evaluation

Run the shipped two-family CPU/fake smoke benchmark, then recompute its summary from the raw run
records:

```bash
uv run oasis evaluate src/oasis/evaluation/manifests/smoke.json \
  --output evaluation-output/offline-smoke-v1
uv run oasis summarize evaluation-output/offline-smoke-v1
```

Evaluation manifests are versioned JSON documents. They lock the instance or frozen fixture,
evaluation track, comparison policies, named budgets, paired run seeds, model/profile or explicit
Hugging Face ID, requested runtime policy, deterministic tool portfolio, generator/reference
limits, and expected evaluator versions. Re-running the same command resumes from atomically
completed run documents; a results directory cannot be reused with a different manifest hash.

The built-in comparisons are a strong problem-specific baseline, fixed deterministic portfolio,
one-shot model strategy choice, iterative model-controlled portfolio, and direct model candidate
construction. Each comparison is one continuing controller run. Quality curves therefore come
from its independently scored incumbent timeline rather than from unrelated reruns. The evaluator
ignores assistant prose, plan metadata that claims a score, and solver-reported objectives.

Synthetic generator version `1.0.0` supports uniform, clustered, grid, ring, corridor, islands,
and outlier geography; heterogeneous demand/capacity; balanced, isolated, or empty group
structures; feasible, tight, and deliberately infeasible constraints; directed travel,
unreachable arcs, scenarios, and explicit development/held-out seed namespaces. Six frozen CC0
definitions cover cooling centers, clinic access, capacity allocation, resilient coverage, mobile
service routing, and environmental monitors. Small cases receive a complete enumeration
certificate. Medium and stress cases are labeled best-known unless the configured exact limit
actually proves the full candidate space.

Every completed run is stored as `runs/RUN_KEY.json`, with a rebuilt `runs.jsonl`. Aggregate output
is written to `summary.json`, `aggregate.csv`, and `aggregate.parquet`. Reports retain feasibility,
raw objective, overall/group/scenario metrics, baseline gain, reference gap, fixed wall/token
checkpoints, complete incumbent curves, log-resource AUC, deadline overshoot, token categories,
tool latency/failures, versions, and seeds. Repetitions use paired seeds and descriptive mean,
sample variance, and normal-approximation intervals. These intervals are not significance claims.
Results are always stratified by recorded model, runtime, visible hardware, driver, and library
versions rather than pooling unlike systems.

Real-model evaluation is deliberately double opt-in: set `model.backend` to `transformers` and
`model.allow_real_model` to `true` in the manifest, obtain the required hardware/model approval,
then pass `--confirm-real-model-evaluation`. Any named Gemma 4 profile or `model.model_id` is
accepted through the normal model registry. CPU remains the default. Only an approved CUDA run
should add `--probe-cuda`; default evaluation never imports Torch or initializes CUDA for hardware
discovery.

## Tool commands

```bash
uv run oasis tools list
uv run oasis tools describe calculator
uv run oasis tools smoke calculator
uv run oasis tools smoke calculator \
  --input '{"operation":"multiply","operands":[6,7]}'
```

`list`, `describe`, and `smoke` validate built-ins and installed `oasis.tools` entry points before
use. Add `--no-plugins` immediately after `tools` to inspect built-ins alone. Smoke tests use the
tool's declared sample input and an absolute deadline; an alternate artifact root can be passed
with `--artifact-root`. Evidence-tool smoke calls require the artifact IDs named by their schemas;
use `oasis evidence demo` to create a complete compatible set first.

Each `ToolSpec` declares versioned input/output JSON Schemas, capability/problem/artifact tags,
side effects, privacy, providers/resources, determinism and seed behavior, a cost estimator,
runtime quantiles, streaming behavior, cancellation/hard-kill properties, and resumability. Each
invocation receives a `ToolContext` with its run ID, artifact store, absolute deadline,
cancellation token, seed, logger, and explicitly allowed handles.

## Adding a tool

The normal onboarding path is deliberately five steps:

1. Implement one class with an immutable `ToolSpec` and an `async run(arguments, context)` method.
   A streaming implementation may instead expose an async `stream` generator matching its flags.
2. Declare the spec beside the implementation, including object-valued input/output JSON Schemas,
   runtime/cancellation behavior, and a valid `smoke_input`.
3. Register built-ins in `oasis.tools.builtins.builtin_tools`, or publish third-party tools through
   an `oasis.tools` package entry point that returns a tool or iterable of tools:

   ```toml
   [project.entry-points."oasis.tools"]
   community_tools = "community_package.tools:create_tools"
   ```

4. Run the reusable `oasis.tools.testing.assert_tool_contract` check with a fresh `ToolContext`,
   then add tool-specific success, failure, cancellation, and serialization tests.
5. Add the tool to the catalog below. Normal tools require no controller, API, UI, or model-adapter
   changes.

| Tool | Version | Purpose | Side effects |
|---|---:|---|---|
| `build_candidates` | 1.0.0 | Build supplied or deterministic-grid facility candidates | Local artifacts |
| `build_demand` | 1.0.0 | Preserve typed need, group, time, and suppression dimensions | Local artifacts |
| `calculator` | 1.0.0 | Basic bounded arithmetic for contract and tool-loop tests | None |
| `compile_problem` | 1.1.0 | Compile immutable location or route evidence/policy and seed a feasible baseline | Local artifacts |
| `derive_health_measure` | 1.0.0 | Compute counts, rates, and direct age-standardized rates | Local artifacts |
| `improve` | 1.1.0 | Run a resumable location/routing strategy and stream verified improvements | Local artifacts |
| `isochrones` | 1.0.0 | Compute graph reachable-node sets at explicit cutoffs | Local artifacts |
| `normalize_artifact` | 1.0.0 | Normalize CRS, geometry, IDs, clipping, and units | Local artifacts |
| `overlay_reduce` | 1.0.0 | Join, reduce, find nearest features, and sample rasters | Local artifacts |
| `profile_artifact` | 1.0.0 | Measure schema, extent, missingness, geometry, and suppression | Local artifacts |
| `render_map` | 1.0.0 | Render a validated location plan as GeoJSON or standalone SVG | Local artifacts |
| `resolve_area` | 1.0.0 | Return explicitly ranked place/area candidates | External read |
| `resolve_locations` | 1.0.0 | Resolve multiple locations without hiding ambiguity | External read |
| `scenario_sweep` | 1.0.0 | Compile isolated policy variants with distinct immutable hashes | Local artifacts |
| `search_sources` | 1.0.0 | Search and snapshot normalized STAC catalog metadata | External read |
| `service_matrix` | 1.0.0 | Convert access impedance to bounded service benefit | Local artifacts |
| `snapshot_source` | 1.0.0 | Canonicalize a freshness-aware CSV/GeoJSON snapshot | External read |
| `summarize_plan` | 1.1.0 | Publish independently measured location/routing and scenario metrics | Local artifacts |
| `travel_matrix` | 1.1.0 | Compute local or provider-routed access matrices | Local/external read |

## Adding a problem

1. Implement the shared `ProblemPlugin` validation, baseline, plan validation, measurement,
   comparison, and fallback-strategy contract without importing a controller or API framework.
2. Define a frozen, versioned problem schema whose hash includes every decision-relevant input,
   policy, and evaluator version.
3. Register the plugin explicitly in `create_problem_registry`; normal controller-facing code
   selects it only by `type_id`.
4. Add independent tiny-oracle, infeasibility, baseline, improvement, cancellation/resumption, and
   serialization tests. Never reuse a solver-reported objective as the authoritative score.
5. Add the family and any supported strategy restrictions to this catalog. A normal problem plugin
   does not require changes to the model adapter, artifact store, provider layer, API, or UI.

## Adding a provider

1. Implement one or more protocols from `oasis.providers`: `PlaceResolver`, `CatalogSearcher`,
   `SourceSnapshotProvider`, or `RoutingMatrixProvider`.
2. Normalize responses into the corresponding provider model and attach a UTC retrieval time,
   redacted source URI, source version where available, and opaque provider metadata.
3. Use `ResilientHttpClient` (or enforce equivalent timeout, retry, rate, page, and byte limits),
   and inject credentials through an authentication hook.
4. Supply the adapter in `ToolContext.providers` under the exported handle name. Supply a
   `SnapshotCache` in `ToolContext.resources` for `snapshot_source`; no controller, API, or domain
   schema edit is needed.
5. Test success, ambiguity/empty results, malformed/oversized responses, timeout/rate limiting,
   pagination, stale fallback, cancellation, provenance, and redaction with mocked HTTP or frozen
   fixtures. Live tests must carry the `network` marker and require explicit opt-in.

## Phase 12 package map

- `oasis.config`: environment/object/CLI settings and non-probing runtime policy.
- `oasis.llm.schemas`: portable messages, requests, deltas, turns, tools, capabilities, and usage.
- `oasis.llm.profiles`: the sole Gemma 4 profile registry.
- `oasis.llm.adapters`: plain, Gemma 4 native-tool, and tagged fallback formatting/parsing.
- `oasis.llm.fake`: deterministic offline streaming backend.
- `oasis.llm.transformers_backend`: lazy real-model streaming backend.
- `oasis.llm.runtime_backend`: public model-backend facade for remote/plugin runtimes.
- `oasis.llm.tool_loop`: bounded parsing repair and registry-driven model/tool exchange.
- `oasis.schemas`: portable artifact, plan-envelope, tool, event, and result schemas.
- `oasis.artifacts`: object-store-neutral protocol and atomic local content-addressed store.
- `oasis.artifacts.codecs`: deterministic vector, raster, table, graph, matrix, and JSON codecs.
- `oasis.schemas.evidence`: canonical demand, candidate, access, and service evidence contracts.
- `oasis.tools`: tool protocols, registry/discovery, execution, contract checks, and built-ins.
- `oasis.tools.evidence`: deterministic profiling, normalization, overlay, health, construction,
  local/provider travel, isochrone, and service operations.
- `oasis.providers`: provider protocols and normalized response models, bounded HTTP resilience,
  Nominatim/STAC/HTTP/OSRM adapters, redaction, and local/in-memory snapshot cache indices.
- `oasis.tools.providers`: stable provider-backed `resolve_area`, `resolve_locations`,
  `search_sources`, and `snapshot_source` evidence tools.
- `oasis.evidence`: the public frozen evidence-plane example used by the CLI.
- `oasis.problems`: immutable location and route problem/policy/evaluation schemas, the shared
  plugin registry, deterministic baselines, scenario evaluators, exact/local search, and OR-Tools.
- `oasis.tools.decision`: stable compile, resumable improve, immutable policy sweep, summary, and
  location GeoJSON/SVG map tools.
- `oasis.decision`: the public frozen cooling-center workflow used by the CLI.
- `oasis.routing`: the public frozen mobile-vaccination workflow used by the CLI.
- `oasis.controller.schemas`: typed budgets, states, terminal reasons, model actions, events,
  compact state, incumbent records, and final run results.
- `oasis.controller.budget`: absolute monotonic deadlines plus exact aggregate token/tool ledgers.
- `oasis.controller.state`: legal state transitions, action generations, duplicate detection,
  redaction, and ordered event construction.
- `oasis.controller.incumbent`: atomic feasible-only, comparator-monotone incumbent replacement.
- `oasis.controller.store`: replaceable in-memory and local JSON/JSONL run/trace persistence.
- `oasis.controller.runner`: framework-neutral admission, model/tool scheduling, candidate/bound
  verification, circuit breakers, deterministic fallback, quiescence, and final rendering.
- `oasis.anytime`: the public frozen fake-model coverage run used by the Phase 7 CLI demo.
- `oasis.api.schemas`: independently versioned request, response, catalog, runtime, run, and error
  wire envelopes.
- `oasis.api.lifecycle`: one lazy/preloaded model backend shared for the complete service lifetime.
- `oasis.api.manager`: capacity-limited controller tasks, source preparation, cancellation,
  restart-safe inspection, map rendering, and non-blocking event subscriber notifications.
- `oasis.api.app`: FastAPI routes, OpenAPI generation, SSE replay/reconnect, request/artifact guards,
  structured error handling, and optional static-file deployment plumbing.
- `oasis.api.examples`: server-advertised, frozen public-health examples prepared through canonical
  evidence and problem tools for the independent client.
- `oasis.runtimes.schemas`: typed inventory, plan, capability, rejection, and measurement
  records with stable serialization/evaluation grouping.
- `oasis.runtimes.inventory`: non-probing CPU discovery, named single-machine fake inventories, and
  explicit CUDA inspection.
- `oasis.runtimes.planner`: conservative model-preserving placement and compatibility validation.
- `oasis.runtimes.transformers`: CPU, single-GPU, and memory-oriented Accelerate execution adapters.
- `oasis.runtimes.remote`: authenticated health/capability/generation/abort/usage worker client.
- `oasis.model_worker`: minimal authenticated, versioned model-worker API with NDJSON streaming.
- `oasis.evaluation.models`: benchmark manifests, generator/reference, raw run, curve, aggregate,
  and paired descriptive-statistics schemas.
- `oasis.evaluation.generators`: versioned seeded location/routing generators and disjoint
  development/held-out seed namespaces.
- `oasis.evaluation.oracles`: complete small-instance enumeration and bounded, independently
  rescored medium/stress references.
- `oasis.evaluation.runner`: resumable paired comparison execution through the public controller.
- `oasis.evaluation.metrics` and `oasis.evaluation.reporting`: fixed-checkpoint/log-AUC metrics,
  hardware-stratified summaries, and raw JSON/JSONL plus aggregate CSV/Parquet publication.
- `oasis.showcase`: the frozen seven-problem manifest runner, concise absolute-metric result, and
  token/context/latency/artifact/deadline plus provenance/privacy release audits.
- `oasis.failure_injection`: deterministic opt-in provider, model, and tool failures used by the
  end-to-end hardening suite.
- `ui/src/api.js`: the only browser HTTP/SSE boundary; it supports ordered replay with
  `Last-Event-ID`, cancellation, artifacts, and map retrieval.
- `ui/src/app.js`, `map.js`, `chart.js`, and `state.js`: framework-free accessible form, run view,
  GeoJSON/SVG rendering, quality history, and reconnect-safe display state.

# OASIS Anytime GeoAI Agent

OASIS is a budget-aware GeoAI agent for public-health and spatial-equity planning. The current
Phase 5 foundation provides typed configuration, model-independent streaming chat, Gemma 4
profiles, immutable content-addressed artifacts, a validated async tool SDK, deterministic and
live/snapshotted evidence planes, and independently evaluated location-allocation optimization.
Gemma 4 uses its native tool format; compatible non-Gemma chat models use a conservative
tagged-JSON fallback. The fake backend and frozen fixtures exercise the complete current workflow
without model weights, network access, or a GPU. Anytime orchestration, the service API, routing,
portable accelerator runtimes, evaluation harness, and UI intentionally arrive in later phases.

## Setup

Install [uv](https://docs.astral.sh/uv/), then create the locked Python 3.12 environment:

```bash
uv sync
```

The repository configures a local `.uv-cache` so setup does not depend on a writable user cache.
The base environment contains the CPU-side dependencies planned for the V1 implementation, and
Torch is locked to its official CPU wheel index. Model weights are never downloaded by setup or by
the default test suite.

Run the common quality gate with:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
git diff --check
```

The default tests are offline and CPU-only. Tests that eventually require network, model weights,
GPU, multi-GPU, SLURM, or long runtimes use explicit pytest markers and opt-in configuration.

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

Gemma weights require accepting the model license in the
[Gemma 4 model collection](https://huggingface.co/collections/google/gemma-4) and authenticating
with an account that has access. A real run may download many gigabytes and can be slow on CPU. It
is never part of `uv sync` or the default tests. Once credentials, disk/RAM, and the project's
explicit hardware-approval policy are satisfied, the smallest profile can be exercised on CPU
with:

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
| `compile_problem` | 1.0.0 | Compile immutable location evidence/policy and seed a feasible baseline | Local artifacts |
| `derive_health_measure` | 1.0.0 | Compute counts, rates, and direct age-standardized rates | Local artifacts |
| `improve` | 1.0.0 | Run a resumable registered search strategy and stream verified improvements | Local artifacts |
| `isochrones` | 1.0.0 | Compute graph reachable-node sets at explicit cutoffs | Local artifacts |
| `normalize_artifact` | 1.0.0 | Normalize CRS, geometry, IDs, clipping, and units | Local artifacts |
| `overlay_reduce` | 1.0.0 | Join, reduce, find nearest features, and sample rasters | Local artifacts |
| `profile_artifact` | 1.0.0 | Measure schema, extent, missingness, geometry, and suppression | Local artifacts |
| `render_map` | 1.0.0 | Render a validated location plan as GeoJSON or standalone SVG | Local artifacts |
| `resolve_area` | 1.0.0 | Return explicitly ranked place/area candidates | External read |
| `resolve_locations` | 1.0.0 | Resolve multiple locations without hiding ambiguity | External read |
| `search_sources` | 1.0.0 | Search and snapshot normalized STAC catalog metadata | External read |
| `service_matrix` | 1.0.0 | Convert access impedance to bounded service benefit | Local artifacts |
| `snapshot_source` | 1.0.0 | Canonicalize a freshness-aware CSV/GeoJSON snapshot | External read |
| `summarize_plan` | 1.0.0 | Publish independently measured overall/group/scenario plan metrics | Local artifacts |
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

## Phase 5 package map

- `oasis.config`: environment/object/CLI settings and non-probing runtime policy.
- `oasis.llm.schemas`: portable messages, requests, deltas, turns, tools, capabilities, and usage.
- `oasis.llm.profiles`: the sole Gemma 4 profile registry.
- `oasis.llm.adapters`: plain, Gemma 4 native-tool, and tagged fallback formatting/parsing.
- `oasis.llm.fake`: deterministic offline streaming backend.
- `oasis.llm.transformers_backend`: lazy real-model streaming backend.
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
- `oasis.problems`: immutable location problem/policy/evaluation schemas, the plugin protocol and
  registry, all Phase 5 family evaluators, deterministic baselines, exact/local search, and CP-SAT.
- `oasis.tools.decision`: stable compile, resumable improve, summary, and GeoJSON/SVG map tools.
- `oasis.decision`: the public frozen cooling-center workflow used by the CLI.

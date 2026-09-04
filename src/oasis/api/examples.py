"""Server-owned frozen examples advertised to the independent web client."""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
from shapely.geometry import Point

from oasis.api.schemas import ExampleProblemSource, ProblemExampleEntry
from oasis.artifacts import ArtifactProvenance, ArtifactStore, put_vector
from oasis.schemas import ToolResult, ToolResultStatus
from oasis.tools import ToolContext, ToolRegistry, invoke_tool


@dataclass(frozen=True, slots=True)
class _ExampleDefinition:
    entry: ProblemExampleEntry
    policy: dict[str, object]


_EVIDENCE_SUMMARY = (
    "Frozen CC0 synthetic neighborhood demand, candidate facilities, travel, and service evidence."
)

_DEFINITIONS = (
    _ExampleDefinition(
        entry=ProblemExampleEntry(
            id="cooling_center_coverage",
            name="Cooling-center coverage",
            description="Choose one cooling center while protecting access for older adults.",
            problem_type="max_weighted_coverage",
            evidence_summary=_EVIDENCE_SUMMARY,
            group_names=("older_adults",),
            equity_templates=("overall", "floors"),
            default_equity_template="floors",
            default_group_floors={"older_adults": 0.4},
        ),
        policy={"site_limit": 1},
    ),
    _ExampleDefinition(
        entry=ProblemExampleEntry(
            id="clinic_average_access",
            name="Clinic average access",
            description="Place one clinic to minimize population-weighted travel distance.",
            problem_type="weighted_p_median",
            evidence_summary=_EVIDENCE_SUMMARY,
            group_names=("older_adults",),
            equity_templates=("overall",),
            default_equity_template="overall",
        ),
        policy={"site_limit": 1},
    ),
    _ExampleDefinition(
        entry=ProblemExampleEntry(
            id="emergency_tail_access",
            name="Emergency tail access",
            description="Place one service point to minimize the 90th-percentile access burden.",
            problem_type="quantile_access",
            evidence_summary=_EVIDENCE_SUMMARY,
            group_names=("older_adults",),
            equity_templates=("overall",),
            default_equity_template="overall",
        ),
        policy={"site_limit": 1, "quantile": 0.9},
    ),
    _ExampleDefinition(
        entry=ProblemExampleEntry(
            id="equity_first_coverage",
            name="Equity-first coverage",
            description="Maximize worst-group coverage before overall population coverage.",
            problem_type="equity_coverage",
            evidence_summary=_EVIDENCE_SUMMARY,
            group_names=("older_adults",),
            equity_templates=("floors", "max_min"),
            default_equity_template="max_min",
            default_group_floors={"older_adults": 0.4},
        ),
        policy={"site_limit": 1},
    ),
)

_BY_ID = {definition.entry.id: definition for definition in _DEFINITIONS}


def example_catalog() -> tuple[ProblemExampleEntry, ...]:
    """Return the complete server-authorized UI example inventory."""

    return tuple(definition.entry for definition in _DEFINITIONS)


@dataclass(frozen=True, slots=True)
class PreparedExample:
    """Immutable problem identities and charged preparation work for one example."""

    problem_artifact_id: str
    baseline_plan_artifact_id: str
    tool_calls: int


def _provenance(role: str) -> ArtifactProvenance:
    return ArtifactProvenance(
        source_uri=f"oasis://api/v1/examples/public-health/{role}",
        source_provider="oasis-ui-examples",
        source_version="1.0.0",
        license="CC0-1.0",
    )


def _publish_sources(store: ArtifactStore) -> tuple[str, str]:
    demand = gpd.GeoDataFrame(
        {
            "demand_id": ["west", "central", "east"],
            "population": [120.0, 80.0, 100.0],
            "older_adult": [0, 1, 1],
            "suppressed": [False, False, False],
        },
        geometry=[Point(0, 0), Point(1_000, 0), Point(2_000, 0)],
        crs="EPSG:3857",
    )
    candidates = gpd.GeoDataFrame(
        {
            "site_id": ["west-library", "east-school"],
            "opening_cost": [1.0, 1.5],
            "capacity": [200.0, 250.0],
        },
        geometry=[Point(0, 0), Point(2_000, 0)],
        crs="EPSG:3857",
    )
    demand_ref = put_vector(store, demand, units="persons", provenance=_provenance("demand"))
    candidate_ref = put_vector(
        store,
        candidates,
        units="meters",
        provenance=_provenance("candidates"),
    )
    return demand_ref.id, candidate_ref.id


async def _complete(
    registry: ToolRegistry,
    name: str,
    arguments: dict[str, object],
    context: ToolContext,
) -> ToolResult:
    result = await invoke_tool(registry.get(name), arguments, context)
    if result.status is not ToolResultStatus.COMPLETE:
        message = result.error.message if result.error is not None else f"{name} did not complete"
        raise ValueError(message)
    return result


def _policy(definition: _ExampleDefinition, source: ExampleProblemSource) -> dict[str, object]:
    if source.equity_template not in definition.entry.equity_templates:
        raise ValueError("the selected equity template is not available for this example")
    unexpected = set(source.group_floors) - set(definition.entry.group_names)
    if unexpected:
        raise ValueError("group floors contain groups that are not present in the example")

    policy = dict(definition.policy)
    if source.equity_template == "overall":
        policy.update({"equity_objective": "none", "group_floors": {}})
    elif source.equity_template == "floors":
        floors = source.group_floors or definition.entry.default_group_floors
        policy.update({"equity_objective": "floors", "group_floors": floors})
    else:
        policy.update({"equity_objective": "max_min", "group_floors": {}})
    return policy


async def prepare_example(
    source: ExampleProblemSource,
    *,
    store: ArtifactStore,
    registry: ToolRegistry,
    context: ToolContext,
) -> PreparedExample:
    """Build canonical frozen evidence and compile one advertised example."""

    try:
        definition = _BY_ID[source.example_id]
    except KeyError as error:
        raise ValueError("the selected example is not advertised by this service") from error

    demand_source_id, candidate_source_id = _publish_sources(store)
    demand = await _complete(
        registry,
        "build_demand",
        {
            "artifact_id": demand_source_id,
            "location_id_field": "demand_id",
            "need_fields": ["population"],
            "group_fields": ["older_adult"],
            "suppression_fields": ["suppressed"],
            "missing_data_policy": "error",
            "spatial_resolution": "frozen synthetic neighborhoods",
        },
        context,
    )
    candidates = await _complete(
        registry,
        "build_candidates",
        {
            "mode": "supplied",
            "artifact_id": candidate_source_id,
            "candidate_id_field": "site_id",
            "opening_cost_field": "opening_cost",
            "capacity_field": "capacity",
        },
        context,
    )
    access = await _complete(
        registry,
        "travel_matrix",
        {
            "origins_artifact_id": demand.metrics["demand_artifact_id"],
            "destinations_artifact_id": candidates.metrics["candidate_artifact_id"],
            "origin_id_field": "demand_id",
            "destination_id_field": "site_id",
            "strategy": "euclidean",
            "output_units": "meters",
        },
        context,
    )
    service = await _complete(
        registry,
        "service_matrix",
        {
            "access_matrix_artifact_id": access.metrics["artifact_id"],
            "strategy": "binary_threshold",
            "threshold": 1_000,
        },
        context,
    )
    compiled = await _complete(
        registry,
        "compile_problem",
        {
            "type_id": definition.entry.problem_type,
            "demand_spec_artifact_id": demand.metrics["demand_spec_artifact_id"],
            "candidate_spec_artifact_id": candidates.metrics["candidate_spec_artifact_id"],
            "access_matrix_artifact_id": access.metrics["artifact_id"],
            "service_matrix_artifact_ids": {"normal": service.metrics["artifact_id"]},
            "need_field": "population",
            "groups": [{"name": "older_adults", "field": "older_adult"}],
            "policy": _policy(definition, source),
        },
        context,
    )
    problem_id = compiled.metrics.get("problem_artifact_id")
    baseline_id = compiled.metrics.get("baseline_plan_artifact_id")
    if not isinstance(problem_id, str) or not isinstance(baseline_id, str):
        raise ValueError("the example compiler did not return problem and baseline artifacts")
    return PreparedExample(problem_id, baseline_id, tool_calls=5)


__all__ = ["PreparedExample", "example_catalog", "prepare_example"]

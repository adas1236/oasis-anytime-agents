"""Experiment runner for the repository's small, mock geospatial datasets.

This module deliberately keeps the data contract lightweight.  The selected
dataset filename determines the problem family; source records do not need a
problem-type, units, license, or version field.  Locations are still addressed
by their human-readable names, with normalized internal IDs used only to make
lookups unambiguous.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import itertools
import json
import math
import os
import random
import re
import statistics
import sys
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

EARTH_RADIUS_KM = 6_371.0088
DATASET_FILES = {
    "max_coverage": "max_coverage.json",
    "minimum_facility": "minimum_facility.json",
    "tsp": "tsp.json",
}


class DatasetKind(StrEnum):
    MAX_COVERAGE = "max_coverage"
    MINIMUM_FACILITY = "minimum_facility"
    TSP = "tsp"


@dataclass(frozen=True, slots=True)
class Location:
    location_id: str
    name: str
    latitude: float
    longitude: float
    population: int | None = None


@dataclass(frozen=True, slots=True)
class MockCase:
    dataset: DatasetKind
    source_index: int
    region: str
    prompt: str
    locations: tuple[Location, ...]
    answer: Any
    centers_to_place: int | None = None
    coverage_radius_km: float | None = None
    coverage_target_percent: float | None = None

    @property
    def record_id(self) -> str:
        return f"{self.dataset.value}:{self.source_index}"


@dataclass(slots=True)
class ExperimentConfig:
    dataset: str
    data_root: Path
    model_type: str
    model: str | None
    profile: str
    revision: str | None
    gpus: str
    dtype: str
    quantization: str
    attention_backend: str
    trust_remote_code: bool
    thinking: bool
    max_generated_tokens: int
    max_tool_rounds: int
    case_timeout_seconds: float | None
    start: int
    limit: int | None
    shuffle: bool
    seed: int
    expected_selection_digest: str | None
    regions: tuple[str, ...]
    output: Path | None
    overwrite: bool
    quiet: bool
    fail_on_error: bool
    osrm_endpoint: str
    osrm_cache: Path
    osrm_cache_only: bool
    osrm_timeout_seconds: float
    tsp_tolerance_km: float
    time_budgets_seconds: tuple[float | None, ...] = (None,)
    token_budgets: tuple[int | None, ...] = (None,)
    max_tool_calls: int | None = None
    resume: bool = False


@dataclass(frozen=True, slots=True)
class BudgetPoint:
    """One cell in the Cartesian anytime-evaluation budget grid.

    ``None`` is deliberately used for an unlimited dimension.  A zero budget
    is also valid and evaluates only the deterministic feasible baseline.
    """

    budget_id: str
    wall_time_seconds: float | None
    max_total_model_tokens: int | None
    max_tool_calls: int | None

    def payload(self) -> dict[str, Any]:
        return {
            "wall_time_seconds": self.wall_time_seconds,
            "max_total_model_tokens": self.max_total_model_tokens,
            "max_tool_calls": self.max_tool_calls,
        }


@dataclass(slots=True)
class AgentRun:
    answer_text: str | None = None
    prediction: dict[str, Any] | None = None
    baseline_prediction: dict[str, Any] | None = None
    prediction_source: str = "baseline"
    calls: list[dict[str, Any]] = field(default_factory=list)
    incumbent_timeline: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    generations: int = 0
    tool_calls_used: int = 0
    usage_complete: bool = True
    budget_elapsed_seconds: float = 0.0
    stop_reason: str | None = None
    terminal_reason: str = "error"
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class _BudgetClock:
    """Aggregate wall/token/tool accounting for a single case and budget cell."""

    def __init__(self, budget: BudgetPoint) -> None:
        self.budget = budget
        self.started = time.monotonic()
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.generations = 0
        self.tool_calls = 0

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started)

    @property
    def remaining_seconds(self) -> float | None:
        if self.budget.wall_time_seconds is None:
            return None
        return max(0.0, self.budget.wall_time_seconds - self.elapsed_seconds)

    @property
    def remaining_tokens(self) -> int | None:
        if self.budget.max_total_model_tokens is None:
            return None
        return max(
            0,
            self.budget.max_total_model_tokens - self.input_tokens - self.output_tokens,
        )

    @property
    def time_exhausted(self) -> bool:
        remaining = self.remaining_seconds
        return remaining is not None and remaining <= 0

    @property
    def tokens_exhausted(self) -> bool:
        remaining = self.remaining_tokens
        return remaining is not None and remaining <= 0

    @property
    def tools_exhausted(self) -> bool:
        maximum = self.budget.max_tool_calls
        return maximum is not None and self.tool_calls >= maximum

    def generation_allowance(self, estimated_input_tokens: int, per_call_cap: int) -> int:
        remaining = self.remaining_tokens
        if remaining is None:
            return per_call_cap
        return max(0, min(per_call_cap, remaining - estimated_input_tokens))

    def record_turn(self, usage: Any) -> None:
        self.input_tokens += int(usage.input_tokens)
        self.output_tokens += int(usage.generated_tokens)
        self.reasoning_tokens += int(usage.reasoning_tokens)
        self.generations += 1

    def sync_run(self, run: AgentRun) -> None:
        run.input_tokens = self.input_tokens
        run.output_tokens = self.output_tokens
        run.reasoning_tokens = self.reasoning_tokens
        run.generations = self.generations
        run.tool_calls_used = self.tool_calls

    def consumed_payload(self) -> dict[str, Any]:
        return {
            "wall_time_seconds": round(self.elapsed_seconds, 6),
            "input_tokens": self.input_tokens,
            "generated_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_model_tokens": self.input_tokens + self.output_tokens,
            "model_generations": self.generations,
            "tool_calls": self.tool_calls,
        }


def _normalize_name(value: str) -> str:
    ascii_text = "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.casefold()).strip()


def _location_id(region: str, name: str) -> str:
    region_part = _normalize_name(region).replace(" ", "-") or "region"
    name_part = _normalize_name(name).replace(" ", "-") or "location"
    return f"{region_part}:{name_part}"


def _require_number(record: dict[str, Any], key: str, source: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: {key!r} must be a number")
    return float(value)


def _parse_location(raw: Any, region: str, source: str) -> Location:
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: each location must be an object")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{source}: each location needs a non-empty name")
    latitude = _require_number(raw, "latitude", source)
    longitude = _require_number(raw, "longitude", source)
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError(f"{source}: coordinates for {name!r} are out of range")
    population_value = raw.get("population")
    population: int | None
    if population_value is None:
        population = None
    elif isinstance(population_value, bool) or not isinstance(population_value, int):
        raise ValueError(f"{source}: population for {name!r} must be an integer")
    else:
        population = population_value
    return Location(
        location_id=_location_id(region, name),
        name=name,
        latitude=latitude,
        longitude=longitude,
        population=population,
    )


def load_dataset(kind: DatasetKind, path: Path) -> list[MockCase]:
    """Load one mock dataset, inferring its schema from ``kind``."""

    try:
        raw_records = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset is not valid JSON: {path}: {exc}") from exc
    if not isinstance(raw_records, list):
        raise ValueError(f"{path}: top-level JSON value must be a list")

    cases: list[MockCase] = []
    for index, raw in enumerate(raw_records):
        source = f"{path}[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{source}: record must be an object")
        prompt = raw.get("prompt")
        region = raw.get("geographic_region")
        locations_raw = raw.get("locations")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{source}: prompt must be a non-empty string")
        if not isinstance(region, str) or not region.strip():
            raise ValueError(f"{source}: region must be a non-empty string")
        if not isinstance(locations_raw, list) or not locations_raw:
            raise ValueError(f"{source}: locations must be a non-empty list")
        locations = tuple(_parse_location(location, region, source) for location in locations_raw)
        normalized_names = [_normalize_name(location.name) for location in locations]
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError(f"{source}: location names must be unique within a record")
        if "answer" not in raw:
            raise ValueError(f"{source}: answer is required for scoring")

        centers_to_place: int | None = None
        coverage_radius_km: float | None = None
        coverage_target_percent: float | None = None
        if kind is DatasetKind.MAX_COVERAGE:
            centers_value = raw.get("centers_to_place")
            if isinstance(centers_value, bool) or not isinstance(centers_value, int):
                raise ValueError(f"{source}: centers_to_place must be an integer")
            centers_to_place = centers_value
            coverage_radius_km = _require_number(raw, "coverage_radius_km", source)
        elif kind is DatasetKind.MINIMUM_FACILITY:
            coverage_target_percent = _require_number(raw, "coverage_target_percent", source)
            coverage_radius_km = _require_number(raw, "coverage_radius_km", source)

        if kind is not DatasetKind.TSP and any(
            location.population is None for location in locations
        ):
            raise ValueError(f"{source}: coverage locations require population")

        cases.append(
            MockCase(
                dataset=kind,
                source_index=index,
                region=region,
                prompt=prompt,
                locations=locations,
                answer=raw["answer"],
                centers_to_place=centers_to_place,
                coverage_radius_km=coverage_radius_km,
                coverage_target_percent=coverage_target_percent,
            )
        )
    return cases


class LocationIndex:
    """Name-first lookup for one prompt's location set."""

    def __init__(self, locations: Sequence[Location]) -> None:
        self.locations = tuple(locations)
        self._by_normalized_name = {
            _normalize_name(location.name): location for location in self.locations
        }
        self._by_id = {location.location_id: location for location in self.locations}

    def resolve(self, query: str) -> Location:
        normalized = _normalize_name(query)
        if query in self._by_id:
            return self._by_id[query]
        exact = self._by_normalized_name.get(normalized)
        if exact is not None:
            return exact
        matches = self.search(query, limit=2)
        if not matches or matches[0][0] < 0.72:
            raise ValueError(f"No location in this problem matches {query!r}")
        if len(matches) > 1 and abs(matches[0][0] - matches[1][0]) < 0.03:
            names = ", ".join(match[1].name for match in matches)
            raise ValueError(f"Ambiguous location {query!r}; closest matches: {names}")
        return matches[0][1]

    def resolve_many(self, queries: Any) -> tuple[Location, ...]:
        if not isinstance(queries, list) or not queries:
            raise ValueError("location_names must be a non-empty array of strings")
        resolved: list[Location] = []
        seen: set[str] = set()
        for query in queries:
            if not isinstance(query, str):
                raise ValueError("Every location name must be a string")
            location = self.resolve(query)
            if location.location_id in seen:
                raise ValueError(f"Location {location.name!r} was supplied more than once")
            seen.add(location.location_id)
            resolved.append(location)
        return tuple(resolved)

    def search(self, query: str, *, limit: int = 5) -> list[tuple[float, Location]]:
        normalized = _normalize_name(query)
        if not normalized:
            return []
        ranked: list[tuple[float, Location]] = []
        for name, location in self._by_normalized_name.items():
            if name == normalized:
                score = 1.0
            elif normalized in name or name in normalized:
                score = 0.9
            else:
                score = difflib.SequenceMatcher(None, normalized, name).ratio()
            if score >= 0.35:
                ranked.append((score, location))
        ranked.sort(key=lambda item: (-item[0], item[1].name))
        return ranked[:limit]


def haversine_km(left: Location, right: Location) -> float:
    latitude_1 = math.radians(left.latitude)
    latitude_2 = math.radians(right.latitude)
    latitude_delta = latitude_2 - latitude_1
    longitude_delta = math.radians(right.longitude - left.longitude)
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude_1) * math.cos(latitude_2) * math.sin(longitude_delta / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(haversine))


def _coverage_masks(locations: Sequence[Location], radius_km: float) -> tuple[list[int], list[int]]:
    populations = [location.population or 0 for location in locations]
    masks: list[int] = []
    for center in locations:
        mask = 0
        for index, demand in enumerate(locations):
            if haversine_km(center, demand) <= radius_km + 1e-9:
                mask |= 1 << index
        masks.append(mask)
    return masks, populations


def _population_for_mask(mask: int, populations: Sequence[int]) -> int:
    return sum(population for index, population in enumerate(populations) if mask & (1 << index))


def solve_max_coverage(
    locations: Sequence[Location], centers_to_place: int, radius_km: float
) -> dict[str, Any]:
    if centers_to_place < 1 or centers_to_place > len(locations):
        raise ValueError("centers_to_place must be between 1 and the number of locations")
    if radius_km < 0:
        raise ValueError("coverage_radius_km cannot be negative")
    masks, populations = _coverage_masks(locations, radius_km)
    best_population = -1
    best_combinations: list[tuple[int, ...]] = []
    for combination in itertools.combinations(range(len(locations)), centers_to_place):
        covered_mask = 0
        for center_index in combination:
            covered_mask |= masks[center_index]
        population = _population_for_mask(covered_mask, populations)
        if population > best_population:
            best_population = population
            best_combinations = [combination]
        elif population == best_population:
            best_combinations.append(combination)
    selected = best_combinations[0]
    return {
        "problem": DatasetKind.MAX_COVERAGE.value,
        "people_covered": best_population,
        "center_locations": [locations[index].name for index in selected],
        "optimal_solution_count": len(best_combinations),
    }


def solve_minimum_facility(
    locations: Sequence[Location], target_percent: float, radius_km: float
) -> dict[str, Any]:
    if not 0 < target_percent <= 100:
        raise ValueError("coverage_target_percent must be in (0, 100]")
    if radius_km < 0:
        raise ValueError("coverage_radius_km cannot be negative")
    masks, populations = _coverage_masks(locations, radius_km)
    total_population = sum(populations)
    target_population = math.ceil(total_population * target_percent / 100)
    best_combinations: list[tuple[int, ...]] = []
    people_covered = 0
    for center_count in range(1, len(locations) + 1):
        for combination in itertools.combinations(range(len(locations)), center_count):
            covered_mask = 0
            for center_index in combination:
                covered_mask |= masks[center_index]
            population = _population_for_mask(covered_mask, populations)
            if population >= target_population:
                if not best_combinations:
                    people_covered = population
                best_combinations.append(combination)
        if best_combinations:
            break
    if not best_combinations:
        raise RuntimeError("No facility set can reach the requested target")
    selected = best_combinations[0]
    return {
        "problem": DatasetKind.MINIMUM_FACILITY.value,
        "minimum_centers": len(selected),
        "center_locations": [locations[index].name for index in selected],
        "people_covered": people_covered,
        "target_population": target_population,
        "optimal_solution_count": len(best_combinations),
    }


def _coverage_candidate(
    kind: DatasetKind,
    locations: Sequence[Location],
    selected_indexes: Sequence[int],
    radius_km: float,
    *,
    target_percent: float | None = None,
) -> dict[str, Any]:
    """Evaluate a feasible, not necessarily optimal, coverage incumbent."""

    masks, populations = _coverage_masks(locations, radius_km)
    covered_mask = 0
    for center_index in selected_indexes:
        covered_mask |= masks[center_index]
    people_covered = _population_for_mask(covered_mask, populations)
    result: dict[str, Any] = {
        "problem": kind.value,
        "center_locations": [locations[index].name for index in selected_indexes],
        "people_covered": people_covered,
    }
    if kind is DatasetKind.MAX_COVERAGE:
        result["centers_placed"] = len(selected_indexes)
    else:
        if target_percent is None:
            raise ValueError("target_percent is required for minimum-facility candidates")
        total_population = sum(populations)
        target_population = math.ceil(total_population * target_percent / 100)
        result.update(
            minimum_centers=len(selected_indexes),
            target_population=target_population,
            target_reached=people_covered >= target_population,
        )
    return result


class OsrmMatrixStore:
    """Fetch and cache one driving-distance matrix for each ten-place region."""

    def __init__(
        self,
        *,
        endpoint: str,
        cache_dir: Path,
        cache_only: bool,
        timeout_seconds: float,
        region_locations: dict[str, tuple[Location, ...]],
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.cache_dir = cache_dir
        self.cache_only = cache_only
        self.timeout_seconds = timeout_seconds
        self.region_locations = region_locations
        self._memory: dict[str, list[list[float]]] = {}

    def _cache_path(self, region: str) -> Path:
        slug = _normalize_name(region).replace(" ", "-") or "region"
        endpoint_key = hashlib.sha256(self.endpoint.encode("utf-8")).hexdigest()[:8]
        return self.cache_dir / f"{slug}-{endpoint_key}.json"

    @staticmethod
    def _signature(locations: Sequence[Location]) -> list[dict[str, Any]]:
        return [
            {
                "name": location.name,
                "latitude": location.latitude,
                "longitude": location.longitude,
            }
            for location in locations
        ]

    async def _region_matrix(self, region: str) -> list[list[float]]:
        if region in self._memory:
            return self._memory[region]
        master_locations = self.region_locations.get(region)
        if not master_locations:
            raise ValueError(f"No TSP location catalog is available for region {region!r}")
        cache_path = self._cache_path(region)
        expected_signature = self._signature(master_locations)
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("locations") == expected_signature:
                    matrix = self._validate_matrix(
                        cached.get("distances_m"), len(master_locations), str(cache_path)
                    )
                    self._memory[region] = matrix
                    return matrix
            except (json.JSONDecodeError, OSError, ValueError):
                if self.cache_only:
                    raise
        if self.cache_only:
            raise FileNotFoundError(f"No usable cached OSRM matrix for {region!r} at {cache_path}")

        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required to query OSRM") from exc
        coordinates = ";".join(
            f"{location.longitude},{location.latitude}" for location in master_locations
        )
        url = f"{self.endpoint}/table/v1/driving/{coordinates}"
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(url, params={"annotations": "distance"})
            response.raise_for_status()
            payload = response.json()
        if payload.get("code") != "Ok":
            raise RuntimeError(f"OSRM table request failed: {payload.get('message', payload)}")
        matrix = self._validate_matrix(
            payload.get("distances"), len(master_locations), "OSRM response"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"locations": expected_signature, "distances_m": matrix},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._memory[region] = matrix
        return matrix

    @staticmethod
    def _validate_matrix(raw: Any, size: int, source: str) -> list[list[float]]:
        if not isinstance(raw, list) or len(raw) != size:
            raise ValueError(f"{source}: distance matrix has the wrong number of rows")
        matrix: list[list[float]] = []
        for row in raw:
            if not isinstance(row, list) or len(row) != size:
                raise ValueError(f"{source}: distance matrix is not square")
            parsed_row: list[float] = []
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{source}: distance matrix contains an invalid value")
                parsed_row.append(float(value))
            matrix.append(parsed_row)
        return matrix

    async def solve(self, region: str, locations: Sequence[Location]) -> dict[str, Any]:
        if len(locations) < 2:
            raise ValueError("A TSP problem needs at least two locations")
        master_locations = self.region_locations[region]
        by_name = {
            _normalize_name(location.name): index for index, location in enumerate(master_locations)
        }
        indexes: list[int] = []
        for location in locations:
            try:
                indexes.append(by_name[_normalize_name(location.name)])
            except KeyError as exc:
                raise ValueError(
                    f"Location {location.name!r} is not in the regional OSRM matrix"
                ) from exc
        matrix = await self._region_matrix(region)
        depot = indexes[0]
        best_distance_m = math.inf
        best_order: tuple[int, ...] | None = None
        for middle_order in itertools.permutations(indexes[1:]):
            order = (depot, *middle_order, depot)
            distance_m = sum(matrix[left][right] for left, right in itertools.pairwise(order))
            if distance_m < best_distance_m:
                best_distance_m = distance_m
                best_order = order
        assert best_order is not None
        return {
            "problem": DatasetKind.TSP.value,
            "distance_km": best_distance_m / 1_000,
            "rounded_distance_km": round(best_distance_m / 1_000),
            "route": [master_locations[index].name for index in best_order],
        }

    async def prepare_region(self, region: str) -> None:
        """Load a regional matrix before per-cell budget clocks begin."""

        await self._region_matrix(region)

    async def evaluate_route(self, region: str, locations: Sequence[Location]) -> dict[str, Any]:
        """Evaluate the prompt-order round trip used as the TSP baseline."""

        if len(locations) < 2:
            raise ValueError("A TSP problem needs at least two locations")
        master_locations = self.region_locations[region]
        by_name = {
            _normalize_name(location.name): index for index, location in enumerate(master_locations)
        }
        indexes = [by_name[_normalize_name(location.name)] for location in locations]
        order = (*indexes, indexes[0])
        matrix = await self._region_matrix(region)
        distance_m = sum(matrix[left][right] for left, right in itertools.pairwise(order))
        return {
            "problem": DatasetKind.TSP.value,
            "distance_km": distance_m / 1_000,
            "rounded_distance_km": round(distance_m / 1_000),
            "route": [master_locations[index].name for index in order],
        }


def _build_region_catalog(cases: Sequence[MockCase]) -> dict[str, tuple[Location, ...]]:
    by_region: dict[str, dict[str, Location]] = {}
    for case in cases:
        region = by_region.setdefault(case.region, {})
        for location in case.locations:
            key = _normalize_name(location.name)
            existing = region.get(key)
            if existing is not None and (
                existing.latitude != location.latitude or existing.longitude != location.longitude
            ):
                raise ValueError(
                    f"Coordinates for {location.name!r} vary within region {case.region!r}"
                )
            region.setdefault(key, location)
    return {region: tuple(locations.values()) for region, locations in by_region.items()}


async def _baseline_prediction(case: MockCase, osrm: OsrmMatrixStore | None) -> dict[str, Any]:
    """Return a cheap feasible plan available even when either budget is zero."""

    if case.dataset is DatasetKind.MAX_COVERAGE:
        center_count = case.centers_to_place or 0
        return _coverage_candidate(
            case.dataset,
            case.locations,
            tuple(range(center_count)),
            case.coverage_radius_km or 0,
        )
    if case.dataset is DatasetKind.MINIMUM_FACILITY:
        return _coverage_candidate(
            case.dataset,
            case.locations,
            tuple(range(len(case.locations))),
            case.coverage_radius_km or 0,
            target_percent=case.coverage_target_percent,
        )
    if osrm is None:
        raise RuntimeError("OSRM matrix support was not initialized")
    return await osrm.evaluate_route(case.region, case.locations)


def _incumbent_event(
    *,
    clock: _BudgetClock,
    source: str,
    prediction: dict[str, Any],
    case: MockCase,
    tolerance: float,
) -> dict[str, Any]:
    return {
        "elapsed_seconds": round(clock.elapsed_seconds, 6),
        "total_model_tokens": clock.input_tokens + clock.output_tokens,
        "tool_calls": clock.tool_calls,
        "source": source,
        "correct": _score(case, prediction, tolerance),
        "prediction": prediction,
    }


async def _wait_with_limits(
    awaitable: Any,
    clock: _BudgetClock,
    per_call_timeout_seconds: float | None = None,
) -> Any:
    timeouts = [
        timeout
        for timeout in (clock.remaining_seconds, per_call_timeout_seconds)
        if timeout is not None
    ]
    if not timeouts:
        return await awaitable
    timeout = min(timeouts)
    if timeout <= 0:
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()
        raise TimeoutError
    return await asyncio.wait_for(awaitable, timeout=timeout)


def _solver_arguments(case: MockCase) -> dict[str, Any]:
    arguments: dict[str, Any] = {"location_names": [location.name for location in case.locations]}
    if case.dataset is DatasetKind.MAX_COVERAGE:
        arguments.update(
            centers_to_place=case.centers_to_place,
            coverage_radius_km=case.coverage_radius_km,
        )
    elif case.dataset is DatasetKind.MINIMUM_FACILITY:
        arguments.update(
            coverage_target_percent=case.coverage_target_percent,
            coverage_radius_km=case.coverage_radius_km,
        )
    return arguments


def _solver_schema(kind: DatasetKind) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "location_names": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "All place names from the user's problem, in prompt order.",
        }
    }
    required = ["location_names"]
    if kind is DatasetKind.MAX_COVERAGE:
        properties.update(
            centers_to_place={"type": "integer", "minimum": 1},
            coverage_radius_km={"type": "number", "minimum": 0},
        )
        required.extend(["centers_to_place", "coverage_radius_km"])
        description = "Solve the current maximum population coverage problem exactly."
    elif kind is DatasetKind.MINIMUM_FACILITY:
        properties.update(
            coverage_target_percent={
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 100,
            },
            coverage_radius_km={"type": "number", "minimum": 0},
        )
        required.extend(["coverage_target_percent", "coverage_radius_km"])
        description = "Solve the current minimum facility coverage problem exactly."
    else:
        description = (
            "Solve the current round-trip driving-distance problem exactly. "
            "The first location is the depot."
        )
    return {
        "name": "solve_current_problem",
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def _tool_definitions(kind: DatasetKind) -> list[dict[str, Any]]:
    return [
        {
            "name": "search_locations",
            "description": (
                "Resolve plaintext place names in the current problem to coordinates "
                "and, when available, population."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    }
                },
                "required": ["queries"],
                "additionalProperties": False,
            },
        },
        _solver_schema(kind),
    ]


async def _dispatch_tool(
    case: MockCase,
    name: str,
    arguments: dict[str, Any],
    osrm: OsrmMatrixStore | None,
) -> dict[str, Any]:
    index = LocationIndex(case.locations)
    if name == "search_locations":
        queries = arguments.get("queries")
        if not isinstance(queries, list) or not queries:
            raise ValueError("queries must be a non-empty array of strings")
        results: list[dict[str, Any]] = []
        for query in queries:
            if not isinstance(query, str):
                raise ValueError("Every search query must be a string")
            matches = []
            for score, location in index.search(query):
                match: dict[str, Any] = {
                    "location_id": location.location_id,
                    "name": location.name,
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                    "match_score": round(score, 3),
                }
                if location.population is not None:
                    match["population"] = location.population
                matches.append(match)
            results.append({"query": query, "matches": matches})
        return {"results": results}
    if name != "solve_current_problem":
        raise ValueError(f"Unknown tool {name!r}")

    locations = index.resolve_many(arguments.get("location_names"))
    expected_ids = {location.location_id for location in case.locations}
    supplied_ids = {location.location_id for location in locations}
    if supplied_ids != expected_ids:
        raise ValueError("location_names must include every location in the current problem")
    if case.dataset is DatasetKind.MAX_COVERAGE:
        centers = arguments.get("centers_to_place")
        radius = arguments.get("coverage_radius_km")
        if isinstance(centers, bool) or not isinstance(centers, int):
            raise ValueError("centers_to_place must be an integer")
        if isinstance(radius, bool) or not isinstance(radius, (int, float)):
            raise ValueError("coverage_radius_km must be a number")
        if centers != case.centers_to_place or not math.isclose(
            float(radius), case.coverage_radius_km or 0, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("solver parameters do not match the current problem")
        return solve_max_coverage(locations, centers, float(radius))
    if case.dataset is DatasetKind.MINIMUM_FACILITY:
        target = arguments.get("coverage_target_percent")
        radius = arguments.get("coverage_radius_km")
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise ValueError("coverage_target_percent must be a number")
        if isinstance(radius, bool) or not isinstance(radius, (int, float)):
            raise ValueError("coverage_radius_km must be a number")
        if not math.isclose(
            float(target), case.coverage_target_percent or 0, rel_tol=0, abs_tol=1e-9
        ) or not math.isclose(float(radius), case.coverage_radius_km or 0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("solver parameters do not match the current problem")
        return solve_minimum_facility(locations, float(target), float(radius))
    if osrm is None:
        raise RuntimeError("OSRM matrix support was not initialized")
    if locations[0].location_id != case.locations[0].location_id:
        raise ValueError("the first location must remain the round-trip depot")
    return await osrm.solve(case.region, locations)


def _system_prompt() -> str:
    return (
        "You are solving a small geospatial public-health planning exercise. "
        "Infer what the user is asking from their natural-language prompt. Use "
        "search_locations when name resolution or source attributes are useful, "
        "then call solve_current_problem with every place named in the problem and "
        "the requested parameters. Coordinates are latitude/longitude degrees and "
        "all distances are kilometers. Do not guess a result that the solver can "
        "compute. After the solver succeeds, give a brief answer based on its result."
    )


async def _run_agent_case(
    case: MockCase,
    backend: Any,
    config: ExperimentConfig,
    osrm: OsrmMatrixStore | None,
    budget: BudgetPoint | None = None,
) -> AgentRun:
    # These imports avoid importing model runtimes until the wrapper has applied
    # CUDA_VISIBLE_DEVICES from --gpus.
    from oasis.llm.schemas import ChatMessage, ChatRole, ModelRequest, ToolDefinition

    budget = budget or BudgetPoint(
        budget_id="time-unlimited_tokens-unlimited",
        wall_time_seconds=None,
        max_total_model_tokens=None,
        max_tool_calls=config.max_tool_calls,
    )
    tool_definitions = tuple(
        ToolDefinition(
            name=definition["name"],
            description=definition["description"],
            input_schema=definition["input_schema"],
        )
        for definition in _tool_definitions(case.dataset)
    )
    messages = [
        ChatMessage(role=ChatRole.SYSTEM, content=_system_prompt()),
        ChatMessage(role=ChatRole.USER, content=case.prompt),
    ]
    # Baseline construction and static TSP evidence loading are setup. The
    # budgeted clock starts only after a feasible incumbent has been committed.
    baseline = await _baseline_prediction(case, osrm)
    run = AgentRun(
        prediction=baseline,
        baseline_prediction=baseline,
        prediction_source="baseline",
    )
    clock = _BudgetClock(budget)
    run.incumbent_timeline.append(
        _incumbent_event(
            clock=clock,
            source="baseline",
            prediction=baseline,
            case=case,
            tolerance=config.tsp_tolerance_km,
        )
    )

    def finish(reason: str, *, error: str | None = None) -> AgentRun:
        clock.sync_run(run)
        run.budget_elapsed_seconds = round(clock.elapsed_seconds, 6)
        run.terminal_reason = reason
        if error is not None:
            run.error = error
        return run

    for round_index in range(config.max_tool_rounds + 1):
        if clock.time_exhausted:
            return finish("time_budget_exhausted")
        if clock.tokens_exhausted:
            return finish("token_budget_exhausted")

        request_id = f"mock-{case.record_id}-{budget.budget_id}-{round_index}"
        provisional_request = ModelRequest(
            request_id=request_id,
            messages=tuple(messages),
            max_generated_tokens=config.max_generated_tokens,
            thinking_enabled=config.thinking,
            tools=tool_definitions,
            seed=config.seed + case.source_index,
        )
        try:
            estimated_input_tokens = await _wait_with_limits(
                backend.count_input_tokens(provisional_request),
                clock,
                config.case_timeout_seconds,
            )
        except TimeoutError:
            reason = "time_budget_exhausted" if clock.time_exhausted else "model_call_timeout"
            return finish(reason)

        allowance = clock.generation_allowance(
            int(estimated_input_tokens), config.max_generated_tokens
        )
        context_limit = getattr(getattr(backend, "capabilities", None), "context_limit", None)
        if isinstance(context_limit, int):
            allowance = min(allowance, max(0, context_limit - int(estimated_input_tokens)))
        if allowance < 1:
            reason = (
                "context_limit_exhausted"
                if isinstance(context_limit, int) and estimated_input_tokens >= context_limit
                else "token_budget_exhausted"
            )
            return finish(reason)

        request = provisional_request.model_copy(update={"max_generated_tokens": allowance})
        try:
            turn = await _wait_with_limits(
                backend.generate(request), clock, config.case_timeout_seconds
            )
        except TimeoutError:
            # The pre-counted input was consumed, but a cancelled backend cannot
            # always report how many output tokens it emitted before cancellation.
            clock.input_tokens += int(estimated_input_tokens)
            run.usage_complete = False
            try:
                await backend.abort(request_id)
            except Exception:
                pass
            reason = "time_budget_exhausted" if clock.time_exhausted else "model_call_timeout"
            return finish(reason)
        clock.record_turn(turn.usage)
        if (
            budget.max_total_model_tokens is not None
            and clock.input_tokens + clock.output_tokens > budget.max_total_model_tokens
        ):
            run.usage_complete = False
            return finish(
                "token_budget_overrun",
                error="model reported usage beyond the aggregate token budget",
            )
        run.stop_reason = turn.finish_reason.value
        messages.append(turn.message)
        if not turn.message.tool_calls:
            run.answer_text = turn.message.content
            reason = "completed" if run.prediction_source != "baseline" else "model_stopped"
            return finish(reason)
        if round_index >= config.max_tool_rounds:
            return finish("max_tool_rounds_reached")
        for call in turn.message.tool_calls:
            call_record: dict[str, Any] = {
                "tool_call_id": call.id,
                "round_index": round_index,
                "name": call.name,
                "arguments": call.arguments,
            }
            remaining_tool_time = clock.remaining_seconds
            if remaining_tool_time is not None and remaining_tool_time <= 0:
                call_record["response"] = {
                    "ok": False,
                    "error": "time budget exhausted before tool execution",
                }
                run.calls.append(call_record)
                return finish("time_budget_exhausted")
            if clock.tools_exhausted:
                call_record["response"] = {
                    "ok": False,
                    "error": "tool-call budget exhausted",
                }
                run.calls.append(call_record)
                return finish("tool_call_budget_exhausted")
            clock.tool_calls += 1
            try:
                result = await _wait_with_limits(
                    _dispatch_tool(case, call.name, call.arguments, osrm), clock
                )
                remaining_after_tool = clock.remaining_seconds
                if remaining_after_tool is not None and remaining_after_tool <= 0:
                    call_record["response"] = {
                        "ok": False,
                        "error": "tool result arrived after the time budget",
                    }
                    run.calls.append(call_record)
                    return finish("time_budget_exhausted")
                envelope = {"ok": True, "result": result}
                if call.name == "solve_current_problem":
                    run.prediction = result
                    run.prediction_source = "solve_current_problem"
                    run.error = None
                    run.incumbent_timeline.append(
                        _incumbent_event(
                            clock=clock,
                            source="solve_current_problem",
                            prediction=result,
                            case=case,
                            tolerance=config.tsp_tolerance_km,
                        )
                    )
            except TimeoutError:
                envelope = {"ok": False, "error": "time budget exhausted during tool call"}
                call_record["response"] = envelope
                run.calls.append(call_record)
                return finish("time_budget_exhausted")
            except Exception as exc:  # tool errors are observations for the agent
                error_text = f"{type(exc).__name__}: {exc}"
                envelope = {"ok": False, "error": error_text}
                if call.name == "solve_current_problem":
                    run.error = error_text
            call_record["response"] = envelope
            run.calls.append(call_record)
            messages.append(
                ChatMessage(
                    role=ChatRole.TOOL,
                    content=json.dumps(envelope, ensure_ascii=False),
                    tool_call_id=call.id,
                    name=call.name,
                )
            )
    return finish("max_tool_rounds_reached")


def _score(case: MockCase, prediction: dict[str, Any] | None, tolerance: float) -> bool:
    if prediction is None:
        return False
    if case.dataset is DatasetKind.MAX_COVERAGE:
        expected = case.answer
        if not isinstance(expected, dict):
            return False
        if prediction.get("people_covered") != expected.get("people_covered"):
            return False
        selected = set(prediction.get("center_locations", []))
        solutions = expected.get("optimal_solutions", [])
        return any(selected == set(solution.get("center_locations", [])) for solution in solutions)
    if case.dataset is DatasetKind.MINIMUM_FACILITY:
        expected = case.answer
        if not isinstance(expected, dict):
            return False
        if prediction.get("minimum_centers") != expected.get("minimum_centers"):
            return False
        selected = set(prediction.get("center_locations", []))
        solutions = expected.get("optimal_solutions", [])
        return any(selected == set(solution.get("center_locations", [])) for solution in solutions)
    expected_distance = case.answer
    predicted_distance = prediction.get("distance_km")
    if isinstance(expected_distance, bool) or not isinstance(expected_distance, (int, float)):
        return False
    if isinstance(predicted_distance, bool) or not isinstance(predicted_distance, (int, float)):
        return False
    return abs(float(predicted_distance) - float(expected_distance)) <= tolerance


def _expected_summary(case: MockCase) -> dict[str, Any]:
    if case.dataset is DatasetKind.MAX_COVERAGE and isinstance(case.answer, dict):
        return {
            "people_covered": case.answer.get("people_covered"),
            "solution_count": len(case.answer.get("optimal_solutions", [])),
        }
    if case.dataset is DatasetKind.MINIMUM_FACILITY and isinstance(case.answer, dict):
        return {
            "minimum_centers": case.answer.get("minimum_centers"),
            "solution_count": len(case.answer.get("optimal_solutions", [])),
        }
    return {"distance_km": case.answer}


def _fake_backend(case: MockCase) -> Any:
    from oasis.llm.fake import FakeModelBackend
    from oasis.llm.schemas import ToolCall

    return FakeModelBackend(
        responses=[
            ToolCall(
                id=f"solve-{case.source_index}",
                name="solve_current_problem",
                arguments=_solver_arguments(case),
            ),
            "I used the solver result as the answer.",
        ]
    )


async def _transformers_backend(config: ExperimentConfig) -> Any:
    from oasis.config import (
        BackendKind,
        DevicePolicy,
        OasisSettings,
        RuntimeEngine,
    )
    from oasis.llm.factory import create_model_backend
    from oasis.runtimes import inspect_cuda_inventory

    device = (
        DevicePolicy.CPU
        if config.gpus == "none"
        else DevicePolicy.AUTO
        if config.gpus == "auto"
        else DevicePolicy.CUDA
    )
    inventory = inspect_cuda_inventory() if device is not DevicePolicy.CPU else None
    settings = OasisSettings.resolve(
        explicit_overrides={
            "backend": BackendKind.TRANSFORMERS,
            "model_profile": config.profile,
            "model_id": config.model,
            "model_revision": config.revision,
            "device": device,
            "runtime_engine": RuntimeEngine.AUTO,
            "dtype": config.dtype,
            "quantization": None if config.quantization == "none" else config.quantization,
            "attention_backend": config.attention_backend,
            "trust_remote_code": config.trust_remote_code,
        }
    )
    backend = create_model_backend(settings, inventory=inventory)
    await backend.load()
    return backend


def _select_cases(
    config: ExperimentConfig,
) -> tuple[list[MockCase], list[MockCase]]:
    requested = tuple(DatasetKind) if config.dataset == "all" else (DatasetKind(config.dataset),)
    all_loaded: list[MockCase] = []
    selected: list[MockCase] = []
    for kind in requested:
        cases = load_dataset(kind, config.data_root / DATASET_FILES[kind.value])
        all_loaded.extend(cases)
        selected.extend(cases)
    if config.regions:
        wanted = {region.casefold() for region in config.regions}
        selected = [case for case in selected if case.region.casefold() in wanted]
    if config.shuffle:
        random.Random(config.seed).shuffle(selected)
    stop = None if config.limit is None else config.start + config.limit
    selected = selected[config.start : stop]
    if not selected:
        raise ValueError("No records matched the requested dataset slice")
    actual_digest = selection_digest(selected)
    if (
        config.expected_selection_digest is not None
        and actual_digest != config.expected_selection_digest
    ):
        raise ValueError(
            "Selected records do not match --expected-selection-digest: "
            f"expected {config.expected_selection_digest}, got {actual_digest}"
        )
    return selected, all_loaded


def selection_digest(cases: Sequence[MockCase] | Sequence[str]) -> str:
    """Hash an ordered record selection for reproducible distributed evaluation."""

    record_ids = [case if isinstance(case, str) else case.record_id for case in cases]
    encoded = json.dumps(record_ids, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _output_path(config: ExperimentConfig) -> Path:
    if config.output:
        return config.output
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return Path("evaluation-output") / f"mock-{config.dataset}-{timestamp}.jsonl"


_UNLIMITED_VALUES = frozenset({"unlimited", "none", "inf", "infinite"})


def _parse_time_budgets(value: str) -> tuple[float | None, ...]:
    budgets: list[float | None] = []
    multipliers = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3_600.0}
    for raw in value.split(","):
        item = raw.strip().casefold()
        if not item:
            raise ValueError("time budget list contains an empty value")
        if item in _UNLIMITED_VALUES:
            parsed = None
        else:
            match = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|m|h)?", item)
            if match is None:
                raise ValueError(f"invalid time budget {raw!r}; use seconds or a ms/s/m/h suffix")
            parsed = float(match.group(1)) * multipliers.get(match.group(2) or "s", 1.0)
        if parsed not in budgets:
            budgets.append(parsed)
    return tuple(budgets)


def _parse_token_budgets(value: str) -> tuple[int | None, ...]:
    budgets: list[int | None] = []
    for raw in value.split(","):
        item = raw.strip().casefold().replace("_", "")
        if not item:
            raise ValueError("token budget list contains an empty value")
        if item in _UNLIMITED_VALUES:
            parsed = None
        else:
            match = re.fullmatch(r"(\d+)(k|m)?", item)
            if match is None:
                raise ValueError(
                    f"invalid token budget {raw!r}; use an integer, k/m suffix, or unlimited"
                )
            multiplier = {None: 1, "k": 1_000, "m": 1_000_000}[match.group(2)]
            parsed = int(match.group(1)) * multiplier
        if parsed not in budgets:
            budgets.append(parsed)
    return tuple(budgets)


def _budget_label(value: float | int | None, *, time_value: bool = False) -> str:
    if value is None:
        return "unlimited"
    label = f"{value:g}" if isinstance(value, float) else str(value)
    return f"{label}s" if time_value else label


def _budget_grid(config: ExperimentConfig) -> tuple[BudgetPoint, ...]:
    return tuple(
        BudgetPoint(
            budget_id=(
                f"time-{_budget_label(wall_time, time_value=True)}_tokens-{_budget_label(tokens)}"
            ),
            wall_time_seconds=wall_time,
            max_total_model_tokens=tokens,
            max_tool_calls=config.max_tool_calls,
        )
        for wall_time, tokens in itertools.product(
            config.time_budgets_seconds, config.token_budgets
        )
    )


def _config_payload(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "dataset": config.dataset,
        "data_root": str(config.data_root),
        "model_type": config.model_type,
        "model": config.model,
        "profile": config.profile,
        "revision": config.revision,
        "gpus": config.gpus,
        "dtype": config.dtype,
        "quantization": config.quantization,
        "attention_backend": config.attention_backend,
        "trust_remote_code": config.trust_remote_code,
        "thinking": config.thinking,
        "max_generated_tokens": config.max_generated_tokens,
        "max_tool_rounds": config.max_tool_rounds,
        "model_call_timeout_seconds": config.case_timeout_seconds,
        "time_budgets_seconds": list(config.time_budgets_seconds),
        "token_budgets": list(config.token_budgets),
        "max_tool_calls": config.max_tool_calls,
        "start": config.start,
        "limit": config.limit,
        "shuffle": config.shuffle,
        "seed": config.seed,
        "expected_selection_digest": config.expected_selection_digest,
        "regions": list(config.regions),
        "osrm_endpoint": config.osrm_endpoint,
        "osrm_cache": str(config.osrm_cache),
        "osrm_cache_only": config.osrm_cache_only,
        "osrm_timeout_seconds": config.osrm_timeout_seconds,
        "tsp_tolerance_km": config.tsp_tolerance_km,
    }


def _config_fingerprint(config: ExperimentConfig) -> str:
    encoded = json.dumps(_config_payload(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _resume_results(
    output_path: Path, expected_fingerprint: str
) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    if not output_path.exists():
        return [], set()
    raw_data = output_path.read_bytes()
    results: list[dict[str, Any]] = []
    keys: set[tuple[str, str]] = set()
    valid_bytes = 0
    chunks = raw_data.splitlines(keepends=True)
    for index, chunk in enumerate(chunks):
        if not chunk.strip():
            valid_bytes += len(chunk)
            continue
        try:
            result = json.loads(chunk.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index != len(chunks) - 1:
                raise ValueError(f"Malformed JSONL record {index + 1} in {output_path}") from exc
            fragment_path = output_path.with_suffix(output_path.suffix + ".interrupted")
            fragment_path.write_bytes(raw_data[valid_bytes:])
            with output_path.open("r+b") as stream:
                stream.truncate(valid_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            raw_data = raw_data[:valid_bytes]
            break
        if not isinstance(result, dict):
            raise ValueError(f"JSONL record {index + 1} in {output_path} is not an object")
        if result.get("experiment_fingerprint") != expected_fingerprint:
            raise ValueError(f"Cannot resume {output_path}: its experiment configuration differs")
        record_id = result.get("record_id")
        budget_id = result.get("budget_id")
        if not isinstance(record_id, str) or not isinstance(budget_id, str):
            raise ValueError(
                f"Cannot resume {output_path}: record {index + 1} lacks budget identity"
            )
        key = (record_id, budget_id)
        if key in keys:
            raise ValueError(f"Duplicate completed cell {key!r} in {output_path}")
        keys.add(key)
        results.append(result)
        valid_bytes += len(chunk)
    if output_path.exists() and output_path.stat().st_size and not raw_data.endswith(b"\n"):
        with output_path.open("ab") as stream:
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    return results, keys


def _verify_resume_checkpoint(output_path: Path, expected_fingerprint: str) -> None:
    summary_path = output_path.with_suffix(".summary.json")
    if not summary_path.exists():
        if output_path.exists() and output_path.stat().st_size == 0:
            raise ValueError(
                f"Cannot resume empty {output_path}: its checkpoint summary is missing"
            )
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot resume: checkpoint summary is invalid: {summary_path}") from exc
    if summary.get("experiment_fingerprint") != expected_fingerprint:
        raise ValueError(f"Cannot resume {output_path}: its checkpoint configuration differs")


def _build_summary(
    *,
    config: ExperimentConfig,
    output_path: Path,
    results: Sequence[dict[str, Any]],
    planned_cells: int,
    status: str,
    started_at: datetime,
    started_clock: float,
    resumed_cells: int,
    selected: Sequence[MockCase],
) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset_name in sorted({str(result["dataset"]) for result in results}):
        subset = [result for result in results if result["dataset"] == dataset_name]
        correct = sum(bool(result["correct"]) for result in subset)
        by_dataset[dataset_name] = {
            "cells": len(subset),
            "correct": correct,
            "accuracy": correct / len(subset),
            "errors": sum(result["error"] is not None for result in subset),
        }
    by_budget: dict[str, dict[str, Any]] = {}
    for budget_id in sorted({str(result["budget_id"]) for result in results}):
        subset = [result for result in results if result["budget_id"] == budget_id]
        correct = sum(bool(result["correct"]) for result in subset)
        by_budget[budget_id] = {
            "requested_budget": subset[0]["requested_budget"],
            "cells": len(subset),
            "correct": correct,
            "accuracy": correct / len(subset),
            "errors": sum(result["error"] is not None for result in subset),
            "mean_total_model_tokens": statistics.fmean(
                result["usage"]["total_tokens"] for result in subset
            ),
            "mean_budget_elapsed_seconds": statistics.fmean(
                result["consumed_budget"]["wall_time_seconds"] for result in subset
            ),
        }
    correct = sum(bool(result["correct"]) for result in results)
    completed_at = datetime.now(UTC)
    summary = {
        "status": status,
        "started_at": started_at.isoformat(),
        "updated_at": completed_at.isoformat(),
        "completed_at": completed_at.isoformat() if status == "complete" else None,
        "elapsed_seconds": round(time.monotonic() - started_clock, 6),
        "output": str(output_path),
        "planned_cells": planned_cells,
        "selected_records": planned_cells // len(_budget_grid(config)),
        "budget_cells_per_record": len(_budget_grid(config)),
        "selection_digest": selection_digest(selected),
        "selected_record_ids": [case.record_id for case in selected],
        "completed_cells": len(results),
        "remaining_cells": max(0, planned_cells - len(results)),
        "resumed_cells": resumed_cells,
        "records": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0.0,
        "errors": sum(result["error"] is not None for result in results),
        "mean_input_tokens": (
            statistics.fmean(result["usage"]["input_tokens"] for result in results)
            if results
            else 0.0
        ),
        "mean_output_tokens": (
            statistics.fmean(result["usage"]["output_tokens"] for result in results)
            if results
            else 0.0
        ),
        "by_dataset": by_dataset,
        "by_budget": by_budget,
        "config": _config_payload(config),
        "experiment_fingerprint": _config_fingerprint(config),
    }
    summary["summary_output"] = str(output_path.with_suffix(".summary.json"))
    return summary


def _write_summary(output_path: Path, summary: dict[str, Any]) -> None:
    summary_path = output_path.with_suffix(".summary.json")
    temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(summary_path)


async def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    selected, all_loaded = _select_cases(config)
    budgets = _budget_grid(config)
    output_path = _output_path(config)
    if output_path.exists() and not config.overwrite and not config.resume:
        raise FileExistsError(f"Output already exists: {output_path}; pass --resume or --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = _config_fingerprint(config)
    if config.resume:
        _verify_resume_checkpoint(output_path, fingerprint)
        results, completed_keys = _resume_results(output_path, fingerprint)
    else:
        results, completed_keys = [], set()
    resumed_cells = len(results)
    planned_cells = len(selected) * len(budgets)
    selected_record_ids = {case.record_id for case in selected}
    selected_budget_ids = {budget.budget_id for budget in budgets}
    unexpected_keys = completed_keys - set(
        itertools.product(selected_record_ids, selected_budget_ids)
    )
    if unexpected_keys:
        raise ValueError(
            f"Cannot resume {output_path}: it contains cells outside the requested slice/grid"
        )
    pending_cells = [
        (case, budget)
        for case in selected
        for budget in budgets
        if (case.record_id, budget.budget_id) not in completed_keys
    ]

    started_at = datetime.now(UTC)
    started_clock = time.monotonic()
    summary = _build_summary(
        config=config,
        output_path=output_path,
        results=results,
        planned_cells=planned_cells,
        status="running",
        started_at=started_at,
        started_clock=started_clock,
        resumed_cells=resumed_cells,
        selected=selected,
    )
    _write_summary(output_path, summary)

    tsp_cases = [case for case in all_loaded if case.dataset is DatasetKind.TSP]
    osrm = None
    if any(case.dataset is DatasetKind.TSP for case, _ in pending_cells):
        osrm = OsrmMatrixStore(
            endpoint=config.osrm_endpoint,
            cache_dir=config.osrm_cache,
            cache_only=config.osrm_cache_only,
            timeout_seconds=config.osrm_timeout_seconds,
            region_locations=_build_region_catalog(tsp_cases),
        )
        # OSRM evidence is static experiment setup, like loading model weights;
        # it is shared by every grid cell and excluded from per-cell wall budgets.
        for region in sorted(
            {case.region for case, _ in pending_cells if case.dataset is DatasetKind.TSP}
        ):
            await osrm.prepare_region(region)

    shared_backend = None
    if config.model_type == "transformers" and pending_cells:
        shared_backend = await _transformers_backend(config)

    try:
        mode = "a" if config.resume else "w"
        with output_path.open(mode, encoding="utf-8") as stream:
            for position, (case, budget) in enumerate(pending_cells, start=1):
                if not config.quiet:
                    print(
                        f"[{position}/{len(pending_cells)}] {case.record_id} "
                        f"({case.region}) {budget.budget_id}",
                        file=sys.stderr,
                    )
                backend = shared_backend if shared_backend is not None else _fake_backend(case)
                case_started = time.monotonic()
                try:
                    run = await _run_agent_case(case, backend, config, osrm, budget)
                except Exception as exc:
                    run = AgentRun(
                        terminal_reason="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                finally:
                    if shared_backend is None:
                        await backend.close()
                correct = _score(case, run.prediction, config.tsp_tolerance_km)
                total_elapsed = time.monotonic() - case_started
                consumed_budget = {
                    "wall_time_seconds": run.budget_elapsed_seconds,
                    "input_tokens": run.input_tokens,
                    "generated_tokens": run.output_tokens,
                    "reasoning_tokens": run.reasoning_tokens,
                    "total_model_tokens": run.total_tokens,
                    "model_generations": run.generations,
                    "tool_calls": run.tool_calls_used,
                    "usage_complete": run.usage_complete,
                }
                result = {
                    "experiment_fingerprint": fingerprint,
                    "cell_status": "complete",
                    "record_id": case.record_id,
                    "budget_id": budget.budget_id,
                    "dataset": case.dataset.value,
                    "source_index": case.source_index,
                    "region": case.region,
                    "prompt": case.prompt,
                    "expected": _expected_summary(case),
                    "requested_budget": budget.payload(),
                    "consumed_budget": consumed_budget,
                    "baseline_prediction": run.baseline_prediction,
                    "prediction": run.prediction,
                    "prediction_source": run.prediction_source,
                    "incumbent_timeline": run.incumbent_timeline,
                    "correct": correct,
                    "answer_text": run.answer_text,
                    "tool_calls": run.calls,
                    "usage": {
                        "input_tokens": run.input_tokens,
                        "output_tokens": run.output_tokens,
                        "generated_tokens": run.output_tokens,
                        "reasoning_tokens": run.reasoning_tokens,
                        "total_tokens": run.total_tokens,
                        "complete": run.usage_complete,
                    },
                    "stop_reason": run.stop_reason,
                    "terminal_reason": run.terminal_reason,
                    "error": run.error,
                    "elapsed_seconds": round(total_elapsed, 6),
                    "setup_elapsed_seconds": round(
                        max(0.0, total_elapsed - run.budget_elapsed_seconds), 6
                    ),
                }
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                completed_keys.add((case.record_id, budget.budget_id))
                summary = _build_summary(
                    config=config,
                    output_path=output_path,
                    results=results,
                    planned_cells=planned_cells,
                    status="running",
                    started_at=started_at,
                    started_clock=started_clock,
                    resumed_cells=resumed_cells,
                    selected=selected,
                )
                _write_summary(output_path, summary)
                if config.fail_on_error and run.error:
                    raise RuntimeError(f"{case.record_id}: {run.error}")
    finally:
        if shared_backend is not None:
            await shared_backend.close()

    summary = _build_summary(
        config=config,
        output_path=output_path,
        results=results,
        planned_cells=planned_cells,
        status="complete",
        started_at=started_at,
        started_clock=started_clock,
        resumed_cells=resumed_cells,
        selected=selected,
    )
    _write_summary(output_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run model/tool experiments over the mock geospatial datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASET_FILES, "all"],
        required=True,
        help="Dataset family; the runner infers its problem schema from this selection.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--model-type",
        "--backend",
        dest="model_type",
        choices=["fake", "transformers"],
        default="fake",
        help="Use fake for a deterministic infrastructure smoke test.",
    )
    parser.add_argument("--model", help="Hugging Face model ID for transformers runs.")
    parser.add_argument("--profile", default="gemma4_e2b_it")
    parser.add_argument("--revision")
    parser.add_argument(
        "--gpus",
        default="auto",
        help=(
            "auto uses every scheduler-visible GPU; alternatively use none or an explicit "
            "CUDA_VISIBLE_DEVICES list such as 0 or 0,1."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--quantization", choices=["none", "int8", "int4"], default="none")
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--time-budgets",
        default="unlimited",
        metavar="LIST",
        help=(
            "Comma-separated per-case wall budgets, forming a Cartesian grid with "
            "--token-budgets. Values default to seconds and may use ms/s/m/h; "
            "zero and unlimited are accepted."
        ),
    )
    parser.add_argument(
        "--token-budgets",
        default="unlimited",
        metavar="LIST",
        help=(
            "Comma-separated aggregate input+generated model-token budgets. "
            "Integers, k/m suffixes, zero, and unlimited are accepted."
        ),
    )
    parser.add_argument(
        "--max-generated-tokens",
        type=int,
        default=768,
        help="Per-generation output cap; the aggregate grid cap is --token-budgets.",
    )
    parser.add_argument("--max-tool-rounds", type=int, default=4)
    parser.add_argument(
        "--max-tool-calls",
        default="unlimited",
        help="Optional aggregate tool-call cap per case; default: unlimited.",
    )
    parser.add_argument(
        "--model-call-timeout-seconds",
        "--case-timeout-seconds",
        dest="case_timeout_seconds",
        default="unlimited",
        help=(
            "Optional safety timeout for each model count/generation call. This is "
            "separate from the aggregate --time-budgets; default: unlimited."
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, help="Maximum records after filtering; omitted means all."
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-selection-digest",
        help=(
            "Optional SHA-256 of the ordered selected record IDs. Distributed jobs "
            "fail before model loading if their deterministic sample differs."
        ),
    )
    parser.add_argument(
        "--region",
        action="append",
        default=[],
        help="Restrict to a region; repeat to select more than one.",
    )
    parser.add_argument("--output", type=Path)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Append only missing record/budget cells to a compatible output file.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--osrm-endpoint", default="https://router.project-osrm.org")
    parser.add_argument("--osrm-cache", type=Path, default=Path("data/cache/osrm"))
    parser.add_argument("--osrm-cache-only", action="store_true")
    parser.add_argument("--osrm-timeout-seconds", type=float, default=60)
    parser.add_argument("--tsp-tolerance-km", type=float, default=1.0)
    return parser


def _validated_config(
    namespace: argparse.Namespace, parser: argparse.ArgumentParser
) -> ExperimentConfig:
    if namespace.start < 0:
        parser.error("--start cannot be negative")
    if namespace.limit is not None and namespace.limit < 1:
        parser.error("--limit must be at least 1")
    if namespace.max_tool_rounds < 1:
        parser.error("--max-tool-rounds must be at least 1")
    if namespace.max_generated_tokens < 1:
        parser.error("--max-generated-tokens must be at least 1")
    try:
        time_budgets = _parse_time_budgets(namespace.time_budgets)
        token_budgets = _parse_token_budgets(namespace.token_budgets)
    except ValueError as exc:
        parser.error(str(exc))
    timeout_text = str(namespace.case_timeout_seconds).strip().casefold()
    if timeout_text in _UNLIMITED_VALUES:
        case_timeout_seconds = None
    else:
        try:
            case_timeout_seconds = float(timeout_text)
        except ValueError:
            parser.error("--model-call-timeout-seconds must be positive or unlimited")
        if case_timeout_seconds <= 0:
            parser.error("--model-call-timeout-seconds must be positive or unlimited")
    tool_text = str(namespace.max_tool_calls).strip().casefold()
    if tool_text in _UNLIMITED_VALUES:
        max_tool_calls = None
    else:
        try:
            max_tool_calls = int(tool_text)
        except ValueError:
            parser.error("--max-tool-calls must be a non-negative integer or unlimited")
        if max_tool_calls < 0:
            parser.error("--max-tool-calls must be a non-negative integer or unlimited")
    if namespace.osrm_timeout_seconds <= 0:
        parser.error("--osrm-timeout-seconds must be positive")
    if namespace.tsp_tolerance_km < 0:
        parser.error("--tsp-tolerance-km cannot be negative")
    expected_selection_digest = namespace.expected_selection_digest
    if expected_selection_digest is not None:
        expected_selection_digest = expected_selection_digest.casefold()
        if re.fullmatch(r"[0-9a-f]{64}", expected_selection_digest) is None:
            parser.error("--expected-selection-digest must be a 64-character SHA-256 hex digest")
    gpus = namespace.gpus.strip().casefold()
    if gpus not in {"none", "auto"}:
        gpu_parts = gpus.split(",")
        if not gpu_parts or any(not part.isdigit() for part in gpu_parts):
            parser.error("--gpus must be none, auto, or comma-separated integer IDs")
        if len(set(gpu_parts)) != len(gpu_parts):
            parser.error("--gpus contains a duplicate GPU ID")
    return ExperimentConfig(
        dataset=namespace.dataset,
        data_root=namespace.data_root,
        model_type=namespace.model_type,
        model=namespace.model,
        profile=namespace.profile,
        revision=namespace.revision,
        gpus=gpus,
        dtype=namespace.dtype,
        quantization=namespace.quantization,
        attention_backend=namespace.attention_backend,
        trust_remote_code=namespace.trust_remote_code,
        thinking=namespace.thinking,
        max_generated_tokens=namespace.max_generated_tokens,
        max_tool_rounds=namespace.max_tool_rounds,
        case_timeout_seconds=case_timeout_seconds,
        start=namespace.start,
        limit=namespace.limit,
        shuffle=namespace.shuffle,
        seed=namespace.seed,
        expected_selection_digest=expected_selection_digest,
        regions=tuple(namespace.region),
        output=namespace.output,
        overwrite=namespace.overwrite,
        quiet=namespace.quiet,
        fail_on_error=namespace.fail_on_error,
        osrm_endpoint=namespace.osrm_endpoint,
        osrm_cache=namespace.osrm_cache,
        osrm_cache_only=namespace.osrm_cache_only,
        osrm_timeout_seconds=namespace.osrm_timeout_seconds,
        tsp_tolerance_km=namespace.tsp_tolerance_km,
        time_budgets_seconds=time_budgets,
        token_budgets=token_budgets,
        max_tool_calls=max_tool_calls,
        resume=namespace.resume,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    config = _validated_config(parser.parse_args(argv), parser)
    try:
        summary = asyncio.run(run_experiment(config))
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

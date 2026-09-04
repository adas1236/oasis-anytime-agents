from __future__ import annotations

import json
from pathlib import Path

import pytest

from oasis.mock_experiments import (
    BudgetPoint,
    DatasetKind,
    ExperimentConfig,
    Location,
    LocationIndex,
    OsrmMatrixStore,
    _fake_backend,
    _run_agent_case,
    _score,
    _validated_config,
    build_parser,
    load_dataset,
    run_experiment,
    solve_max_coverage,
    solve_minimum_facility,
)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        dataset="max_coverage",
        data_root=DATA_ROOT,
        model_type="fake",
        model=None,
        profile="gemma4_e2b_it",
        revision=None,
        gpus="none",
        dtype="auto",
        quantization="none",
        attention_backend="auto",
        trust_remote_code=False,
        thinking=False,
        max_generated_tokens=128,
        max_tool_rounds=2,
        case_timeout_seconds=5,
        start=0,
        limit=1,
        shuffle=False,
        seed=42,
        regions=(),
        output=tmp_path / "results.jsonl",
        overwrite=False,
        quiet=True,
        fail_on_error=False,
        osrm_endpoint="https://router.project-osrm.org",
        osrm_cache=tmp_path / "osrm",
        osrm_cache_only=True,
        osrm_timeout_seconds=5,
        tsp_tolerance_km=1,
    )


def test_experiment_cli_defaults_to_bfloat16_and_visible_gpus() -> None:
    arguments = build_parser().parse_args(["--dataset", "max_coverage"])

    assert arguments.dtype == "bfloat16"
    assert arguments.gpus == "auto"


def test_cli_builds_independently_unlimited_budget_dimensions() -> None:
    parser = build_parser()
    config = _validated_config(
        parser.parse_args(
            [
                "--dataset",
                "max_coverage",
                "--time-budgets",
                "0,250ms,2m,unlimited",
                "--token-budgets",
                "0,2k,unlimited",
            ]
        ),
        parser,
    )

    assert config.time_budgets_seconds == (0.0, 0.25, 120.0, None)
    assert config.token_budgets == (0, 2_000, None)
    assert config.case_timeout_seconds is None


def test_plaintext_lookup_accepts_case_punctuation_and_internal_id() -> None:
    location = Location(
        location_id="us:johns-hopkins-hospital",
        name="Johns Hopkins Hospital",
        latitude=39.2964,
        longitude=-76.5924,
        population=7_267,
    )
    index = LocationIndex([location])

    assert index.resolve("JOHNS-HOPKINS hospital") == location
    assert index.resolve(location.location_id) == location


def test_first_coverage_records_reproduce_the_supplied_oracles() -> None:
    maximum = load_dataset(DatasetKind.MAX_COVERAGE, DATA_ROOT / "max_coverage.json")[0]
    minimum = load_dataset(DatasetKind.MINIMUM_FACILITY, DATA_ROOT / "minimum_facility.json")[0]

    max_prediction = solve_max_coverage(
        maximum.locations,
        maximum.centers_to_place or 0,
        maximum.coverage_radius_km or 0,
    )
    min_prediction = solve_minimum_facility(
        minimum.locations,
        minimum.coverage_target_percent or 0,
        minimum.coverage_radius_km or 0,
    )

    assert _score(maximum, max_prediction, 1)
    assert _score(minimum, min_prediction, 1)


def test_tsp_dataset_uses_stored_coordinates_without_population() -> None:
    case = load_dataset(DatasetKind.TSP, DATA_ROOT / "tsp.json")[0]

    assert case.region == "United States"
    assert case.answer == 176
    assert all(location.population is None for location in case.locations)


@pytest.mark.asyncio
async def test_fake_agent_calls_solver_with_plaintext_names(tmp_path: Path) -> None:
    case = load_dataset(DatasetKind.MAX_COVERAGE, DATA_ROOT / "max_coverage.json")[0]
    backend = _fake_backend(case)

    result = await _run_agent_case(case, backend, _config(tmp_path), None)
    await backend.close()

    assert result.error is None
    assert result.prediction is not None
    assert _score(case, result.prediction, 1)
    assert result.calls[0]["arguments"]["location_names"] == [
        location.name for location in case.locations
    ]


@pytest.mark.asyncio
async def test_zero_token_budget_returns_feasible_baseline(tmp_path: Path) -> None:
    case = load_dataset(DatasetKind.MAX_COVERAGE, DATA_ROOT / "max_coverage.json")[0]
    backend = _fake_backend(case)

    result = await _run_agent_case(
        case,
        backend,
        _config(tmp_path),
        None,
        BudgetPoint("time-unlimited_tokens-0", None, 0, None),
    )
    await backend.close()

    assert result.terminal_reason == "token_budget_exhausted"
    assert result.total_tokens == 0
    assert result.prediction == result.baseline_prediction
    assert result.incumbent_timeline[0]["source"] == "baseline"


@pytest.mark.asyncio
async def test_grid_results_are_checkpointed_and_resumable(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.time_budgets_seconds = (0.0, None)
    config.token_budgets = (0, None)

    first_summary = await run_experiment(config)
    lines_after_first_run = config.output.read_text(encoding="utf-8").splitlines()

    assert first_summary["status"] == "complete"
    assert first_summary["planned_cells"] == 4
    assert len(lines_after_first_run) == 4
    assert {json.loads(line)["budget_id"] for line in lines_after_first_run} == {
        "time-0s_tokens-0",
        "time-0s_tokens-unlimited",
        "time-unlimited_tokens-0",
        "time-unlimited_tokens-unlimited",
    }
    unlimited_cell = next(
        json.loads(line)
        for line in lines_after_first_run
        if json.loads(line)["budget_id"] == "time-unlimited_tokens-unlimited"
    )
    assert unlimited_cell["correct"] is True
    assert unlimited_cell["terminal_reason"] == "completed"
    assert unlimited_cell["prediction_source"] == "solve_current_problem"

    # Simulate termination in the middle of the next JSONL write. Resume
    # preserves the fragment separately and restarts from the last durable cell.
    with config.output.open("ab") as stream:
        stream.write(b'{"record_id": "interrupted"')

    config.resume = True
    resumed_summary = await run_experiment(config)

    assert resumed_summary["resumed_cells"] == 4
    assert len(config.output.read_text(encoding="utf-8").splitlines()) == 4
    assert config.output.with_suffix(".jsonl.interrupted").exists()


@pytest.mark.asyncio
async def test_tsp_solver_uses_directed_driving_matrix(tmp_path: Path) -> None:
    locations = tuple(
        Location(f"r:{name.lower()}", name, 0, float(index))
        for index, name in enumerate(("Depot", "Clinic", "Shelter"))
    )
    store = OsrmMatrixStore(
        endpoint="https://example.invalid",
        cache_dir=tmp_path,
        cache_only=True,
        timeout_seconds=1,
        region_locations={"Region": locations},
    )
    store._memory["Region"] = [
        [0, 1_000, 5_000],
        [5_000, 0, 1_000],
        [1_000, 5_000, 0],
    ]

    result = await store.solve("Region", locations)

    assert result["distance_km"] == 3
    assert result["route"] == ["Depot", "Clinic", "Shelter", "Depot"]

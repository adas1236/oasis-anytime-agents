"""CPU-only checks against Google's actual native declaration renderer."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jinja2 import Environment

from oasis.llm.adapters import Gemma4ChatAdapter
from oasis.llm.gemma_schema import gemma_tool_schema
from oasis.llm.schemas import ChatMessage, ToolDefinition
from oasis.tools import create_tool_registry


@pytest.fixture(scope="module")
def definitions():
    return create_tool_registry(discover_entry_points=False).model_definitions()


@pytest.fixture(scope="module")
def native_template():
    path = Path(__file__).parents[1] / "fixtures/gemma4_schema_template.jinja"
    return Environment().from_string(path.read_text()).module


def test_adapter_renders_every_live_tool_without_blank_types(definitions, native_template):
    original = copy.deepcopy([tool.model_dump() for tool in definitions])

    class Processor:
        chat_template = "native Gemma schema fixture"

        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "A user's planning prompt."}]
            self.tools = kwargs["tools"]
            return {
                "input_ids": "\n".join(
                    native_template.format_function_declaration(t) for t in self.tools
                )
            }

    processor = Processor()
    inputs = Gemma4ChatAdapter("google/gemma-4-E2B-it").prepare_inputs(
        processor,
        [ChatMessage(role="user", content="A user's planning prompt.")],
        tools=definitions,
        thinking_enabled=True,
    )
    assert len(processor.tools) == 21
    assert 'type:<|"|><|"|>' not in inputs["input_ids"]
    assert "$ref" not in inputs["input_ids"]
    assert "#/$defs/" not in inputs["input_ids"]
    assert [tool.model_dump() for tool in definitions] == original


def test_problem_enums_and_both_policy_branches_survive(definitions, native_template):
    tool = next(t for t in definitions if t.name == "compile_problem")
    before = native_template.format_function_declaration(tool.transformers_schema())
    assert 'type_id:{type:<|"|><|"|>}' in before  # Reproduces the pilot defect.
    projected = gemma_tool_schema(tool)
    props = projected["function"]["parameters"]["properties"]
    assert {"tsp", "max_weighted_coverage", "min_cost_target_coverage"} <= set(
        props["type_id"]["enum"]
    )
    assert props["policy"]["type"] == "object"
    assert {"site_limit", "coverage_target", "depot_ids", "shift_length", "time_units"} <= props[
        "policy"
    ]["properties"].keys()
    after = native_template.format_function_declaration(projected)
    for value in [
        "tsp",
        "max_weighted_coverage",
        "site_limit",
        "depot_ids",
        '"required":["depot_ids","shift_length","time_units"]',
    ]:
        assert value in after
    # Maps must not acquire bogus literal properties named title/additionalProperties.
    mapping = props["travel_matrix_artifact_ids"]
    assert mapping["properties"] == {}
    assert '"type":"string"' in mapping["description"]


def test_travel_enums_and_optional_types_survive(definitions, native_template):
    tool = next(t for t in definitions if t.name == "travel_matrix")
    projected = gemma_tool_schema(tool)
    props = projected["function"]["parameters"]["properties"]
    assert "routed_provider" in props["strategy"]["enum"]
    assert props["route_annotation"]["enum"] == ["duration", "distance"]
    assert props["graph_artifact_id"]["nullable"] is True
    assert props["graph_artifact_id"]["type"] == "string"
    rendered = native_template.format_function_declaration(projected)
    assert '"pattern":"^sha256-' in rendered
    assert '"default":"meters"' in rendered
    assert "graph_shortest_path" in rendered
    assert "For routed_provider, omit all" in rendered
    assert "route_annotation" in projected["function"]["description"]


def test_constraints_mixed_unions_and_array_items_remain_visible(native_template):
    tool = ToolDefinition(
        name="example",
        description="Example",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "value": {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "boolean"}]},
                "points": {
                    "type": "array",
                    "items": {"type": "number", "maximum": 90},
                    "minItems": 2,
                },
                "mode": {"type": "string", "const": "fixed"},
            },
        },
    )
    rendered = native_template.format_function_declaration(gemma_tool_schema(tool))
    for constraint in [
        '"minimum":0',
        '"type":"boolean"',
        '"maximum":90',
        '"minItems":2',
        '"const":"fixed"',
        "additionalProperties (JSON Schema): false",
    ]:
        assert constraint in rendered
    assert 'type:<|"|><|"|>' not in rendered


@pytest.mark.parametrize("ref", ["https://example.org/schema", "#/$defs/Recursive"])
def test_unsupported_references_fail_instead_of_silently_disappearing(ref):
    tool = ToolDefinition(
        name="example",
        description="Example",
        input_schema={
            "$defs": {"Recursive": {"$ref": "#/$defs/Recursive"}},
            "type": "object",
            "properties": {"value": {"$ref": ref}},
        },
    )
    with pytest.raises(ValueError, match="Unsupported or recursive"):
        gemma_tool_schema(tool)


def test_nullable_reference_does_not_change_execution_schema():
    tool = ToolDefinition(
        name="example",
        description="Example",
        input_schema={
            "$defs": {"Choice": {"type": "string", "enum": ["a", "b"]}},
            "type": "object",
            "properties": {
                "choice": {"anyOf": [{"$ref": "#/$defs/Choice"}, {"type": "null"}], "default": None}
            },
        },
    )
    original = json.dumps(tool.input_schema, sort_keys=True)
    field = gemma_tool_schema(tool)["function"]["parameters"]["properties"]["choice"]
    assert field["type"] == "string" and field["nullable"] is True
    assert field["enum"] == ["a", "b"]
    assert json.dumps(tool.input_schema, sort_keys=True) == original

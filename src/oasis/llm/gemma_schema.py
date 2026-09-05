"""Project JSON Schema into Gemma's native tool-declaration vocabulary.

Gemma's template understands types/properties/items/enums, but not JSON Schema
references or unions. Keep unsupported constraints in visible descriptions;
the original registry schema remains authoritative for execution validation.
"""

from __future__ import annotations

import json
from typing import Any

from oasis.llm.schemas import ToolDefinition


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _expand(value: Any, root: dict[str, Any], trail: tuple[str, ...] = ()) -> Any:
    if isinstance(value, list):
        return [_expand(item, root, trail) for item in value]
    if not isinstance(value, dict):
        return value
    if "$ref" in value:
        ref = value["$ref"]
        if not ref.startswith("#/") or ref in trail:
            raise ValueError(f"Unsupported or recursive Gemma tool schema reference: {ref}")
        target: Any = root
        for part in ref[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        expanded = _expand(target, root, (*trail, ref))
        return {**expanded, **_expand({k: v for k, v in value.items() if k != "$ref"}, root, trail)}
    return {k: _expand(v, root, trail) for k, v in value.items() if k != "$defs"}


def _describe(schema: dict[str, Any], text: str) -> None:
    schema["description"] = " ".join(filter(None, (schema.get("description"), text)))


def _project(schema: dict[str, Any]) -> dict[str, Any]:
    source = dict(schema)
    result: dict[str, Any]
    union_key = next((key for key in ("anyOf", "oneOf") if key in source), None)
    if union_key:
        alternatives = source.pop(union_key)
        non_null = [item for item in alternatives if item.get("type") != "null"]
        nullable = len(non_null) != len(alternatives)
        if len(non_null) == 1:
            result = _project({**non_null[0], **source})
        elif non_null and all(item.get("type") == "string" and "enum" in item for item in non_null):
            result = _project(
                {
                    **source,
                    "type": "string",
                    "enum": list(dict.fromkeys(v for item in non_null for v in item["enum"])),
                }
            )
        elif non_null and all(item.get("type") == "object" for item in non_null):
            names = dict.fromkeys(name for item in non_null for name in item.get("properties", {}))
            properties = {}
            for name in names:
                choices = [
                    item["properties"][name]
                    for item in non_null
                    if name in item.get("properties", {})
                ]
                properties[name] = (
                    choices[0] if all(c == choices[0] for c in choices) else {"anyOf": choices}
                )
            required = set(non_null[0].get("required", []))
            for item in non_null[1:]:
                required.intersection_update(item.get("required", []))
            result = _project(
                {**source, "type": "object", "properties": properties, "required": sorted(required)}
            )
        else:
            result = _project({**source, "type": "any"})
        if len(non_null) != 1:
            # A merged native shape is only a guide, never a replacement for the
            # branch-specific requirements (including oneOf's exclusive choice).
            _describe(
                result, f"Exact JSON Schema alternatives: {_json({union_key: alternatives})}."
            )
        if nullable:
            result["nullable"] = True
        return result

    kind = source.get("type", "object" if "properties" in source else "any")
    if isinstance(kind, list):
        return _project(
            {
                **{k: v for k, v in source.items() if k != "type"},
                "anyOf": [{"type": k} for k in kind],
            }
        )
    result = {"type": kind}
    if source.get("description"):
        result["description"] = source["description"]
    if source.get("nullable"):
        result["nullable"] = True
    if kind == "object":
        # An explicit empty properties map prevents the template from treating
        # additionalProperties/title as literal argument names.
        result["properties"] = {
            name: _project(value) for name, value in source.get("properties", {}).items()
        }
        if source.get("required"):
            result["required"] = source["required"]
        if "additionalProperties" in source:
            _describe(
                result,
                "Dictionary additionalProperties (JSON Schema): "
                f"{_json(source['additionalProperties'])}.",
            )
    elif kind == "array":
        items = source.get("items", {})
        result["items"] = _project(items) if isinstance(items, dict) else {"type": "any"}
    if kind == "string" and "enum" in source:
        result["enum"] = source["enum"]
    # The template drops these keys. Put them in the field's description so
    # defaults, numeric bounds, patterns, tuple schemas, etc. remain visible.
    rendered = {
        "type",
        "description",
        "title",
        "nullable",
        "properties",
        "required",
        "additionalProperties",
        "items",
    }
    if kind == "string":
        rendered.add("enum")
    constraints = {k: v for k, v in source.items() if k not in rendered}
    if constraints:
        _describe(result, f"JSON Schema constraints: {_json(constraints)}.")
    if kind == "any":
        _describe(result, "Any JSON value unless constrained above.")
    return result


def gemma_tool_schema(tool: ToolDefinition) -> dict[str, Any]:
    """Return a fresh native declaration; never mutate the shared tool schema."""

    parameters = _project(_expand(tool.input_schema, tool.input_schema))
    description = tool.description
    # Gemma ignores the root parameter description, unlike field descriptions.
    if parameters.get("description"):
        description += " " + parameters.pop("description")
    return {
        "type": "function",
        "function": {"name": tool.name, "description": description, "parameters": parameters},
    }

"""Reusable pattern third-party tools can apply to each declared smoke input."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from oasis.artifacts import LocalArtifactStore
from oasis.tools import CancellationToken, Tool, ToolContext
from oasis.tools.builtins import builtin_tools
from oasis.tools.testing import assert_tool_contract


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", builtin_tools(), ids=lambda tool: tool.spec.name)
async def test_builtin_conforms_to_public_tool_contract(tool: Tool, tmp_path: Path) -> None:
    context = ToolContext(
        run_id="contract-test",
        artifact_store=LocalArtifactStore(tmp_path),
        deadline_monotonic=time.monotonic() + 2,
        cancellation=CancellationToken(),
        seed=0,
    )

    await assert_tool_contract(tool, context=context)

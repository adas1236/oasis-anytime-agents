"""Explicit validated registry for problem-family plugins."""

from __future__ import annotations

import re
from collections.abc import Iterable

from oasis.problems.protocols import ProblemPlugin


class ProblemRegistryError(ValueError):
    """Raised when a problem plugin cannot safely enter a registry."""


class ProblemRegistry:
    """Read-mostly registry keyed by stable problem type identifiers."""

    def __init__(self, plugins: Iterable[ProblemPlugin] = ()) -> None:
        self._plugins: dict[str, ProblemPlugin] = {}
        for plugin in plugins:
            self.register(plugin)

    def register(self, plugin: ProblemPlugin) -> None:
        if not isinstance(plugin, ProblemPlugin):
            raise ProblemRegistryError("problem plugin does not implement the shared protocol")
        if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", plugin.type_id) is None:
            raise ProblemRegistryError("problem type ID must be a stable lowercase identifier")
        if re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", plugin.version) is None:
            raise ProblemRegistryError("problem plugin version must use MAJOR.MINOR.PATCH")
        if plugin.type_id in self._plugins:
            raise ProblemRegistryError(f"duplicate problem type {plugin.type_id!r}")
        self._plugins[plugin.type_id] = plugin

    def get(self, type_id: str) -> ProblemPlugin:
        try:
            return self._plugins[type_id]
        except KeyError as error:
            raise ProblemRegistryError(f"unknown problem type {type_id!r}") from error

    def list(self) -> tuple[ProblemPlugin, ...]:
        return tuple(self._plugins[key] for key in sorted(self._plugins))


def create_builtin_problem_registry() -> ProblemRegistry:
    """Create every built-in problem plugin without import-time registration."""

    from oasis.problems.location_allocation import LocationAllocationPlugin
    from oasis.problems.routing import RouteServicePlugin
    from oasis.problems.schemas import LocationProblemType, RouteProblemType

    return ProblemRegistry(
        (
            *(LocationAllocationPlugin(type_id) for type_id in LocationProblemType),
            *(RouteServicePlugin(type_id) for type_id in RouteProblemType),
        )
    )

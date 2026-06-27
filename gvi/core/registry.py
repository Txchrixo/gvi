"""Plugin registry with capability-based lookup."""
from __future__ import annotations

from collections import defaultdict

from gvi.core.plugin import Plugin, load_builtin_plugins
from gvi.core.types import CapabilityType


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}
        self._by_type: dict[CapabilityType, list[Plugin]] = defaultdict(list)

    def register(self, plugin: Plugin) -> None:
        cap = plugin.capability()
        if cap.id in self._plugins:
            raise ValueError(f"Plugin already registered: {cap.id}")
        self._plugins[cap.id] = plugin
        self._by_type[cap.type].append(plugin)
        self._by_type[cap.type].sort(key=lambda p: p.capability().priority, reverse=True)

    def get(self, capability_id: str) -> Plugin:
        if capability_id not in self._plugins:
            raise KeyError(f"Plugin '{capability_id}' not found in registry")
        return self._plugins[capability_id]

    def by_type(self, capability_type: CapabilityType) -> list[Plugin]:
        return list(self._by_type[capability_type])

    def all(self) -> list[Plugin]:
        return list(self._plugins.values())

    def has(self, capability_id: str) -> bool:
        return capability_id in self._plugins

    @classmethod
    def with_builtins(cls) -> "PluginRegistry":
        reg = cls()
        for plugin in load_builtin_plugins():
            try:
                reg.register(plugin)
            except ValueError:
                continue
        return reg
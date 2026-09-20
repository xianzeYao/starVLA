"""Minimal import compatibility for DA3's optional ``addict`` dependency."""

from __future__ import annotations

import importlib
import sys
import types


def ensure_addict_compatibility() -> None:
    """Expose the attribute-dict surface imported by DA3 when addict is absent."""

    try:
        importlib.import_module("addict")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "addict":
            raise

    class AttributeDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

        def __delattr__(self, name):
            try:
                del self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    module = types.ModuleType("addict")
    module.Dict = AttributeDict
    sys.modules.setdefault("addict", module)

"""
MINIMAL PYDANTIC-COMPATIBLE TEST SHIM -- NOT REAL PYDANTIC.

Why this exists: the audit sandbox this was written in has no network
access, so `pip install pydantic` is impossible. Without *some* stand-in,
`gvi.core.types` and `gvi.core.plugin` (the only two files in the whole
package that import pydantic) cannot even be imported, which means the
real Orchestrator/Planner/Registry pipeline -- the thing that actually
matters for "did the v1.1 rewrite of opencv_segmenter.py wire up
correctly" -- could never be exercised end-to-end. Only the dependency-free
`_opencv_core.py` module could be tested directly.

This shim implements just enough of the public pydantic v2 surface that
`gvi/core/types.py` and `gvi/core/plugin.py` use -- BaseModel with
annotated fields, `Field(default=..., default_factory=..., ge=..., le=...)`,
`field_validator`, and `.model_dump(mode=...)` -- to let those two files
run UNMODIFIED against real test images, with real DetectedElement /
SegmentationResult / ConversionRequest objects flowing through the real
Orchestrator.

What it deliberately does NOT do, unlike real pydantic:
  - No strict type coercion/validation beyond a tiny bit of Path coercion.
  - No JSON schema generation, no serialization edge cases beyond a basic
    `model_dump`.
  - No error aggregation / `ValidationError` with rich diagnostics --
    failures raise a plain `TypeError`/`ValueError`.
  - Not perf-tested, not fuzzed, not meant to be installed for anything
    other than this one-time verification run.

**Do not use this in production. Run `pip install pydantic` for real use.**
Its only purpose is to upgrade "the wiring was never executed" to "the
wiring was executed, here are the real results" for this specific audit.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, get_args, get_origin


class _FieldInfo:
    __slots__ = ("default", "default_factory", "ge", "le")

    def __init__(self, default=..., default_factory=None, ge=None, le=None):
        self.default = default
        self.default_factory = default_factory
        self.ge = ge
        self.le = le


def Field(default: Any = ..., *, default_factory=None, ge=None, le=None, **_ignored) -> _FieldInfo:
    return _FieldInfo(default=default, default_factory=default_factory, ge=ge, le=le)


def field_validator(*field_names: str):
    def decorator(fn):
        target = getattr(fn, "__func__", fn)
        target._shim_validator_fields = field_names
        return fn
    return decorator


def _is_basemodel(tp) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _resolve_annotation(annotation, module_name):
    if not isinstance(annotation, str):
        return annotation
    import sys, typing
    mod = sys.modules.get(module_name)
    ns = vars(mod) if mod is not None else {}
    try:
        return eval(annotation, dict(ns), {**vars(typing)})  # noqa: S307 (test shim only)
    except Exception:
        return annotation


def _coerce(value, annotation, module_name=""):
    annotation = _resolve_annotation(annotation, module_name)
    if value is None:
        return None
    origin = get_origin(annotation)
    if origin is not None:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if origin in (list, tuple, set) and args and isinstance(value, (list, tuple, set)):
            inner = args[0]
            return type(value)(_coerce(v, inner, module_name) for v in value)
        if origin is dict and len(args) == 2 and isinstance(value, dict):
            kt, vt = args
            return {k: _coerce(v, vt, module_name) for k, v in value.items()}
        if value is not None:
            for a in args:
                try:
                    return _coerce(value, a, module_name)
                except Exception:
                    continue
        return value
    if _is_basemodel(annotation):
        if isinstance(annotation, type) and isinstance(value, annotation):
            return value
        if isinstance(value, dict):
            return annotation(**value)
        return value
    try:
        if annotation is Path and isinstance(value, str):
            return Path(value)
    except TypeError:
        pass
    return value


class _ModelMeta(type):
    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        fields: dict[str, dict[str, Any]] = {}
        for base in reversed(cls.__mro__[1:]):
            fields.update(getattr(base, "__shim_fields__", {}))

        annotations = namespace.get("__annotations__", {})
        for fname, ftype in annotations.items():
            raw_default = namespace.get(fname, ...)
            # Keep the raw (possibly string) annotation; it's resolved lazily
            # at coercion time via _resolve_annotation, because forward refs
            # to sibling models may not exist yet when the metaclass runs.
            info = {"annotation": ftype, "default": ..., "default_factory": None, "required": True}
            if isinstance(raw_default, _FieldInfo):
                info["default"] = raw_default.default
                info["default_factory"] = raw_default.default_factory
                info["required"] = raw_default.default is ... and raw_default.default_factory is None
            elif raw_default is not ...:
                info["default"] = raw_default
                info["required"] = False
            fields[fname] = info
        cls.__shim_fields__ = fields
        cls.__shim_module__ = namespace.get("__module__", "")

        validators: dict[str, list] = {}
        for base in reversed(cls.__mro__[1:]):
            for fname, fns in getattr(base, "__shim_validators__", {}).items():
                validators.setdefault(fname, []).extend(fns)
        for attr in namespace.values():
            target = getattr(attr, "__func__", attr)
            field_names = getattr(target, "_shim_validator_fields", None)
            if field_names:
                for fname in field_names:
                    validators.setdefault(fname, []).append(attr)
        cls.__shim_validators__ = validators
        return cls


class BaseModel(metaclass=_ModelMeta):
    # NOTE: deliberately NOT type-annotated (`: dict[...]`) -- annotating
    # them would make the metaclass treat these internal registries as if
    # they were user model fields, which was an actual bug caught by
    # actually running this against gvi.core.types (see CHANGELOG_v1.1).
    __shim_fields__ = {}
    __shim_validators__ = {}

    def __init__(self, **data: Any) -> None:
        for fname, info in type(self).__shim_fields__.items():
            if fname in data:
                value = data.pop(fname)
            elif info["default_factory"] is not None:
                value = info["default_factory"]()
            elif info["default"] is not ...:
                value = info["default"]
            elif info["required"]:
                raise TypeError(f"{type(self).__name__}: missing required field '{fname}'")
            else:
                value = None

            value = _coerce(value, info["annotation"], getattr(type(self), "__shim_module__", ""))

            for validator in type(self).__shim_validators__.get(fname, []):
                bound = getattr(validator, "__func__", validator)
                value = bound(type(self), value)

            setattr(self, fname, value)

        for k, v in data.items():
            setattr(self, k, v)

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        out = {}
        for fname in type(self).__shim_fields__:
            out[fname] = self._dump_value(getattr(self, fname), mode)
        return out

    # pydantic v1 compat alias some code might call
    def dict(self, **kw) -> dict[str, Any]:
        return self.model_dump(mode="python")

    @classmethod
    def _dump_value(cls, value, mode):
        if isinstance(value, BaseModel):
            return value.model_dump(mode=mode)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value) if mode == "json" else value
        if isinstance(value, dict):
            return {k: cls._dump_value(v, mode) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._dump_value(v, mode) for v in value]
        if isinstance(value, set):
            return [cls._dump_value(v, mode) for v in value]
        return value

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={getattr(self, k)!r}" for k in type(self).__shim_fields__)
        return f"{type(self).__name__}({fields})"

    @classmethod
    def model_validate(cls, data):
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            return cls(**data)
        raise TypeError(f"{cls.__name__}.model_validate expects dict or instance, got {type(data)}")

    @classmethod
    def model_validate_json(cls, json_str):
        import json as _json
        return cls.model_validate(_json.loads(json_str))

    def model_dump_json(self, indent=None, **kw):
        import json as _json
        return _json.dumps(self.model_dump(mode="json"), indent=indent, default=str)

    @classmethod
    def model_rebuild(cls, **_kwargs) -> None:
        # Real pydantic v2 needs this to resolve forward refs (e.g. a
        # self-referential `children: list["IRNode"]`). This shim does no
        # strict type resolution at all, so there is nothing to rebuild --
        # but the method must exist, or unmodified gvi source calling it
        # crashes on import. Found by actually running the real code, not
        # by reading it.
        return None

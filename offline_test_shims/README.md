# Offline test shims — NOT part of the GVI package

This directory exists purely so the v1.1.1 audit could run the real
`Orchestrator`/`Planner`/`Registry` pipeline end-to-end in a sandbox with no
network access (so `pip install pydantic` was impossible). It contains a
minimal, explicitly non-production pydantic-compatible `BaseModel`/`Field`/
`field_validator` implementation — see the module docstring in
`pydantic/__init__.py` for exactly what it does and does not do.

**Do not import this in production.** It is not on `gvi`'s dependency list,
not installed by `pip install -e .`, and not referenced anywhere in
`gvi/`. It is here only so the claims in `docs/CHANGELOG_v1.1.md` about
"the real pipeline was executed end-to-end" are themselves reproducible —
prepend this directory to `PYTHONPATH`/`sys.path` *before* a real
`pydantic` would be found, and the unmodified `gvi` package runs against
it. With real `pydantic` installed normally, this directory is simply
never imported and has zero effect.

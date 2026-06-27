"""Minimal rich.console shim — TEST ONLY."""
import json as _json, re as _re
class _Status:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
def _strip(s):
    return _re.sub(r"\[/?[a-zA-Z0-9_ ]+\]", "", str(s))
class Console:
    def print(self, *args, **kwargs):
        for a in args:
            print(_strip(a))
    def print_json(self, data=None, **kwargs):
        if data is not None:
            print(data if isinstance(data, str) else _json.dumps(data, default=str, indent=2))
    def status(self, *a, **k):
        return _Status()

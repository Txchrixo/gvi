"""Minimal rich.table shim — TEST ONLY."""
class Table:
    def __init__(self, *a, **k):
        self.title = k.get("title", "")
        self.rows = []
    def add_column(self, *a, **k): pass
    def add_row(self, *cells):
        self.rows.append([str(c) for c in cells])
    def __rich_console__(self, *a, **k): return []
    def __str__(self):
        out = [self.title] if self.title else []
        for r in self.rows:
            out.append("  " + " | ".join(r))
        return "\n".join(out)

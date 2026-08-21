from __future__ import annotations

import csv
import os
from typing import Dict, List


class MetricsLogger:
    def __init__(self, path: str, fields: List[str]):
        self.path = path
        self.fields = list(fields)
        self._rows: List[Dict[str, float]] = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.fields)

    def add(self, **kw) -> None:
        self._rows.append({k: kw.get(k, float("nan")) for k in self.fields})

    def flush(self) -> None:
        if not self._rows:
            return
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            for r in self._rows:
                w.writerow(r)
        self._rows.clear()

    def save(self) -> None:
        self.flush()

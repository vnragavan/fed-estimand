"""FLamby Fed-Heart-Disease center names.

Matches ``centers_number`` in ``flamby.datasets.fed_heart_disease.dataset``:
cleveland=0, hungarian=1, switzerland=2, va (Long Beach VA)=3.
"""
from __future__ import annotations

from typing import Dict

FLAMBY_SITE_LABELS: Dict[int, str] = {
    0: "Cleveland",
    1: "Hungary",
    2: "Switzerland",
    3: "Long Beach VA",
}

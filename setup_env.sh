#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python - <<'PY'
import importlib
import platform

print("Python:", platform.python_version())
for pkg in ["flwr", "flamby", "torch", "opacus", "numpy", "pandas", "sklearn", "matplotlib", "tqdm"]:
    try:
        module = importlib.import_module(pkg if pkg != "sklearn" else "sklearn")
        version = getattr(module, "__version__", "unknown")
        print(f"{pkg}: {version}")
    except Exception as exc:
        print(f"{pkg}: ERROR ({exc})")
PY

echo "Now download Fed-Heart-Disease per FLamby README. Check https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_heart_disease for the current download command."

# fed-estimand-workshop — reproduce experiments 1, 3, and 5 (CPU).
#
# Prerequisites:
#   1. bash setup_env.sh          (creates .venv)
#   2. Download Fed-Heart-Disease (see README § "Download Fed-Heart-Disease")
#   3. From this directory: make   OR   make all
#
# Override Python: make PYTHON=python3 all

WORKDIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= $(WORKDIR)/.venv/bin/python

.PHONY: all help check-env check-outputs exp1 exp3 exp5 clean fresh

help:
	@echo "fed-estimand-workshop — experiment targets"
	@echo ""
	@echo "  make all     Run Exp 1 → Exp 3 → Exp 5, then verify key outputs (default)."
	@echo "  make exp1    Exp 1 sweep + figure (results/exp1_* + figures/exp1.pdf)."
	@echo "  make exp3    Per-site calibration (requires Exp 1 model exp1_alpha1.0_seed0)."
	@echo "  make exp5    Exp 5 DP sweep + paper figure + LaTeX snippets."
	@echo "  make check-env   Verify venv + FLamby data load."
	@echo "  make check-outputs  Assert expected figures/results exist."
	@echo "  make clean   Remove results/models, predictions, logs, exp tables, figures."
	@echo "  make fresh   clean, then make all (full clean reproduction)."
	@echo ""
	@echo "Prereqs: bash setup_env.sh ; download Fed-Heart-Disease (README)."
	@echo "Python: $(PYTHON)"

# Default for bare `make`
.DEFAULT_GOAL := all

all: check-env
	@echo "==> Running experiments 1 → 3 → 5 (CPU; wall time depends on machine)."
	$(MAKE) exp1
	$(MAKE) exp3
	$(MAKE) exp5
	$(MAKE) check-outputs
	@echo "==> All experiments finished."

check-env:
	@test -x "$(PYTHON)" || (echo "error: $(PYTHON) not found. Run:  bash setup_env.sh"; exit 1)
	@cd "$(WORKDIR)" && "$(PYTHON)" -c "from src.data import get_site_sizes; s=get_site_sizes(); print('site train sizes:', s); assert len(s)==4"

exp1: check-env
	cd "$(WORKDIR)" && "$(PYTHON)" -m experiments.exp1_sweep
	cd "$(WORKDIR)" && "$(PYTHON)" -m experiments.exp1_figure

exp3: check-env
	@test -f "$(WORKDIR)/results/models/exp1_alpha1.0_seed0.pt" || \
		(echo "error: missing results/models/exp1_alpha1.0_seed0.pt — run  make exp1  first."; exit 1)
	cd "$(WORKDIR)" && "$(PYTHON)" -m experiments.exp3_calibration

exp5: check-env
	cd "$(WORKDIR)" && "$(PYTHON)" -m experiments.exp5_sweep
	cd "$(WORKDIR)" && "$(PYTHON)" -m experiments.exp5_figure

clean:
	bash "$(WORKDIR)/scripts/clean_results.sh"

fresh: clean
	@$(MAKE) all

check-outputs:
	@test -f "$(WORKDIR)/figures/exp1.pdf" || (echo "missing: figures/exp1.pdf"; exit 1)
	@test -f "$(WORKDIR)/figures/exp3.pdf" || (echo "missing: figures/exp3.pdf"; exit 1)
	@test -f "$(WORKDIR)/figures/exp5.pdf" || (echo "missing: figures/exp5.pdf"; exit 1)
	@test -f "$(WORKDIR)/results/exp1_results.json" || (echo "missing: results/exp1_results.json"; exit 1)
	@test -f "$(WORKDIR)/results/exp3_results.json" || (echo "missing: results/exp3_results.json"; exit 1)
	@test -f "$(WORKDIR)/results/exp5_results.json" || (echo "missing: results/exp5_results.json"; exit 1)
	@echo "check-outputs: OK"

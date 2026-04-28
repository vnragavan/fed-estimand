from __future__ import annotations

import os
import random
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

import flwr as fl
import numpy as np
import torch
from flwr.server import ServerConfig
from flwr.server.strategy import Strategy
from numpy.random import SeedSequence

import src.data
from src.client import FlowerClient
from src.data import get_client_dataloader
from src.model import build_model

torch.set_default_device("cpu")


def _log_run(experiment: str, params: Dict[str, Any], elapsed: float) -> None:
    log_path = Path("results") / "run_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{datetime.utcnow().isoformat()}Z | experiment={experiment} | "
            f"params={params} | elapsed_sec={elapsed:.2f}\n"
        )


def run_federation(
    strategy: Strategy,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    lr: float,
    batch_size: int,
    run_id: str,
    seed: int,
) -> Dict[str, Any]:
    torch.set_default_device("cpu")
    os.environ["RUN_ID"] = str(run_id)
    os.environ["SEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(strategy, "run_id"):
        try:
            setattr(strategy, "run_id", run_id)
        except Exception:
            pass

    seeds_log = Path("results") / "diagnostics_seeds.log"
    seeds_log.parent.mkdir(parents=True, exist_ok=True)
    with seeds_log.open("a", encoding="utf-8") as f:
        f.write(f"[{run_id}] seed={seed} torch_initial_state={torch.initial_seed()}\n")

    start = perf_counter()

    def client_fn(cid: str) -> FlowerClient:
        client_id = int(cid)
        NUM_CLIENTS_LOCAL = src.data.NUM_CLIENTS
        ss = SeedSequence(int(seed))
        child_seeds = ss.spawn(NUM_CLIENTS_LOCAL)
        client_seed_int = int(child_seeds[client_id].generate_state(1)[0])
        torch.manual_seed(client_seed_int)
        np.random.seed(client_seed_int)
        random.seed(client_seed_int)

        model = build_model()
        train_loader = get_client_dataloader(client_id=client_id, batch_size=batch_size, train=True)
        test_loader = get_client_dataloader(client_id=client_id, batch_size=batch_size, train=False)
        return FlowerClient(
            client_id=client_id,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            local_epochs=local_epochs,
            lr=lr,
            run_id=run_id,
        )

    def fit_cfg(server_round: int) -> Dict[str, Any]:
        return {"run_id": run_id, "server_round": server_round}

    def eval_cfg(server_round: int) -> Dict[str, Any]:
        return {"run_id": run_id, "server_round": server_round}

    if hasattr(strategy, "on_fit_config_fn"):
        strategy.on_fit_config_fn = fit_cfg
    if hasattr(strategy, "on_evaluate_config_fn"):
        strategy.on_evaluate_config_fn = eval_cfg

    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
    except Exception:
        traceback.print_exc()
        raise

    final_params = None
    if hasattr(strategy, "latest_parameters") and getattr(strategy, "latest_parameters") is not None:
        final_params = getattr(strategy, "latest_parameters")

    if final_params is None:
        raise RuntimeError("Final parameters unavailable on strategy; ensure strategy tracks latest aggregated parameters.")

    model = build_model()
    state = model.state_dict()
    keys = list(state.keys())
    model.load_state_dict({k: torch.tensor(v, dtype=state[k].dtype) for k, v in zip(keys, final_params)}, strict=True)
    model_path = Path("results") / "models" / f"{run_id}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    elapsed = perf_counter() - start
    _log_run(
        experiment="federated_train",
        params={
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "run_id": run_id,
            "seed": seed,
        },
        elapsed=elapsed,
    )

    return {"final_params": final_params, "history": history.__dict__, "run_id": run_id}

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, Strategy
from flwr.server.strategy.aggregate import aggregate
from opacus.accountants.utils import get_noise_multiplier


class AlphaWeightedFedAvg(FedAvg):
    def __init__(self, alpha: float, run_id: Optional[str] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.run_id = run_id
        self.latest_parameters: Optional[List[np.ndarray]] = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}

        items = []
        for client_proxy, fit_res in results:
            try:
                cid = int(client_proxy.cid)
            except Exception:
                cid = client_proxy.cid
            items.append((cid, int(fit_res.num_examples), fit_res))
        items.sort(key=lambda x: (str(type(x[0])), x[0]))

        cids = [it[0] for it in items]
        num_examples_per_client = [it[1] for it in items]
        raw_weights = [float(n) ** self.alpha for n in num_examples_per_client]
        total = sum(raw_weights) if sum(raw_weights) > 0 else 1.0
        normalized_weights = [w / total for w in raw_weights]

        log_path = Path("results") / "diagnostics_alpha_weights.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"[{self.run_id}] round={server_round} alpha={self.alpha} "
                f"cids={cids} num_examples={num_examples_per_client} "
                f"normalized_weights={[round(w, 6) for w in normalized_weights]}\n"
            )

        weighted: List[Tuple[List[np.ndarray], float]] = []
        for _, fit_res in [(it[0], it[2]) for it in items]:
            weighted.append((parameters_to_ndarrays(fit_res.parameters), float(fit_res.num_examples) ** self.alpha))

        aggregated_ndarrays = aggregate(weighted)
        self.latest_parameters = aggregated_ndarrays
        return ndarrays_to_parameters(aggregated_ndarrays), {}


class _GaussianNoiseStrategy(Strategy):
    """Fallback DP wrapper if Flower DP wrapper API is unavailable."""

    def __init__(self, base_strategy: Strategy, noise_multiplier: float, num_clients: int, clip_norm: float = 1.0) -> None:
        self.base_strategy = base_strategy
        self.noise_multiplier = float(noise_multiplier)
        self.num_clients = int(num_clients)
        self.clip_norm = float(clip_norm)

    def initialize_parameters(self, client_manager):
        return self.base_strategy.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        return self.base_strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.base_strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        agg_params, metrics = self.base_strategy.aggregate_fit(server_round, results, failures)
        if agg_params is None:
            return None, metrics
        arrays = parameters_to_ndarrays(agg_params)
        std = self.noise_multiplier * self.clip_norm / max(self.num_clients, 1)
        noisy = [a + np.random.normal(0.0, std, size=a.shape).astype(a.dtype) for a in arrays]
        return ndarrays_to_parameters(noisy), metrics

    def aggregate_evaluate(self, server_round, results, failures):
        return self.base_strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round, parameters):
        return self.base_strategy.evaluate(server_round, parameters)


try:
    from flwr.common import differential_privacy as _flwr_dp

    if not getattr(_flwr_dp, "_zero_norm_guard_applied", False):
        _orig_clip_inputs_inplace = _flwr_dp.clip_inputs_inplace

        def _safe_clip_inputs_inplace(input_arrays, clipping_norm):
            input_norm = _flwr_dp.get_norm(input_arrays)
            if input_norm == 0.0:
                return
            scaling_factor = min(1.0, float(clipping_norm) / float(input_norm))
            for array in input_arrays:
                array *= scaling_factor

        _flwr_dp.clip_inputs_inplace = _safe_clip_inputs_inplace
        _flwr_dp._zero_norm_guard_applied = True
except Exception:
    pass

try:
    from flwr.server.strategy import DifferentialPrivacyServerSideFixedClipping as _DPServerFixed

    class _DPServerFixedWithCapture(_DPServerFixed):
        """Server-side fixed-clipping DP wrapper that records the post-noise
        aggregated parameters.

        Flower's DP wrappers run after the inner strategy's `aggregate_fit`,
        which is where clipping (server-side here) and noise are added. The
        inner `AlphaWeightedFedAvg` only sees pre-noise params, so we shadow
        `aggregate_fit` here to expose the post-noise tensor list as
        `latest_parameters` for `run_federation` to persist.
        """

        def __init__(self, *args, run_id: Optional[str] = None, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.latest_parameters: Optional[List[np.ndarray]] = None
            self.run_id = run_id

        def aggregate_fit(self, server_round, results, failures):
            params, metrics = super().aggregate_fit(server_round, results, failures)
            if params is not None:
                try:
                    self.latest_parameters = parameters_to_ndarrays(params)
                except Exception:
                    pass
            return params, metrics

except Exception:
    _DPServerFixedWithCapture = None  # type: ignore[assignment]


def make_dp_strategy(
    base_strategy: Strategy,
    num_clients: int,
    num_rounds: int,
    target_epsilon: float | None,
    target_delta: float,
    sample_rate: float,
    clip_norm: float = 1.0,
    run_id: Optional[str] = None,
) -> Strategy:
    """Wrap a base strategy with client-side fixed-norm clipping + Gaussian noise.

    If `target_epsilon` is None or +inf, returns the base strategy unchanged
    (no-DP baseline). Otherwise computes the noise multiplier required to
    achieve `(target_epsilon, target_delta)`-DP over `num_rounds` privacy
    epochs with the given `sample_rate`.
    """
    import math

    if target_epsilon is None or math.isinf(float(target_epsilon)):
        return base_strategy

    noise_multiplier = get_noise_multiplier(
        target_epsilon=float(target_epsilon),
        target_delta=float(target_delta),
        sample_rate=float(sample_rate),
        epochs=int(num_rounds),
    )

    log_path = Path("results") / "diagnostics_dp.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"[{run_id}] target_eps={target_epsilon} target_delta={target_delta} "
            f"sample_rate={float(sample_rate):.4f} epochs={int(num_rounds)} "
            f"noise_multiplier={float(noise_multiplier):.4f} clip_norm={float(clip_norm):.4f}\n"
        )

    if float(noise_multiplier) > 50.0:
        print(
            f"WARNING: noise_multiplier={float(noise_multiplier):.2f} is extreme; "
            f"model will be heavily noised at eps={target_epsilon}"
        )

    if _DPServerFixedWithCapture is not None:
        return _DPServerFixedWithCapture(
            strategy=base_strategy,
            noise_multiplier=float(noise_multiplier),
            clipping_norm=float(clip_norm),
            num_sampled_clients=int(num_clients),
            run_id=run_id,
        )

    return _GaussianNoiseStrategy(
        base_strategy=base_strategy,
        noise_multiplier=float(noise_multiplier),
        num_clients=int(num_clients),
        clip_norm=float(clip_norm),
    )

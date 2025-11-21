import numpy as np
from typing import Any, Callable, Optional
from numpy.typing import NDArray
from catsim.utils.rng import NPKey
from joblib import Parallel, delayed
from tqdm import tqdm


def batch_simulate(
        theta: dict[str, NDArray],
        model_callable: Callable[..., tuple[NDArray, NDArray]],
        n_workers: int,
        rng_key: Optional[NPKey] = None,
        parallel_kwargs: Optional[dict[str, Any]] = None
) -> tuple[NDArray, NDArray]:
    theta_np = {key: np.asarray(val) for key, val in theta.items()}

    def _leading_dim(arr: np.ndarray) -> Optional[int]:
        return None if arr.shape == () else arr.shape[0]

    n_simulations = 1
    for arr in theta_np.values():
        leading = _leading_dim(arr)
        if leading is not None:
            n_simulations = leading
            break

    def _slice_param(arr: np.ndarray, idx: int):
        if arr.shape == ():
            return arr
        return arr[idx]

    params_per_sim = [
        {key: _slice_param(arr, idx) for key, arr in theta_np.items()}
        for idx in range(n_simulations)
    ] if n_simulations > 1 else [
        {key: (arr if arr.shape == () else arr) for key, arr in theta_np.items()}
    ]

    sim_keys: list[Optional[NPKey]]
    if rng_key is not None:
        sim_keys = [rng_key.fold_in(idx) for idx in range(n_simulations)]
    else:
        sim_keys = [None] * n_simulations

    if n_simulations == 1:
        kwargs = params_per_sim[0]
        key = sim_keys[0]
        if key is not None:
            kwargs = {**kwargs, 'rng_key': key}
        return model_callable(**kwargs)

    def _run_single(idx: int, key: Optional[NPKey], kwargs: dict[str, NDArray]):
        call_kwargs = dict(kwargs)
        if key is not None:
            call_kwargs['rng_key'] = key
        return idx, model_callable(**call_kwargs)

    parallel_opts = parallel_kwargs or {}
    iterator = Parallel(return_as='generator', n_jobs=n_workers, **parallel_opts)(
        delayed(_run_single)(idx, key, kwargs)
        for idx, (key, kwargs) in enumerate(zip(sim_keys, params_per_sim))
    )

    progress = tqdm(total=n_simulations)

    simulation_outputs: list[tuple[int, tuple[NDArray, NDArray]]] = []
    for idx, result in enumerate(iterator, start=1):
        simulation_outputs.append(result)
        progress.update(1)

    # Restore original ordering; joblib yields results as workers finish.
    simulation_outputs.sort(key=lambda item: item[0])

    progress.close()

    x = np.vstack([output[0] for _, output in simulation_outputs])
    mask = np.vstack([output[1] for _, output in simulation_outputs])
    return x, mask

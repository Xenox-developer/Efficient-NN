import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

from equations import bytes_moved, flops, latency


def fit_latency(s, b, observed):
    if len(observed) < 4:
        raise ValueError

    def unpack(values):
        return dict(zip(
            ["overhead_s", "compute_flops_s", "bandwidth_bytes_s"],
            np.exp(values).tolist(),
        ))

    def residual(values):
        return np.log(latency(s, b, unpack(values)) / observed)

    fits = []
    for compute in [1e12, 1e13]:
        for bandwidth in [5e10, 3e11]:
            fit = least_squares(
                residual,
                np.log([1e-5, compute, bandwidth]),
                bounds=(np.log([1e-12, 1e8, 1e7]), np.log([0.1, 1e16, 1e14])),
                max_nfev=2000,
            )
            if fit.success:
                fits.append(fit)
    if not fits:
        raise RuntimeError
    return unpack(min(fits, key=lambda fit: fit.cost).x)


def fit_energy(s, b, observed, theta):
    if len(observed) < 4:
        raise ValueError
    values = np.column_stack([
        latency(s, b, theta),
        flops(s, b),
        bytes_moved(s, b),
    ])
    values = values / observed[:, None]
    scales = np.linalg.norm(values, axis=0)
    coefficients = nnls(values / scales, np.ones(len(observed)))[0] / scales
    result = dict(zip(
        ["base_power_w", "joules_per_flop", "joules_per_byte"],
        coefficients.tolist(),
    ))
    result["latency"] = theta
    return result


def calibrate(directory):
    data = pd.read_csv(directory / "measurements.csv")
    validation = data.is_validation.astype(str).str.lower().map({
        "true": True, "false": False, "1": True, "0": False,
    })
    if validation.isna().any():
        raise ValueError
    data["is_validation"] = validation
    for column in ["latency", "energy"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    train = data[(data.status == "OK") & ~data.is_validation]

    times = train[np.isfinite(train.latency) & (train.latency > 0)]
    theta = fit_latency(times.S.to_numpy(), times.B.to_numpy(), times.latency.to_numpy())
    energies = train[np.isfinite(train.energy) & (train.energy > 0)]
    theta_energy = fit_energy(
        energies.S.to_numpy(), energies.B.to_numpy(), energies.energy.to_numpy(), theta,
    )

    result = {"latency": theta, "energy": theta_energy}
    (directory / "theta.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    calibrate(parser.parse_args().results)

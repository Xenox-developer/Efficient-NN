import numpy as np


PARAMETERS = 1_040_324
WEIGHT_BYTES = 4 * PARAMETERS


def _inputs(image_size, batch):
    s, b = np.broadcast_arrays(
        np.asarray(image_size, dtype=float),
        np.asarray(batch, dtype=float),
    )
    if (
        np.any(~np.isfinite(s))
        or np.any(~np.isfinite(b))
        or np.any(s <= 0)
        or np.any(s % 16 != 0)
        or np.any(b <= 0)
        or np.any(b % 1 != 0)
    ):
        raise ValueError
    return s, b


def _result(value):
    return float(value) if np.ndim(value) == 0 else value


def layer_costs(image_size, batch):
    s, b = _inputs(image_size, batch)
    x = b * s**2
    operations = np.stack(
        [
            2352*x, 8*x, 18*x, 6400*x, 4*x, 2304*x, 2*x,
            1024*x, 4*x, 4608*x, x, 1024*x, 2*x, 2*x,
            262400*b, 256*b, 51300*b,
        ],
        axis=-1,
    )
    traffic = np.stack(
        [
            44*x + 18816, 64*x, 40*x, 24*x + 204800, 32*x,
            24*x + 294912, 16*x, 24*x + 131072, 32*x,
            20*x + 2359296, 8*x, 12*x + 524288, 16*x,
            8*x + 2048*b, 3072*b + 525312, 2048*b,
            1424*b + 102800,
        ],
        axis=-1,
    )
    return operations, traffic


def flops(image_size, batch):
    s, b = _inputs(image_size, batch)
    return _result(b * (17753*s**2 + 313956))


def memory(image_size, batch):
    s, b = _inputs(image_size, batch)
    return _result(WEIGHT_BYTES + 104*b*s**2 + 3472*b)


def bytes_moved(image_size, batch):
    s, b = _inputs(image_size, batch)
    return _result(WEIGHT_BYTES + 364*b*s**2 + 8592*b)


def latency(image_size, batch, theta):
    overhead = theta["overhead_s"]
    compute = theta["compute_flops_s"]
    bandwidth = theta["bandwidth_bytes_s"]
    if (
        not np.all(np.isfinite([overhead, compute, bandwidth]))
        or overhead < 0
        or compute <= 0
        or bandwidth <= 0
    ):
        raise ValueError
    operations, traffic = layer_costs(image_size, batch)
    times = overhead + np.maximum(operations / compute, traffic / bandwidth)
    return _result(times.sum(axis=-1))


def energy(image_size, batch, theta_energy):
    power = theta_energy["base_power_w"]
    per_flop = theta_energy["joules_per_flop"]
    per_byte = theta_energy["joules_per_byte"]
    if (
        not np.all(np.isfinite([power, per_flop, per_byte]))
        or min(power, per_flop, per_byte) < 0
    ):
        raise ValueError
    time = latency(image_size, batch, theta_energy["latency"])
    return _result(
        power * time
        + per_flop * flops(image_size, batch)
        + per_byte * bytes_moved(image_size, batch)
    )

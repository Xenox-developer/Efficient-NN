import argparse
import csv
import gc
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from equations import memory
from models import SmallCNN


BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
FIELDS = [
    "S", "B", "latency", "memory", "energy", "is_validation", "status",
    "memory_predicted", "free_bytes_before", "total_bytes", "flops_counted",
    "energy_method", "energy_iterations", "energy_elapsed_s",
]


def make_grid(seed=42):
    rng = np.random.default_rng(seed)
    extra_s = rng.choice([s for s in range(32, 513, 16) if s not in BASE_S], 4, replace=False)
    extra_b = rng.choice([b for b in range(1, 257) if b not in BASE_B], 3, replace=False)
    sizes = sorted(BASE_S + extra_s.tolist())
    batches = sorted(BASE_B + extra_b.tolist())
    grid = [(s, b, s not in BASE_S or b not in BASE_B) for s in sizes for b in batches]
    rng.shuffle(grid)
    return grid


@torch.inference_mode()
def count_flops(model, x):
    counts = []

    def record(layer, inputs, output):
        if isinstance(layer, torch.nn.Conv2d):
            counts.append(2 * output.numel() * layer.weight[0].numel())
        elif isinstance(layer, torch.nn.Linear):
            counts.append(output.numel() * (2 * layer.in_features + int(layer.bias is not None)))
        elif isinstance(layer, torch.nn.AdaptiveAvgPool2d):
            counts.append(inputs[0].numel())
        elif isinstance(layer, torch.nn.ReLU):
            counts.append(output.numel())
        elif isinstance(layer, torch.nn.MaxPool2d):
            counts.append(output.numel() * layer.kernel_size**2)

    handles = [layer.register_forward_hook(record) for layer in model.layers]
    try:
        model(x)
    finally:
        for handle in handles:
            handle.remove()
    return sum(counts)


class GPUEnergy:
    def __init__(self):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        try:
            uuid = str(torch.cuda.get_device_properties(0).uuid)
            if not uuid.startswith("GPU-"):
                uuid = "GPU-" + uuid
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            self.uuid = uuid
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.method = "nvml_energy_counter"
        except Exception:
            pynvml.nvmlShutdown()
            raise

    def close(self):
        self.nvml.nvmlShutdown()

    @torch.inference_mode()
    def measure(self, model, x, seconds):
        torch.cuda.synchronize()
        before = self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        start = time.perf_counter()
        count = 0
        end = start
        while count < 3 or end - start < seconds:
            output = model(x)
            torch.cuda.synchronize()
            del output
            count += 1
            end = time.perf_counter()
        after = self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        joules = (after - before) / 1000
        if not np.isfinite(joules) or joules <= 0:
            raise ValueError
        return {
            "energy": float(joules / count),
            "energy_iterations": count,
            "energy_elapsed_s": end - start,
        }


@torch.inference_mode()
def measure_one(model, meter, s, b, args, row):
    x = torch.randn(b, 3, s, s, device="cuda", dtype=torch.float32)
    for _ in range(args.warmup):
        model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    output = model(x)
    torch.cuda.synchronize()
    row["memory"] = torch.cuda.max_memory_allocated()
    del output

    times = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        del output
    row["latency"] = statistics.median(times)
    row["flops_counted"] = count_flops(model, x)
    row["status"] = "OK"
    row.update(meter.measure(model, x, args.energy_seconds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--energy-seconds", type=float, default=3.0)
    args = parser.parse_args()
    if args.seed < 0 or args.seed >= 2**63:
        raise ValueError
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError
    if not np.isfinite(args.energy_seconds) or args.energy_seconds < 1:
        raise ValueError
    if not torch.cuda.is_available():
        raise RuntimeError

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    model = SmallCNN().to(device="cuda", dtype=torch.float32).eval()
    grid = make_grid(args.seed)
    meter = GPUEnergy()
    try:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "measurements.csv"
        completed = set()
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                completed = {(int(row["S"]), int(row["B"])) for row in csv.DictReader(handle)}
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
            handle.flush()
            for s, b, validation in grid:
                if (s, b) in completed:
                    continue
                gc.collect()
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info()
                row = {
                    "S": s,
                    "B": b,
                    "is_validation": validation,
                    "memory_predicted": memory(s, b),
                    "free_bytes_before": free,
                    "total_bytes": total,
                    "energy_method": meter.method,
                }
                try:
                    measure_one(model, meter, s, b, args, row)
                except torch.cuda.OutOfMemoryError:
                    row.update(status="OOM", memory="OOM", latency="", energy="")
                writer.writerow(row)
                handle.flush()
    finally:
        meter.close()


if __name__ == "__main__":
    main()

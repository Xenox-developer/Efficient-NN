# Measured results

Accuracy is measured on the test set using the checkpoint selected by validation accuracy.

| Method | Seeds | Test accuracy, % | SD, pp | Updates | Mean time, min |
|---|---|---:|---:|---:|---:|
| baseline_long | 42,43,44 | 89.87 | 0.25 | 10560 | 84.95 |
| baseline_short | 42,43,44 | 84.00 | 0.27 | 2112 | 17.13 |
| onecycle | 42,43,44 | 89.59 | 0.24 | 2112 | 17.00 |

A budget ratio is not a measured speedup. Time-to-target uses the first epoch-end validation measurement at or above the fixed target.

Hardware: Tesla T4, PyTorch 2.11.0+cu128, FP32.

Paired comparisons:
```json
[
  {
    "seed": 42,
    "test_delta_pp": -0.10999999999999899,
    "budget_ratio": 5.0,
    "time_to_target_ratio": 1.8827833867683554,
    "baseline_hit_target": true,
    "onecycle_hit_target": true
  },
  {
    "seed": 43,
    "test_delta_pp": -0.32999999999999696,
    "budget_ratio": 5.0,
    "time_to_target_ratio": 1.4053309052735214,
    "baseline_hit_target": true,
    "onecycle_hit_target": true
  },
  {
    "seed": 44,
    "test_delta_pp": -0.41000000000001036,
    "budget_ratio": 5.0,
    "time_to_target_ratio": 1.5325760850819163,
    "baseline_hit_target": true,
    "onecycle_hit_target": true
  }
]
```

Interpretation must consider test accuracy differences, missing runs and seed variability.
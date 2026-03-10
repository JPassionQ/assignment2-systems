# Benchmark Results

**配置:** Context: 128 | Backward: True | Device: cuda

**执行命令：** `uv run python benchmark.py --model-size small --context-length 128 --warmup-steps 2 --num-steps 10 --backward`

## Warmup Steps: 5

| Model | Forward Pass | Backward Pass | Total Step |
|-------|--------------|---------------|-----------|
| small | 24.96 ± 0.53 ms | 51.68 ± 0.79 ms | 76.64 ± 0.70 ms |
| medium | 80.02 ± 1.52 ms | 173.43 ± 4.12 ms | 253.45 ± 5.00 ms |
| large | 261.45 ± 145.04 ms | 867.42 ± 94.73 ms | 1128.87 ± 145.87 ms |

## Warmup Steps: 0

| Model | Forward Pass | Backward Pass | Total Step |
|-------|--------------|---------------|-----------|
| small | 62.81 ± 113.05 ms | 62.80 ± 32.22 ms | 125.61 ± 145.16 ms |
| medium | 102.96 ± 82.34 ms | 173.11 ± 22.81 ms | 276.08 ± 105.10 ms |
| large | 425.54 ± 349.58 ms | 2166.96 ± 1959.79 ms | 2592.50 ± 2193.59 ms |

## Warmup Steps: 2

| Model | Forward Pass | Backward Pass | Total Step |
|-------|--------------|---------------|-----------|
| small | 28.27 ± 5.32 ms | 55.00 ± 2.47 ms | 83.27 ± 5.83 ms |
| medium | 81.60 ± 3.63 ms | 174.26 ± 6.10 ms | 255.86 ± 8.90 ms |
| large | 211.66 ± 15.82 ms | 858.96 ± 70.77 ms | 1070.62 ± 72.16 ms |

## Warmup Steps: 5 (Mixed Precision)

| Model | Forward Pass | Backward Pass | Total Step |
|-------|--------------|---------------|-----------|
| small | 18.75 ± 0.83 ms | 30.83 ± 0.79 ms | 49.57 ± 1.18 ms |
| medium | 43.27 ± 1.67 ms | 85.96 ± 1.23 ms | 129.23 ± 1.91 ms |
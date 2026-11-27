| Configuration | Total ms | FPS | CPU % | RAM MB |
|---|---|---|---|---|
| full (align+CLAHE+edges) | 2030.8 | 0.49 | 112 | 120 |
| no alignment | 1499.1 | 0.67 | 114 | 119 |
| no CLAHE | 1775.8 | 0.56 | 113 | 119 |
| no edge-diff | 1725.2 | 0.58 | 115 | 120 |
| fast (no align/CLAHE/edges) | 939.0 | 1.06 | 120 | 120 |
| fast @320x240 | 344.1 | 2.90 | 102 | 117 |

Per-stage mean latency (ms), full configuration:

| Stage | ms |
|---|---|
| preprocess | 262.66 |
| align | 600.60 |
| diff_threshold | 97.81 |
| morphology | 39.52 |
| components | 1012.80 |
| decision | 2.47 |
| annotate | 14.92 |
| total | 2030.79 |

# 2-minute demo video script (1080p, 16:9, MP4)

| Time | Show | Say |
|---|---|---|
| 0:00-0:15 | Title slide + printed sheets | Problem: undetected print defects waste paper in high-speed printing. Name, programme, project title. |
| 0:15-0:35 | Board + camera + paper setup | Zynq-7000, ARM Cortex-A9 only, camera. Reference-based, no ML training, no dataset. |
| 0:35-0:50 | Architecture diagram | Pipeline stages; reference is preprocessed once. |
| 0:50-1:40 | Terminal live run: reference, a normal sheet (PASS), then 2-3 defective sheets (DEFECT) with annotated images | Point out boxes, defect %, verdict and per-stage timing printed in the terminal. |
| 1:40-1:55 | Benchmark table + accuracy numbers | Latency, FPS, CPU/RAM, precision/recall. |
| 1:55-2:00 | Repo + report link | Future work: PL acceleration. |

Tips: record the terminal at 1920x1080, keep font large, run commands from `src/`, trim to under 2:00.

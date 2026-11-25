# Live Camera-Based Printing Defect Detection: Normal Reference Capture and Inspection Using the PYNQ-Z2 FPGA Platform

**Author:** Manoj A (NDU202500375), M.Tech Embedded Systems, NIELIT Calicut

A lightweight, **training-free** inspection system that compares each newly printed sheet against a
golden reference image and flags abnormal regions (ink spots, missing ink, streaks, blurred text,
ink spreading). It runs entirely on the **ARM Cortex-A9** processing system of a Zynq-7000 board
using Python + OpenCV. No dataset collection or ML training is needed; when a new print design is
introduced, only the reference image is replaced.

> This submission is **CPU-only (ARM Cortex-A9)**. FPGA (PL) acceleration with Vitis HLS is planned
> future work for the dissertation and is not part of this entry.

## Why reference-based?
- No large labelled dataset, no training time, no model to deploy on a small embedded CPU.
- Reference is preprocessed **once** and cached, so per-sheet work is minimal.
- New design = swap one reference image.

## Pipeline
![architecture](docs/architecture_diagram.png)

1. Resize to 640x480, grayscale, CLAHE, Gaussian blur
2. ECC (Euclidean) alignment to tolerate small camera/paper shifts
3. Absolute difference + Sobel edge difference (catches blur and streaks)
4. Threshold = max(noise floor, Otsu)
5. Morphological open/close to remove speckle
6. Connected components, small-blob filter
7. Defect-area percentage, PASS/DEFECT decision, annotated output

## Repository layout
```
src/defect_detector.py   pipeline (CLI + importable)
src/live_demo.py        real-time webcam / video inspection (r = set reference)
src/benchmark.py         per-stage latency, FPS, CPU %, RAM, optimisation ablation
src/evaluate.py          TP/FP/FN/TN, precision, recall, F1 from labels.csv
dataset/reference/       golden reference image (ref.png)
dataset/test_samples/    defective + normal samples, labels.csv
dataset/outputs/         difference, mask, annotated results, benchmark/accuracy summaries
docs/                    final report PDF, architecture diagram, demo script
```

## Setup (on the board / any Linux)
```bash
pip3 install -r requirements.txt
```

## Run
```bash
cd src
# single image
python3 defect_detector.py --ref ../dataset/reference/ref.png --test ../dataset/test_samples/ink_spot_01.png

# accuracy
python3 evaluate.py --ref ../dataset/reference/ref.png --dir ../dataset/test_samples --labels ../dataset/test_samples/labels.csv

# performance on the A9
python3 benchmark.py --ref ../dataset/reference/ref.png --dir ../dataset/test_samples --repeats 20
```
Useful flags for `defect_detector.py`: `--no-align`, `--no-edges`, `--no-clahe`.

## Results (measured on ARM Cortex-A9)
Fill these from `dataset/outputs/benchmark_summary.md` and `accuracy_summary.txt`.

| Metric | Value |
|---|---|
| Board / A9 clock | __ |
| Resolution | 640x480 |
| Latency (full pipeline) | __ ms |
| Throughput | __ FPS |
| CPU load / RAM | __ % / __ MB |
| Precision / Recall / F1 | __ / __ / __ |

## Optimisations
Cached reference preprocessing, half-resolution ECC alignment, in-place OpenCV ops, fixed small
kernels, optional stages (`--no-align/--no-clahe/--no-edges`), reduced resolution mode. The
benchmark script reports each configuration's latency so trade-offs are measured, not assumed.

## Limitations / future work
Sensitive to strong lighting changes and large misalignment; image-level accuracy only.
Future: Vitis HLS acceleration of Gaussian, subtraction, threshold, Sobel and morphology on the PL.

## Links
Demo video: __  |  Report: `docs/Final_Project_Report.pdf`

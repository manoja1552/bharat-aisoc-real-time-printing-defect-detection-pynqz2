"""
CPU benchmark: per-stage latency, FPS, CPU load and RAM on the ARM A9.

  python3 benchmark.py --ref ../dataset/reference/ref.png \
      --dir ../dataset/test_samples --repeats 20 --out ../dataset/outputs

Writes benchmark_results.csv and benchmark_summary.md (copy numbers into the report).
Also compares configurations (optimisation ablation).
"""
import argparse
import csv
import glob
import os
import statistics as st
import time

import cv2
import psutil

from defect_detector import Params, Reference, inspect

CONFIGS = {
    "full (align+CLAHE+edges)": dict(),
    "no alignment": dict(align=False),
    "no CLAHE": dict(clahe=False),
    "no edge-diff": dict(edges=False),
    "fast (no align/CLAHE/edges)": dict(align=False, clahe=False, edges=False),
    "fast @320x240": dict(align=False, clahe=False, edges=False, width=320, height=240),
}
STAGES = ["preprocess", "align", "diff_threshold", "morphology", "components", "decision", "annotate", "total"]


def run_config(name, kw, ref_path, imgs, repeats):
    p = Params(**kw)
    ref = Reference(ref_path, p)
    proc = psutil.Process(os.getpid())
    inspect(ref, imgs[0], annotate=False)  # warm-up
    proc.cpu_percent(None)
    per = {s: [] for s in STAGES}
    wall0 = time.perf_counter()
    n = 0
    for _ in range(repeats):
        for im in imgs:
            r, _ = inspect(ref, im, annotate=True)
            for s in STAGES:
                per[s].append(r["timing_ms"][s])
            n += 1
    wall = time.perf_counter() - wall0
    cpu = proc.cpu_percent(None)
    rss = proc.memory_info().rss / 1e6
    return dict(config=name, frames=n, fps=n / wall, cpu_pct=cpu, rss_mb=rss,
                **{f"{s}_ms": st.mean(per[s]) for s in STAGES})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default="../dataset/outputs")
    a = ap.parse_args()

    files = sorted(sum([glob.glob(os.path.join(a.dir, e)) for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp")], []))
    imgs = [cv2.imread(f) for f in files]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        raise SystemExit("No test images found")

    rows = [run_config(n, kw, a.ref, imgs, a.repeats) for n, kw in CONFIGS.items()]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "benchmark_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = ["| Configuration | Total ms | FPS | CPU % | RAM MB |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['config']} | {r['total_ms']:.1f} | {r['fps']:.2f} | {r['cpu_pct']:.0f} | {r['rss_mb']:.0f} |")
    md += ["", "Per-stage mean latency (ms), full configuration:", "", "| Stage | ms |", "|---|---|"]
    md += [f"| {s} | {rows[0][s + '_ms']:.2f} |" for s in STAGES]
    text = "\n".join(md)
    open(os.path.join(a.out, "benchmark_summary.md"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

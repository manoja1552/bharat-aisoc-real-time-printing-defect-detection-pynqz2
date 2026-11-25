"""
Reference-based printing defect detector (CPU-only, ARM Cortex-A9 friendly).

Pipeline (each stage is timed):
  resize -> grayscale -> CLAHE -> Gaussian blur -> ECC alignment
  -> abs-difference (+ Sobel edge difference) -> threshold (max(floor, Otsu))
  -> morphology -> connected components -> area filter
  -> defect % + PASS/DEFECT decision -> annotated output

The reference is preprocessed ONCE (Reference class) and reused for every
inspected sheet, which is the main run-time optimisation.

Usage:
  python3 defect_detector.py --ref ../dataset/reference/ref.png \
      --test ../dataset/test_samples/ink_spot_01.png --out ../dataset/outputs
"""
import argparse
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Params:
    width: int = 640
    height: int = 480
    blur: int = 5            # Gaussian kernel (odd)
    clahe: bool = True
    align: bool = True       # ECC registration (handles small shifts/rotation)
    edges: bool = True       # add Sobel edge-difference (helps blur/streaks)
    thr_floor: int = 30      # minimum abs-diff threshold (noise floor)
    edge_thr: int = 60       # threshold on edge-magnitude difference
    min_area: int = 30       # ignore blobs smaller than this (pixels)
    min_pct: float = 0.005   # min defective area (% of image) to flag DEFECT


def _sobel_mag(gray):
    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    return cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0)


class Reference:
    """Golden reference, preprocessed once and cached."""

    def __init__(self, img_or_path, p: Params):
        img = cv2.imread(img_or_path) if isinstance(img_or_path, str) else img_or_path
        if img is None:
            raise FileNotFoundError(f"Cannot read reference: {img_or_path}")
        self.p = p
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if p.clahe else None
        self.color = cv2.resize(img, (p.width, p.height), interpolation=cv2.INTER_AREA)
        self.gray = self._gray(self.color)
        self.edge = _sobel_mag(self.gray) if p.edges else None
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def _gray(self, color):
        g = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        if self.clahe is not None:
            g = self.clahe.apply(g)
        return cv2.GaussianBlur(g, (self.p.blur, self.p.blur), 0)


def _align(ref_gray, mov_gray, mov_color):
    """ECC Euclidean alignment on half-resolution images (fast). Falls back to identity."""
    h, w = ref_gray.shape
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    try:
        r = cv2.resize(ref_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        m = cv2.resize(mov_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        _, warp = cv2.findTransformECC(r, m, warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
        warp[:, 2] *= 2.0
        flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        g = cv2.warpAffine(mov_gray, warp, (w, h), flags=flags, borderMode=cv2.BORDER_REPLICATE)
        c = cv2.warpAffine(mov_color, warp, (w, h), flags=flags, borderMode=cv2.BORDER_REPLICATE)
        return g, c
    except cv2.error:
        return mov_gray, mov_color


def inspect(ref: Reference, test_img, annotate=True):
    """Compare one captured image against the reference. Returns (result_dict, images_dict)."""
    p = ref.p
    t = {}
    t0 = time.perf_counter()

    def lap(name, start):
        now = time.perf_counter()
        t[name] = (now - start) * 1000.0
        return now

    s = t0
    color = cv2.resize(test_img, (p.width, p.height), interpolation=cv2.INTER_AREA)
    gray = ref._gray(color)
    s = lap("preprocess", s)

    if p.align:
        gray, color = _align(ref.gray, gray, color)
    s = lap("align", s)

    diff = cv2.absdiff(ref.gray, gray)
    thr = max(p.thr_floor, int(cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]))
    _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    if p.edges:
        ediff = cv2.absdiff(ref.edge, _sobel_mag(gray))
        _, emask = cv2.threshold(ediff, p.edge_thr, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(mask, emask)
    s = lap("diff_threshold", s)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ref.kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ref.kernel_close)
    s = lap("morphology", s)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= p.min_area]
    boxes = [tuple(int(v) for v in stats[i]) for i in keep]
    clean = np.zeros_like(mask)
    for i in keep:
        clean[labels == i] = 255
    s = lap("components", s)

    defect_px = int(cv2.countNonZero(clean))
    pct = 100.0 * defect_px / clean.size
    verdict = "DEFECT" if (len(boxes) > 0 and pct >= p.min_pct) else "PASS"
    s = lap("decision", s)

    out = None
    if annotate:
        out = color.copy()
        for (x, y, w, h, a) in boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(out, f"{verdict}  {pct:.3f}%  blobs={len(boxes)}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if verdict == "DEFECT" else (0, 160, 0), 2)
    lap("annotate", s)
    t["total"] = (time.perf_counter() - t0) * 1000.0

    result = dict(verdict=verdict, defect_pct=pct, blobs=len(boxes), threshold=thr,
                  boxes=boxes, timing_ms=t)
    return result, dict(diff=diff, mask=clean, annotated=out)


def main():
    ap = argparse.ArgumentParser(description="Reference-based printing defect detector")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="../dataset/outputs")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--no-clahe", action="store_true")
    a = ap.parse_args()

    p = Params(align=not a.no_align, edges=not a.no_edges, clahe=not a.no_clahe)
    ref = Reference(a.ref, p)
    test = cv2.imread(a.test)
    if test is None:
        raise SystemExit(f"Cannot read test image: {a.test}")
    res, imgs = inspect(ref, test)

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.test))[0]
    cv2.imwrite(os.path.join(a.out, f"{stem}_difference.png"), imgs["diff"])
    cv2.imwrite(os.path.join(a.out, f"{stem}_mask.png"), imgs["mask"])
    cv2.imwrite(os.path.join(a.out, f"{stem}_annotated.png"), imgs["annotated"])

    print(f"{stem}: {res['verdict']}  defect={res['defect_pct']:.3f}%  blobs={res['blobs']}  thr={res['threshold']}")
    for k, v in res["timing_ms"].items():
        print(f"  {k:15s} {v:8.2f} ms")


if __name__ == "__main__":
    main()

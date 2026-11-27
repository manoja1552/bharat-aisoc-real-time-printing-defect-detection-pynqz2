"""
Reference-based printing defect detector.
CPU-only implementation for ARM Cortex-A9 / PYNQ-Z2.

Pipeline:
resize -> grayscale -> illumination normalization
-> CLAHE -> Gaussian blur
-> ECC alignment
-> abs-difference + Sobel edge difference
-> threshold -> morphology
-> connected components -> PASS/DEFECT
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
    blur: int = 5
    clahe: bool = True
    align: bool = True
    edges: bool = True
    thr_floor: int = 30
    edge_thr: int = 60
    min_area: int = 30
    min_pct: float = 0.005


def _sobel_mag(gray):

    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)

    return cv2.addWeighted(
        cv2.convertScaleAbs(gx),
        0.5,
        cv2.convertScaleAbs(gy),
        0.5,
        0
    )


class Reference:

    def __init__(self, img_or_path, p: Params):

        img = (
            cv2.imread(img_or_path)
            if isinstance(img_or_path, str)
            else img_or_path
        )

        if img is None:
            raise FileNotFoundError(
                f"Cannot read reference: {img_or_path}"
            )

        self.p = p

        self.clahe = (
            cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )
            if p.clahe else None
        )

        self.color = cv2.resize(
            img,
            (p.width, p.height),
            interpolation=cv2.INTER_AREA
        )

        self.gray = self._gray(self.color)

        self.edge = (
            _sobel_mag(self.gray)
            if p.edges else None
        )

        self.kernel_open = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3)
        )

        self.kernel_close = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5)
        )

    def _gray(self, color):

        # Convert BGR to grayscale
        g = cv2.cvtColor(
            color,
            cv2.COLOR_BGR2GRAY
        )

        # Estimate illumination at reduced resolution
        small = cv2.resize(
            g,
            (
                self.p.width // 4,
                self.p.height // 4
            ),
            interpolation=cv2.INTER_AREA
        )

        # Smooth illumination background
        background_small = cv2.GaussianBlur(
            small,
            (0, 0),
            6
        )

        # Resize background
        background = cv2.resize(
            background_small,
            (self.p.width, self.p.height),
            interpolation=cv2.INTER_LINEAR
        )

        # Illumination normalization
        normalized = cv2.divide(
            g,
            background,
            scale=128
        )

        # CLAHE contrast enhancement
        if self.clahe is not None:
            normalized = self.clahe.apply(normalized)

        # Noise reduction
        result = cv2.GaussianBlur(
            normalized,
            (self.p.blur, self.p.blur),
            0
        )

        return result


def _align(ref_gray, mov_gray, mov_color):

    h, w = ref_gray.shape

    warp = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        50,
        1e-4
    )

    try:

        # Half-resolution alignment
        ref_small = cv2.resize(
            ref_gray,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA
        )

        mov_small = cv2.resize(
            mov_gray,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA
        )

        _, warp = cv2.findTransformECC(
            ref_small,
            mov_small,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            5
        )

        # Convert translation to full resolution
        warp[:, 2] *= 2.0

        flags = (
            cv2.INTER_LINEAR |
            cv2.WARP_INVERSE_MAP
        )

        aligned_gray = cv2.warpAffine(
            mov_gray,
            warp,
            (w, h),
            flags=flags,
            borderMode=cv2.BORDER_REPLICATE
        )

        aligned_color = cv2.warpAffine(
            mov_color,
            warp,
            (w, h),
            flags=flags,
            borderMode=cv2.BORDER_REPLICATE
        )

        return aligned_gray, aligned_color

    except cv2.error:

        return mov_gray, mov_color


def inspect(ref: Reference, test_img, annotate=True):

    p = ref.p

    timings = {}

    t0 = time.perf_counter()

    def lap(name, start):

        now = time.perf_counter()

        timings[name] = (
            now - start
        ) * 1000.0

        return now

    s = t0

    # Resize input
    color = cv2.resize(
        test_img,
        (p.width, p.height),
        interpolation=cv2.INTER_AREA
    )

    # Preprocessing
    gray = ref._gray(color)

    s = lap("preprocess", s)

    # Image alignment
    if p.align:

        gray, color = _align(
            ref.gray,
            gray,
            color
        )

    s = lap("align", s)

    # Absolute difference
    diff = cv2.absdiff(
        ref.gray,
        gray
    )

    # Otsu adaptive threshold
    otsu_thr = int(
        cv2.threshold(
            diff,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[0]
    )

    thr = max(
        p.thr_floor,
        otsu_thr
    )

    _, mask = cv2.threshold(
        diff,
        thr,
        255,
        cv2.THRESH_BINARY
    )

    # Sobel edge difference
    if p.edges:

        edge_current = _sobel_mag(gray)

        edge_diff = cv2.absdiff(
            ref.edge,
            edge_current
        )

        _, edge_mask = cv2.threshold(
            edge_diff,
            p.edge_thr,
            255,
            cv2.THRESH_BINARY
        )

        mask = cv2.bitwise_or(
            mask,
            edge_mask
        )

    s = lap("diff_threshold", s)

    # Morphological opening
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        ref.kernel_open
    )

    # Morphological closing
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        ref.kernel_close
    )

    s = lap("morphology", s)

    # Connected component analysis
    n, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    # Efficient component filtering
    keep = np.where(
        stats[1:, cv2.CC_STAT_AREA] >= p.min_area
    )[0] + 1

    # Extract bounding boxes
    boxes = [
        tuple(int(v) for v in stats[i])
        for i in keep
    ]

    # Generate filtered mask using lookup table
    keep_labels = np.zeros(
        n,
        dtype=np.uint8
    )

    keep_labels[keep] = 255

    clean = keep_labels[labels]

    s = lap("components", s)

    # Calculate defect percentage
    defect_px = int(
        cv2.countNonZero(clean)
    )

    pct = (
        100.0 * defect_px / clean.size
    )

    # PASS / DEFECT decision
    verdict = (
        "DEFECT"
        if (
            len(boxes) > 0 and
            pct >= p.min_pct
        )
        else "PASS"
    )

    s = lap("decision", s)

    # Annotated output
    output = None

    if annotate:

        output = color.copy()

        for (x, y, w, h, area) in boxes:

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

        cv2.putText(
            output,
            f"{verdict} {pct:.3f}% blobs={len(boxes)}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255)
            if verdict == "DEFECT"
            else (0, 160, 0),
            2
        )

    lap("annotate", s)

    timings["total"] = (
        time.perf_counter() - t0
    ) * 1000.0

    result = {
        "verdict": verdict,
        "defect_pct": pct,
        "blobs": len(boxes),
        "threshold": thr,
        "boxes": boxes,
        "timing_ms": timings
    }

    images = {
        "diff": diff,
        "mask": clean,
        "annotated": output
    }

    return result, images


def main():

    parser = argparse.ArgumentParser(
        description="Reference-based printing defect detector"
    )

    parser.add_argument(
        "--ref",
        required=True
    )

    parser.add_argument(
        "--test",
        required=True
    )

    parser.add_argument(
        "--out",
        default="../dataset/outputs"
    )

    parser.add_argument(
        "--no-align",
        action="store_true"
    )

    parser.add_argument(
        "--no-edges",
        action="store_true"
    )

    parser.add_argument(
        "--no-clahe",
        action="store_true"
    )

    args = parser.parse_args()

    # Detector configuration
    p = Params(
        align=not args.no_align,
        edges=not args.no_edges,
        clahe=not args.no_clahe
    )

    # Load reference
    ref = Reference(
        args.ref,
        p
    )

    # Load test image
    test = cv2.imread(args.test)

    if test is None:
        raise SystemExit(
            f"Cannot read test image: {args.test}"
        )

    # Run detection
    result, images = inspect(
        ref,
        test
    )

    # Create output directory
    os.makedirs(
        args.out,
        exist_ok=True
    )

    stem = os.path.splitext(
        os.path.basename(args.test)
    )[0]

    # Save difference image
    cv2.imwrite(
        os.path.join(
            args.out,
            f"{stem}_difference.png"
        ),
        images["diff"]
    )

    # Save defect mask
    cv2.imwrite(
        os.path.join(
            args.out,
            f"{stem}_mask.png"
        ),
        images["mask"]
    )

    # Save annotated image
    cv2.imwrite(
        os.path.join(
            args.out,
            f"{stem}_annotated.png"
        ),
        images["annotated"]
    )

    # Print detection result
    print(
        f"{stem}: {result['verdict']} "
        f"defect={result['defect_pct']:.3f}% "
        f"blobs={result['blobs']} "
        f"thr={result['threshold']}"
    )

    # Print processing time
    for key, value in result["timing_ms"].items():

        print(
            f"  {key:15s} {value:8.2f} ms"
        )


if __name__ == "__main__":
    main()

"""
Accuracy evaluation (image-level): TP/FP/FN/TN, precision, recall, F1.

labels.csv format (one row per image in the folder, include normal samples):
    filename,label,defect_type
    ink_spot_01.png,1,ink_spot
    normal_01.png,0,none

  python3 evaluate.py --ref ../dataset/reference/ref.png --dir ../dataset/test_samples \
      --labels ../dataset/test_samples/labels.csv --out ../dataset/outputs
"""
import argparse
import csv
import os
from collections import defaultdict

import cv2

from defect_detector import Params, Reference, inspect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="../dataset/outputs")
    a = ap.parse_args()

    ref = Reference(a.ref, Params())
    tp = fp = fn = tn = 0
    per_type = defaultdict(lambda: [0, 0])  # type -> [detected, total]
    rows = []
    os.makedirs(a.out, exist_ok=True)

    with open(a.labels) as f:
        for row in csv.DictReader(f):
            img = cv2.imread(os.path.join(a.dir, row["filename"]))
            if img is None:
                print("skip (unreadable):", row["filename"])
                continue
            gt = int(row["label"])
            res, imgs = inspect(ref, img)
            pred = 1 if res["verdict"] == "DEFECT" else 0
            tp += int(gt == 1 and pred == 1)
            fp += int(gt == 0 and pred == 1)
            fn += int(gt == 1 and pred == 0)
            tn += int(gt == 0 and pred == 0)
            if gt == 1:
                dt = row.get("defect_type", "defect")
                per_type[dt][1] += 1
                per_type[dt][0] += pred
            stem = os.path.splitext(row["filename"])[0]
            cv2.imwrite(os.path.join(a.out, f"{stem}_annotated.png"), imgs["annotated"])
            rows.append((row["filename"], gt, pred, res["defect_pct"], res["blobs"]))
            print(f"{row['filename']:28s} gt={gt} pred={pred} defect%={res['defect_pct']:.3f} blobs={res['blobs']}")

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    lines = [f"TP={tp} FP={fp} FN={fn} TN={tn}",
             f"Accuracy={acc:.3f} Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}", "",
             "Detection rate per defect type:"]
    lines += [f"  {k}: {v[0]}/{v[1]}" for k, v in per_type.items()]
    text = "\n".join(lines)
    print("\n" + text)
    open(os.path.join(a.out, "accuracy_summary.txt"), "w").write(text + "\n")
    with open(os.path.join(a.out, "accuracy_per_image.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "pred", "defect_pct", "blobs"])
        w.writerows(rows)


if __name__ == "__main__":
    main()

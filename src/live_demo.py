"""
Real-time inspection from a webcam (USB/V4L2) or a video file.

  python3 live_demo.py --source 0            # webcam index 0
  python3 live_demo.py --source clip.mp4     # video file (for testing)
  python3 live_demo.py --source 0 --headless # no display (SSH to board); prints verdicts

Keys (display mode):  r = capture current frame as REFERENCE
                      s = save current annotated frame to ../dataset/outputs
                      q = quit
Tips: fix the camera and lighting, lock focus/exposure, put the paper in the same place.
"""
import argparse
import os
import time

import cv2

from defect_detector import Params, Reference, inspect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--ref", default=None, help="optional reference image; otherwise press 'r'")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (headless testing)")
    ap.add_argument("--out", default="../dataset/outputs")
    a = ap.parse_args()

    p = Params(align=not a.no_align, clahe=not a.no_clahe, edges=not a.no_edges)
    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {a.source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # keep latency low

    ref = Reference(a.ref, p) if a.ref else None
    os.makedirs(a.out, exist_ok=True)
    fps, last, n = 0.0, time.perf_counter(), 0
    if ref is None:
        print("No reference yet: show a good print and press 'r' (or pass --ref).")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        view, verdict = frame, "NO REFERENCE"
        if ref is not None:
            res, imgs = inspect(ref, frame)
            view, verdict = imgs["annotated"], res["verdict"]
            if a.headless:
                print(f"frame {n}: {verdict} {res['defect_pct']:.3f}% blobs={res['blobs']} {res['timing_ms']['total']:.1f} ms")
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if n > 1 else 0.0
        last = now

        if not a.headless:
            show = cv2.resize(view, (640, 480))
            cv2.putText(show, f"FPS {fps:.1f}", (8, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            if ref is None:
                cv2.putText(show, "Press 'r' to set reference", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.imshow("Printing defect inspection (r=ref, s=save, q=quit)", show)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("r"):
                ref = Reference(frame.copy(), p)
                print("Reference captured.")
            if k == ord("s"):
                path = os.path.join(a.out, f"live_{int(time.time())}_{verdict}.png")
                cv2.imwrite(path, show)
                print("saved", path)
        if a.max_frames and n >= a.max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

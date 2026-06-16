import argparse
import os
import time
from collections import deque
from datetime import datetime

import cv2


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Motion-sensitive CCTV using laptop camera")
    p.add_argument("--min-area", type=int, default=20,
                   help="Min contour area to count as motion (default 20, tuned for roaches)")
    p.add_argument("--max-area", type=int, default=8000,
                   help="Max contour area — filters out large blobs like shadows or people")
    p.add_argument("--threshold", type=int, default=15,
                   help="Pixel diff threshold 0-255 (lower = more sensitive to subtle contrast)")
    p.add_argument("--cooldown", type=float, default=5.0,
                   help="Seconds between burst triggers")
    p.add_argument("--pre-frames", type=int, default=3,
                   help="Frames saved from buffer before motion trigger")
    p.add_argument("--post-frames", type=int, default=5,
                   help="Frames saved after motion trigger")
    p.add_argument("--save-dir", default="captures",
                   help="Directory to save motion bursts")
    p.add_argument("--display", action="store_true",
                   help="Show live video window with motion highlighted (useful for tuning)")
    p.add_argument("--max-contours", type=int, default=10,
                   help="Max number of motion contours before the frame is ignored (person vs roach)")
    p.add_argument("--calibrate", action="store_true",
                   help="Calibration mode: press Enter to start a 10-second area recording session")
    return p.parse_args()


def draw_boxes(frame, contours):
    out = frame.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return out


def calibrate(cap, args, background):
    DURATION = 10
    input(f"\nPosition your subject, then press ENTER to start {DURATION}-second recording...")
    print("Recording...\n")

    area_samples = []
    count_samples = []
    deadline = time.time() + DURATION

    while time.time() < deadline:
        remaining = deadline - time.time()
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        cv2.accumulateWeighted(gray, background, 0.2)
        bg = cv2.convertScaleAbs(background)

        diff = cv2.absdiff(bg, gray)
        _, thresh = cv2.threshold(diff, args.threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = sorted([cv2.contourArea(c) for c in contours
                        if args.min_area < cv2.contourArea(c) < args.max_area], reverse=True)

        if areas:
            area_samples.append(areas[0])
            count_samples.append(len(areas))
            print(f"  {remaining:4.1f}s  contours: {len(areas):>3}  largest: {int(areas[0]):>7}"
                  f"  (all: {[int(a) for a in areas[:5]]})")

    if not area_samples:
        print("\nNo motion detected within min/max-area bounds. Try --min-area 5 --max-area 999999.")
        return

    area_samples.sort()
    count_samples.sort()
    median_count = count_samples[len(count_samples) // 2]
    max_count = count_samples[-1]

    print(f"\n--- Calibration results (threshold={args.threshold}, "
          f"min-area={args.min_area}, max-area={args.max_area}) ---")
    print(f"  Contours per frame : min={count_samples[0]}  median={median_count}  max={max_count}")
    print(f"  Largest area seen  : {int(area_samples[-1])}")
    print(f"  Smallest area seen : {int(area_samples[0])}")
    print(f"\n  KEY METRIC -> use --max-contours between your subject's count and people's count")


def main():
    args = parse_args()
    ensure_dir(args.save_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera. Check System Settings -> Privacy & Security.")
        return

    time.sleep(0.5)
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read from camera.")
        cap.release()
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    background = gray.copy().astype("float")

    if args.calibrate:
        calibrate(cap, args, background)
        cap.release()
        return

    frame_buffer = deque(maxlen=args.pre_frames)
    last_trigger = 0
    post_remaining = 0
    burst_dir = None
    burst_idx = 0

    print("Monitoring started. Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            cv2.accumulateWeighted(gray, background, 0.2)
            bg = cv2.convertScaleAbs(background)

            diff = cv2.absdiff(bg, gray)
            _, thresh = cv2.threshold(diff, args.threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion = [c for c in contours if args.min_area < cv2.contourArea(c) < args.max_area]

            now = time.time()

            if motion and len(motion) <= args.max_contours and (now - last_trigger) > args.cooldown:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                burst_dir = os.path.join(args.save_dir, f"burst_{timestamp}")
                ensure_dir(burst_dir)
                print(f"Motion detected! Saving burst -> {burst_dir}/")

                for i, pre in enumerate(frame_buffer):
                    cv2.imwrite(os.path.join(burst_dir, f"pre_{i:02d}.jpg"), pre)

                burst_idx = 0
                post_remaining = args.post_frames
                last_trigger = now

            if post_remaining > 0:
                annotated = draw_boxes(frame, motion)
                cv2.imwrite(os.path.join(burst_dir, f"motion_{burst_idx:02d}.jpg"), annotated)
                burst_idx += 1
                post_remaining -= 1

            frame_buffer.append(frame.copy())

            if args.display:
                disp = draw_boxes(frame, motion) if motion else frame.copy()
                area = sum(cv2.contourArea(c) for c in motion)
                cv2.putText(disp, f"Motion area: {int(area)}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("RoachCam", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("Stopped by user.")

    cap.release()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import argparse
import os
import time
from collections import deque
from datetime import datetime

import cv2


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Camera abstraction — tries picamera2 first, falls back to cv2.VideoCapture
# ---------------------------------------------------------------------------

class PiCamera2Capture:
    def __init__(self, width, height, fps):
        from picamera2 import Picamera2
        self._cam = Picamera2()
        cfg = self._cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self._cam.configure(cfg)
        self._cam.start()

    def read(self):
        frame = self._cam.capture_array()
        return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def release(self):
        self._cam.stop()


class CV2Capture:
    def __init__(self, width, height, fps):
        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

    def isOpened(self):
        return self._cap.isOpened()

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()


def open_camera(width, height, fps):
    try:
        cam = PiCamera2Capture(width, height, fps)
        print("Using picamera2.")
        return cam
    except Exception:
        pass
    cam = CV2Capture(width, height, fps)
    if not cam.isOpened():
        return None
    print("Using cv2.VideoCapture.")
    return cam


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Motion-sensitive CCTV for Raspberry Pi camera module")
    p.add_argument("--min-area", type=int, default=20,
                   help="Min contour area to count as motion")
    p.add_argument("--max-area", type=int, default=8000,
                   help="Max contour area — filters large blobs")
    p.add_argument("--threshold", type=int, default=15,
                   help="Pixel diff threshold 0-255")
    p.add_argument("--cooldown", type=float, default=5.0,
                   help="Seconds between burst triggers")
    p.add_argument("--pre-frames", type=int, default=3,
                   help="Frames buffered before motion trigger")
    p.add_argument("--post-frames", type=int, default=5,
                   help="Frames saved after motion trigger")
    p.add_argument("--save-dir", default="captures",
                   help="Directory to save motion bursts")
    p.add_argument("--max-contours", type=int, default=10,
                   help="Max contours before frame is ignored (person vs roach)")
    p.add_argument("--width", type=int, default=640,
                   help="Capture width in pixels (lower = faster on Pi Zero)")
    p.add_argument("--height", type=int, default=480,
                   help="Capture height in pixels")
    p.add_argument("--fps", type=int, default=15,
                   help="Target capture frame rate")
    p.add_argument("--process-every", type=int, default=2,
                   help="Only run motion detection on every Nth frame (reduces CPU load)")
    p.add_argument("--display", action="store_true",
                   help="Show live window — requires X11 (use with SSH -X or a local display)")
    p.add_argument("--calibrate", action="store_true",
                   help="Calibration mode: press Enter to start a 10-second recording session")
    return p.parse_args()


def draw_boxes(frame, contours):
    out = frame.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return out


def detect_motion(frame, background, args):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    cv2.accumulateWeighted(gray, background, 0.2)
    bg = cv2.convertScaleAbs(background)
    diff = cv2.absdiff(bg, gray)
    _, thresh = cv2.threshold(diff, args.threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion = [c for c in contours if args.min_area < cv2.contourArea(c) < args.max_area]
    return gray, motion


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

        _, motion = detect_motion(frame, background, args)
        areas = sorted([cv2.contourArea(c) for c in motion], reverse=True)

        if areas:
            area_samples.append(areas[0])
            count_samples.append(len(areas))
            print(f"  {remaining:4.1f}s  contours: {len(areas):>3}  largest: {int(areas[0]):>7}"
                  f"  (all: {[int(a) for a in areas[:5]]})")

    if not area_samples:
        print("\nNo motion detected. Try --min-area 5 --max-area 999999.")
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

    if args.display:
        print("NOTE: --display requires a local screen or SSH with X11 forwarding (ssh -X).")

    cam = open_camera(args.width, args.height, args.fps)
    if cam is None:
        print("ERROR: Could not open camera.")
        print("  Pi camera: enable via raspi-config -> Interface Options -> Camera")
        print("  Verify with: libcamera-hello  (or raspistill -o test.jpg for legacy stack)")
        return

    time.sleep(1.0)
    ret, frame = cam.read()
    if not ret:
        print("ERROR: Failed to read first frame.")
        cam.release()
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    background = gray.copy().astype("float")

    if args.calibrate:
        calibrate(cam, args, background)
        cam.release()
        return

    frame_buffer = deque(maxlen=args.pre_frames)
    last_trigger = 0
    post_remaining = 0
    burst_dir = None
    burst_idx = 0
    frame_count = 0

    print(f"Monitoring started at {args.width}x{args.height} "
          f"(processing every {args.process_every} frames). Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                break

            frame_count += 1
            frame_buffer.append(frame.copy())

            # skip frames to reduce CPU load on Pi Zero
            if frame_count % args.process_every != 0:
                continue

            _, motion = detect_motion(frame, background, args)
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

            if args.display:
                disp = draw_boxes(frame, motion) if motion else frame.copy()
                area = sum(cv2.contourArea(c) for c in motion)
                cv2.putText(disp, f"Motion area: {int(area)}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("RoachCam", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("Stopped by user.")

    cam.release()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

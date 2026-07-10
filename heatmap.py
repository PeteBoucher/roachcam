import argparse
import glob
import os
import sys

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Generate a motion heatmap from overnight roachcam captures")
    p.add_argument("--captures-dir", default="captures", help="Directory containing burst_* folders")
    p.add_argument("--output", default="heatmap.jpg", help="Output image path")
    p.add_argument("--alpha", type=float, default=0.6, help="Heatmap overlay opacity (0-1)")
    p.add_argument("--skip-first", type=int, default=0, help="Skip this many burst folders from the start")
    p.add_argument("--skip-last", type=int, default=0, help="Skip this many burst folders from the end")
    p.add_argument("--from", dest="from_time", default=None,
                   help="Only include bursts at or after this timestamp, e.g. 20260616_2324")
    p.add_argument("--to", dest="to_time", default=None,
                   help="Only include bursts up to this timestamp, e.g. 20260617_0600")
    p.add_argument("--min-displacement", type=int, default=20,
                   help="Min pixels a target must move within a burst to draw a trail")
    p.add_argument("--edge-margin", type=int, default=60,
                   help="Entry map: only plot burst starts within this many px of a floor-level edge (0=off)")
    p.add_argument("--floor-zone", type=float, default=0.55,
                   help="Fraction of frame height (from bottom) to treat as floor for edge detection")
    p.add_argument("--hour-from", type=int, default=None,
                   help="Only include bursts at or after this hour of day 0-23 (wraps midnight, e.g. 22)")
    p.add_argument("--hour-to", type=int, default=None,
                   help="Only include bursts before this hour of day 0-23 (e.g. 7 to keep until 07:00)")
    return p.parse_args()


def load_reference(bursts):
    for burst in bursts:
        path = os.path.join(burst, "pre_00.jpg")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return cv2.imread(path)
    for burst in bursts:
        for path in sorted(glob.glob(os.path.join(burst, "motion_*.jpg"))):
            if os.path.getsize(path) > 0:
                img = cv2.imread(path)
                if img is not None:
                    return img
    return None


def extract_detection_boxes(frame):
    """Pull bounding rectangles from the red boxes drawn by main.py."""
    r = frame[:, :, 2].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    b = frame[:, :, 0].astype(np.float32)
    red_mask = (
        (r > 100) &
        (r > g * 1.5) &
        (r > b * 1.5)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 4]


def burst_trail(burst_path):
    """Return ordered (cx, cy) centroids across a burst's motion frames."""
    trail = []
    for path in sorted(glob.glob(os.path.join(burst_path, "motion_*.jpg"))):
        frame = cv2.imread(path)
        if frame is None:
            continue
        boxes = extract_detection_boxes(frame)
        if not boxes:
            continue
        total_area = sum(bw * bh for _, _, bw, bh in boxes)
        if total_area == 0:
            continue
        cx = int(sum((x + bw / 2) * (bw * bh) for x, _, bw, bh in boxes) / total_area)
        cy = int(sum((y + bh / 2) * (bw * bh) for _, y, bw, bh in boxes) / total_area)
        trail.append((cx, cy))
    return trail


def build_edge_mask(image, margin, floor_zone=0.55, debug_path=None):
    """Return a binary mask that is True within `margin` px of any strong structural edge.
    Only edges in the lower `floor_zone` fraction of the frame are considered."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    floor_top = int(h * (1.0 - floor_zone))

    # blank out everything above the floor zone before edge detection
    gray[:floor_top, :] = 0

    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # include the full frame boundary so right/left edge entry still fires
    cv2.rectangle(edges, (0, floor_top), (w - 1, h - 1), 255, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
    mask = cv2.dilate(edges, kernel)
    if debug_path:
        debug = image.copy()
        debug[:floor_top] = (debug[:floor_top] * 0.3).astype(np.uint8)
        debug[mask > 0] = (debug[mask > 0] * 0.4 + np.array([0, 80, 0])).clip(0, 255).astype(np.uint8)
        cv2.imwrite(debug_path, debug)
    return mask


def entry_heatmap(image, bursts, radius=40, edge_margin=0, floor_zone=0.55):
    """Accumulate the start point of each burst's trail into a heatmap."""
    h, w = image.shape[:2]
    acc = np.zeros((h, w), dtype=np.float32)

    kernel_size = radius * 2 + 1
    gauss_1d = cv2.getGaussianKernel(kernel_size, radius / 2)
    kernel = gauss_1d @ gauss_1d.T

    debug_path = "heatmap_edgemask.jpg" if edge_margin else None
    edge_mask = build_edge_mask(image, edge_margin, floor_zone, debug_path) if edge_margin else None

    kept = 0
    for burst in bursts:
        trail = burst_trail(burst)
        if not trail:
            continue
        cx, cy = trail[0]
        if edge_mask is not None and not edge_mask[cy, cx]:
            continue
        kept += 1
        x1, x2 = max(cx - radius, 0), min(cx + radius + 1, w)
        y1, y2 = max(cy - radius, 0), min(cy + radius + 1, h)
        kx1 = radius - (cx - x1)
        kx2 = kx1 + (x2 - x1)
        ky1 = radius - (cy - y1)
        ky2 = ky1 + (y2 - y1)
        acc[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]

    label = f"edge margin={edge_margin}px" if edge_margin else "no edge filter"
    print(f"Entry heatmap: {kept} bursts plotted ({label})")
    if acc.max() == 0:
        return image.copy()

    norm = cv2.normalize(acc, None, 0.0, 1.0, cv2.NORM_MINMAX)
    norm = np.power(norm, 0.4)
    norm = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)

    mask = (acc > 0).astype(np.uint8)
    mask_3ch = cv2.merge([mask * 255] * 3)
    colored = cv2.bitwise_and(colored, mask_3ch)

    out = image.copy()
    out[mask == 1] = cv2.addWeighted(image, 0.4, colored, 0.6, 0)[mask == 1]
    return out


def draw_trails(image, bursts, min_displacement=20):
    """Draw per-burst movement trails as green directed lines."""
    out = image.copy()
    drawn = 0
    for burst in bursts:
        trail = burst_trail(burst)
        if len(trail) < 2:
            continue
        dx = trail[-1][0] - trail[0][0]
        dy = trail[-1][1] - trail[0][1]
        if (dx ** 2 + dy ** 2) ** 0.5 < min_displacement:
            continue
        for i in range(len(trail) - 1):
            cv2.line(out, trail[i], trail[i + 1], (0, 255, 0), 1, cv2.LINE_AA)
        seg_len = max(((trail[-1][0] - trail[-2][0]) ** 2 +
                       (trail[-1][1] - trail[-2][1]) ** 2) ** 0.5, 1)
        tip = min(8 / seg_len, 1.0)  # fixed 8px arrowhead regardless of line length
        cv2.arrowedLine(out, trail[-2], trail[-1], (0, 255, 0), 2,
                        cv2.LINE_AA, tipLength=tip)
        drawn += 1
    print(f"Drew {drawn} trails (min displacement {min_displacement}px)")
    return out


def make_contact_sheet(bursts, thumb_w=320, thumb_h=240, cols=4):
    """Tile the first motion frame from each burst into a labelled grid."""
    thumbs = []
    for burst in bursts:
        frames = sorted(glob.glob(os.path.join(burst, "motion_*.jpg")))
        if not frames:
            frames = sorted(glob.glob(os.path.join(burst, "pre_*.jpg")))
        if not frames:
            continue
        img = cv2.imread(frames[0])
        if img is None:
            continue
        img = cv2.resize(img, (thumb_w, thumb_h))
        label = os.path.basename(burst).replace("burst_", "")
        cv2.rectangle(img, (0, thumb_h - 22), (thumb_w, thumb_h), (0, 0, 0), -1)
        cv2.putText(img, label, (4, thumb_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        thumbs.append(img)

    if not thumbs:
        return None

    rows = (len(thumbs) + cols - 1) // cols
    # pad to full grid
    blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
    while len(thumbs) % cols:
        thumbs.append(blank)

    grid_rows = [np.hstack(thumbs[i * cols:(i + 1) * cols]) for i in range(rows)]
    return np.vstack(grid_rows)


def main():
    args = parse_args()

    bursts = sorted(glob.glob(os.path.join(args.captures_dir, "burst_*")))
    total = len(bursts)

    def burst_ts(path):
        return os.path.basename(path).replace("burst_", "").replace("_", "")

    if args.from_time:
        from_key = args.from_time.replace("_", "")
        bursts = [b for b in bursts if burst_ts(b) >= from_key]
    if args.to_time:
        to_key = args.to_time.replace("_", "")
        bursts = [b for b in bursts if burst_ts(b) <= to_key]

    # time-of-day filter (wraps midnight when hour_from > hour_to)
    if args.hour_from is not None or args.hour_to is not None:
        hf = args.hour_from if args.hour_from is not None else 0
        ht = args.hour_to if args.hour_to is not None else 24
        def in_night_window(path):
            name = os.path.basename(path).replace("burst_", "")  # YYYYMMDD_HHMMSS
            hour = int(name[9:11])
            if hf <= ht:
                return hf <= hour < ht
            else:  # wraps midnight
                return hour >= hf or hour < ht
        bursts = [b for b in bursts if in_night_window(b)]

    # skip-first/skip-last apply within the time-filtered window
    if args.skip_first:
        bursts = bursts[args.skip_first:]
    if args.skip_last:
        bursts = bursts[:-args.skip_last]

    print(f"Using {len(bursts)} of {total} bursts"
          + (f" from {args.from_time}" if args.from_time else "")
          + (f" to {args.to_time}" if args.to_time else "")
          + (f", hours {args.hour_from:02d}:00-{args.hour_to:02d}:00" if args.hour_from is not None or args.hour_to is not None else "")
          + (f", skipping first {args.skip_first}" if args.skip_first else "")
          + (f", skipping last {args.skip_last}" if args.skip_last else ""))
    if not bursts:
        print(f"No burst folders found in {args.captures_dir}/")
        sys.exit(1)

    reference = load_reference(bursts)
    if reference is None:
        print("No reference frame found.")
        sys.exit(1)

    h, w = reference.shape[:2]
    accumulator = np.zeros((h, w), dtype=np.float32)
    frame_count = 0
    box_count = 0

    for burst in bursts:
        motion_frames = sorted(glob.glob(os.path.join(burst, "motion_*.jpg")))
        for path in motion_frames:
            frame = cv2.imread(path)
            if frame is None:
                continue
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))

            boxes = extract_detection_boxes(frame)
            for x, y, bw, bh in boxes:
                accumulator[y:y + bh, x:x + bw] += 1
            box_count += len(boxes)
            frame_count += 1

    if frame_count == 0:
        print("No motion frames found.")
        sys.exit(1)

    # Normalise then apply gamma < 1 to push sparse counts into the warm range
    norm = cv2.normalize(accumulator, None, 0.0, 1.0, cv2.NORM_MINMAX)
    gamma = 0.35  # lower = warmer colours for sparse overnight data
    norm = np.power(norm, gamma)
    norm = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)

    # Zero-motion areas should be transparent — mask them out
    motion_mask = (accumulator > 0).astype(np.uint8)
    motion_mask_3ch = cv2.merge([motion_mask * 255] * 3)
    colored = cv2.bitwise_and(colored, motion_mask_3ch)

    # Blend heatmap over reference
    overlay = reference.copy()
    overlay[motion_mask == 1] = cv2.addWeighted(
        reference, 1 - args.alpha, colored, args.alpha, 0
    )[motion_mask == 1]

    cv2.imwrite(args.output, overlay)

    bare_path = args.output.replace(".jpg", "_bare.jpg")
    cv2.imwrite(bare_path, colored)
    print(f"Bare heatmap saved to {bare_path}")

    trails_on_ref = draw_trails(reference.copy(), bursts, args.min_displacement)
    trails_path = args.output.replace(".jpg", "_trails.jpg")
    cv2.imwrite(trails_path, trails_on_ref)
    print(f"Trails saved to {trails_path}")

    trails_on_heatmap = draw_trails(overlay.copy(), bursts, args.min_displacement)
    combined_path = args.output.replace(".jpg", "_combined.jpg")
    cv2.imwrite(combined_path, trails_on_heatmap)
    print(f"Combined saved to {combined_path}")

    entries_img = entry_heatmap(reference.copy(), bursts,
                               edge_margin=args.edge_margin, floor_zone=args.floor_zone)
    entries_path = args.output.replace(".jpg", "_entries.jpg")
    cv2.imwrite(entries_path, entries_img)
    print(f"Entry point map saved to {entries_path}")

    print(f"Processed {frame_count} frames, {box_count} detection boxes across {len(bursts)} bursts.")
    print(f"Heatmap saved to {args.output}")
    hot = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    print(f"Hottest pixel: x={hot[1]}, y={hot[0]}  (hit {int(accumulator[hot])} times)")

    sheet = make_contact_sheet(bursts)
    if sheet is not None:
        sheet_path = args.output.replace(".jpg", "_sheet.jpg")
        cv2.imwrite(sheet_path, sheet)
        print(f"Contact sheet saved to {sheet_path}")


if __name__ == "__main__":
    main()

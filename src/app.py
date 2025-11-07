import os
import cv2
import time
import yaml
import numpy as np
from ultralytics import YOLO
from collections import deque

# ============================================
# OPTION 1: Using Ultralytics Built-in ByteTrack
# (Easiest - no extra installation needed)
# ============================================

# Parameters for debounce/occlusion filtering
DEBOUNCE_FRAMES = 5
HEIGHT_RATIO_THRESH = 0.7
RESET_DISTANCE = 60
COUNT_COOLDOWN = 20

# ByteTrack specific parameters
TRACK_HIGH_THRESH = 0.5  # High confidence detection threshold
TRACK_LOW_THRESH = 0.1  # Low confidence detection threshold (for occlusion recovery)
TRACK_BUFFER = 30  # Frames to keep lost tracks

# -------- CONFIG LOADING SECTION --------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "virat.yaml")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

VIDEO_IN = cfg["video_path"]
VIDEO_OUT = os.path.join("outputs", "videos", os.path.basename(VIDEO_IN).replace(".mp4", "_bytetrack_out.mp4"))
LINE = tuple(map(tuple, cfg["line"]))
MIN_CONFIDENCE = cfg.get("min_conf", 0.3)  # Lower for ByteTrack
FPS_OVERRIDE = None
PERSON_CLASS = 0
YOLO_WEIGHTS = cfg.get("yolo_weights", "yolov8s.pt")

print("✅ Loaded config (ByteTrack mode):")
print(f"   Video: {VIDEO_IN}")
print(f"   Line: {LINE}")
print(f"   Confidence threshold: {MIN_CONFIDENCE}")
print(f"   Track buffer: {TRACK_BUFFER} frames")
print("-----------------------------------------")

os.makedirs(os.path.dirname(VIDEO_OUT) or ".", exist_ok=True)


def bbox_midpoint(box):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int(y2)  # bottom y-coordinate
    return cx, cy


def main():
    # Load model with built-in ByteTrack tracker
    print("Loading YOLO model with ByteTrack tracker...")
    model = YOLO(YOLO_WEIGHTS)

    # Open video
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open {VIDEO_IN}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    fps = FPS_OVERRIDE or int(src_fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    line_p1, line_p2 = LINE

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (width, height))

    counts = {"Entries": 0, "Exit": 0}
    track_hist = {}

    frame_idx = 0
    t0 = time.time()

    print("Processing video with ByteTrack...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ✅ YOLO tracking with ByteTrack (built-in)
        # Key parameters:
        # - persist=True: Keep track history between frames
        # - tracker="bytetrack.yaml": Use ByteTrack algorithm
        # - conf: Minimum confidence for high-confidence detections
        results = model.track(
            frame,
            persist=True,
            tracker=os.path.join(BASE_DIR, "configs", "bytetrack.yaml")
            ,
            conf=MIN_CONFIDENCE,
            classes=[PERSON_CLASS],  # Only track persons
            verbose=False
        )

        if results[0].boxes.id is None:
            # No tracks this frame
            out.write(frame)
            continue

        # Extract tracking results
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()

        # Process each track
        for box, track_id, conf in zip(boxes, track_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = bbox_midpoint([x1, y1, x2, y2])
            h = y2 - y1

            # Initialize track record if new
            if track_id not in track_hist:
                track_hist[track_id] = {
                    "positions": deque(maxlen=5),
                    "median_h": float(h),
                    "counted": False,
                    "state": None,
                    "last_count_frame": -100000,
                    "confidence": 0.0,
                    "stable_frames": 0,
                    "first_seen_frame": frame_idx
                }

            rec = track_hist[track_id]
            rec["positions"].append((cx, cy))
            rec["median_h"] = 0.9 * rec["median_h"] + 0.1 * float(h)

            # Smoothed centroid
            if len(rec["positions"]) >= 2:
                cx_smooth = int(sum(p[0] for p in list(rec["positions"])[-2:]) / 2)
                cy_smooth = int(sum(p[1] for p in list(rec["positions"])[-2:]) / 2)
            else:
                cx_smooth, cy_smooth = cx, cy

            # Height ratio analysis
            height_ratio = (h / rec["median_h"]) if rec["median_h"] > 0 else 1.0
            occluded = height_ratio < 0.7 or height_ratio > 1.3

            # Update tracking confidence
            if not occluded and len(rec["positions"]) >= 2:
                rec["stable_frames"] += 1
                rec["confidence"] = min(1.0, rec["confidence"] + 0.15)  # Faster confidence build
            else:
                rec["stable_frames"] = 0
                rec["confidence"] = max(0.0, rec["confidence"] - 0.1)  # Slower decay

            track_age = frame_idx - rec["first_seen_frame"]
            MIN_TRACK_AGE = 8  # Reduced from 10 (ByteTrack is more reliable)

            # Compute line crossing
            x1_line, y1_line = line_p1
            x2_line, y2_line = line_p2
            if x2_line == x1_line:
                line_y_at_cx = (y1_line + y2_line) / 2.0
            else:
                line_y_at_cx = y1_line + (y2_line - y1_line) * ((cx_smooth - x1_line) / (x2_line - x1_line))

            side_now = "below" if cy_smooth > line_y_at_cx else "above"
            side_prev = rec["state"]

            def majority_on_side(side):
                if len(rec["positions"]) < 2:
                    return False
                cnt = 0
                for px, py in rec["positions"]:
                    line_y_at_px = y1_line + (y2_line - y1_line) * ((px - x1_line) / max(1, (x2_line - x1_line)))
                    if side == "below":
                        if py > line_y_at_px:
                            cnt += 1
                    else:
                        if py <= line_y_at_px:
                            cnt += 1
                return cnt >= len(rec["positions"]) // 2 + 1

            # Count crossing
            if side_prev is not None and side_prev != side_now:
                frames_since_last_count = frame_idx - rec['last_count_frame']

                # Validation criteria (more lenient with ByteTrack)
                passes_occlusion = not occluded
                passes_cooldown = frames_since_last_count > COUNT_COOLDOWN
                passes_confidence = rec["confidence"] > 0.4  # Lower threshold (was 0.5)
                passes_track_age = track_age >= MIN_TRACK_AGE
                passes_majority = majority_on_side(side_prev)

                if (passes_occlusion and passes_cooldown and passes_confidence
                        and passes_track_age and passes_majority):
                    counts["Entries" if side_prev == "above" else "Exit"] += 1
                    rec["last_count_frame"] = frame_idx
                    rec["counted"] = True
                    print(f"✅ COUNT: Track {track_id} crossing {side_prev}->{side_now} "
                          f"at frame {frame_idx} (conf={rec['confidence']:.2f}, age={track_age})")

            rec["state"] = side_now

            # Color coding
            if rec["confidence"] > 0.7:
                color = (0, 255, 0)  # Green
            elif rec["confidence"] > 0.4:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 100, 255)  # Orange

            if occluded:
                color = (0, 0, 255)  # Red

            # Draw visualization
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track_id} | C:{rec['confidence']:.1f}"
            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

            # Trajectory
            if len(rec["positions"]) > 1:
                pts = list(rec["positions"])
                for i in range(len(pts) - 1):
                    cv2.line(frame, pts[i], pts[i + 1], color, 1)

        # Draw counting line and stats
        cv2.line(frame, line_p1, line_p2, (255, 0, 0), 2)
        cv2.putText(frame, f"Exit: {counts['Exit']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, f"Entries: {counts['Entries']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, f"ByteTrack", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {1.0 / max((time.time() - t0) / max(1, frame_idx), 1e-6):.1f}",
                    (width - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out.write(frame)

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames - Current counts: Exit={counts['Exit']}, Entries={counts['Entries']}")

    cap.release()
    out.release()
    total_t = time.time() - t0
    avg_fps = frame_idx / total_t if total_t > 0 else 0.0

    print("\n=== ByteTrack Results ===")
    print(f"Frames processed: {frame_idx}")
    print(f"Time elapsed: {total_t:.2f}s  Avg FPS: {avg_fps:.2f}")
    print(f"Counts Exit: {counts['Exit']}  Entries: {counts['Entries']}")
    print(f"Saved annotated video to: {VIDEO_OUT}")

    # --- Save per-video CSV summary ---
    csv_dir = os.path.join("outputs", "counters")
    os.makedirs(csv_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, os.path.basename(VIDEO_IN).replace(".avi", ".csv"))

    with open(csv_path, "w") as f:
        f.write("video,frames,avg_fps,Exit,Entries\n")
        f.write(f"{os.path.basename(VIDEO_IN)},{frame_idx},{avg_fps:.2f},{counts['Exit']},{counts['Entries']}\n")

    print(f"📄 Saved count summary to: {csv_path}")



if __name__ == "__main__":
    main()
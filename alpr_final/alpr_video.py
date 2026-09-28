#!/usr/bin/env python3

import argparse
import re
import time
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import TextRecognition


# ============================================================
# CONFIGURATION
# ============================================================

OCR_MODEL = "PP-OCRv5_mobile_rec"

# OCR is not run on every frame.
# Example: 5 means OCR every 5th frame for each plate track.
OCR_INTERVAL = 5

# Maximum distance for matching a detected plate to an old track.
MAX_MATCH_DISTANCE = 90

# Remove a track after this many missed frames.
MAX_MISSED_FRAMES = 20

# YOLO confidence threshold.
YOLO_CONF = 0.35

# Minimum detected plate size.
MIN_PLATE_W = 35
MIN_PLATE_H = 15

# Rectified plate size.
RECT_W = 320
RECT_H = 80

# Number of OCR results kept for voting.
MAX_HISTORY = 30


# ============================================================
# TRACK OBJECT
# ============================================================

@dataclass
class PlateTrack:

    track_id: int

    bbox: tuple

    center: tuple

    last_seen: int

    missed: int = 0

    last_ocr_frame: int = -999999

    current_text: str = ""

    current_score: float = 0.0

    history: list = field(default_factory=list)

    def add_ocr(self, text, score):

        if not text:
            return

        self.history.append(
            (text, float(score))
        )

        if len(self.history) > MAX_HISTORY:
            self.history.pop(0)

        # Count how many times each OCR result occurred.
        counts = Counter(
            text
            for text, score in self.history
        )

        candidates = []

        for text, count in counts.items():

            scores = [
                score
                for t, score in self.history
                if t == text
            ]

            avg_score = sum(scores) / len(scores)

            candidates.append(
                (
                    count,
                    avg_score,
                    text
                )
            )

        # Most frequent result first.
        # OCR confidence is used as a tie breaker.
        candidates.sort(
            key=lambda x: (x[0], x[1]),
            reverse=True
        )

        count, avg_score, best_text = candidates[0]

        self.current_text = best_text

        self.current_score = avg_score


# ============================================================
# OCR TEXT CLEANING
# ============================================================

def normalize_plate(text):

    if not text:
        return ""

    text = text.upper().strip()

    # Remove spaces, dots, hyphens, etc.
    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    # Ignore very short garbage.
    if len(text) < 4:
        return ""

    return text


# ============================================================
# POINT ORDERING
# ============================================================

def order_points(points):

    points = np.asarray(
        points,
        dtype=np.float32
    )

    # top-left has smallest x+y
    # bottom-right has largest x+y
    s = points.sum(axis=1)

    tl = points[np.argmin(s)]

    br = points[np.argmax(s)]

    # top-right has smallest x-y
    # bottom-left has largest x-y
    d = np.diff(
        points,
        axis=1
    ).reshape(-1)

    tr = points[np.argmin(d)]

    bl = points[np.argmax(d)]

    return np.array(
        [
            tl,
            tr,
            br,
            bl
        ],
        dtype=np.float32
    )


# ============================================================
# QUADRILATERAL VALIDATION
# ============================================================

def quad_geometry_valid(
    quad,
    crop_w,
    crop_h
):

    q = order_points(quad)

    # Reject contours touching crop boundary.
    margin = 2

    if np.any(q[:, 0] <= margin):
        return False

    if np.any(q[:, 1] <= margin):
        return False

    if np.any(
        q[:, 0] >= crop_w - margin - 1
    ):
        return False

    if np.any(
        q[:, 1] >= crop_h - margin - 1
    ):
        return False

    width_top = np.linalg.norm(
        q[1] - q[0]
    )

    width_bottom = np.linalg.norm(
        q[2] - q[3]
    )

    height_left = np.linalg.norm(
        q[3] - q[0]
    )

    height_right = np.linalg.norm(
        q[2] - q[1]
    )

    width = (
        width_top +
        width_bottom
    ) / 2

    height = (
        height_left +
        height_right
    ) / 2

    if width < 15 or height < 5:
        return False

    aspect = width / height

    # License plate should be wider than tall.
    if aspect < 2.0:
        return False

    if aspect > 12.0:
        return False

    # Reject extremely distorted shapes.
    if (
        max(width_top, width_bottom) /
        max(
            1.0,
            min(width_top, width_bottom)
        )
    ) > 2.0:
        return False

    if (
        max(height_left, height_right) /
        max(
            1.0,
            min(height_left, height_right)
        )
    ) > 2.0:
        return False

    return True


# ============================================================
# FIND PLATE QUADRILATERAL
# ============================================================

def find_plate_quad(crop):

    h, w = crop.shape[:2]

    if w < 20 or h < 10:
        return None

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    # Upscale small plate crop.
    scale = 3

    gray_big = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    blur = cv2.GaussianBlur(
        gray_big,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        60,
        180
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 3)
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < (
            w * h *
            0.03 *
            scale *
            scale
        ):
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.04 * perimeter,
            True
        )

        if len(approx) != 4:
            continue

        quad_big = approx.reshape(
            4,
            2
        ).astype(np.float32)

        quad = quad_big / scale

        if not quad_geometry_valid(
            quad,
            w,
            h
        ):
            continue

        rect = cv2.boundingRect(
            quad.astype(np.float32)
        )

        x, y, rw, rh = rect

        rect_area = max(
            1,
            rw * rh
        )

        rectangularity = (
            area /
            (
                rect_area *
                scale *
                scale
            )
        )

        score = (
            area *
            max(
                0.1,
                rectangularity
            )
        )

        candidates.append(
            (
                score,
                quad
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return order_points(
        candidates[0][1]
    )


# ============================================================
# FALLBACK GEOMETRY
# ============================================================

def fallback_quad(crop):

    h, w = crop.shape[:2]

    x1 = 0.06 * w
    y1 = 0.08 * h

    x2 = 0.94 * w
    y2 = 0.92 * h

    return np.float32(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ]
    )


# ============================================================
# PERSPECTIVE CORRECTION
# ============================================================

def rectify_plate(
    crop,
    quad
):

    destination = np.float32(
        [
            [0, 0],
            [RECT_W - 1, 0],
            [RECT_W - 1, RECT_H - 1],
            [0, RECT_H - 1]
        ]
    )

    matrix = cv2.getPerspectiveTransform(
        np.float32(quad),
        destination
    )

    rectified = cv2.warpPerspective(
        crop,
        matrix,
        (
            RECT_W,
            RECT_H
        ),
        flags=cv2.INTER_CUBIC
    )

    return rectified


# ============================================================
# CLAHE ENHANCEMENT
# ============================================================

def enhance_plate(plate):

    lab = cv2.cvtColor(
        plate,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    result = cv2.merge(
        (l, a, b)
    )

    result = cv2.cvtColor(
        result,
        cv2.COLOR_LAB2BGR
    )

    return result


# ============================================================
# PADDLEOCR
# ============================================================

def run_ocr(
    recognizer,
    image
):

    try:

        results = recognizer.predict(
            image
        )

        for result in results:

            data = getattr(
                result,
                "json",
                None
            )

            if callable(data):
                data = data()

            if isinstance(data, dict):

                data = data.get(
                    "res",
                    data
                )

                text = data.get(
                    "rec_text",
                    ""
                )

                score = data.get(
                    "rec_score",
                    0.0
                )

                text = normalize_plate(
                    text
                )

                try:
                    score = float(score)
                except Exception:
                    score = 0.0

                return (
                    text,
                    score
                )

            # Compatibility with other PaddleOCR versions.
            text = getattr(
                result,
                "rec_text",
                ""
            )

            score = getattr(
                result,
                "rec_score",
                0.0
            )

            text = normalize_plate(
                text
            )

            try:
                score = float(score)
            except Exception:
                score = 0.0

            return (
                text,
                score
            )

    except Exception as e:

        print(
            f"OCR error: {e}"
        )

    return "", 0.0


# ============================================================
# TRACKING
# ============================================================

def center_of_bbox(
    bbox
):

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def distance(
    a,
    b
):

    return float(
        np.hypot(
            a[0] - b[0],
            a[1] - b[1]
        )
    )


def update_tracks(
    tracks,
    detections,
    frame_number,
    next_track_id
):

    matched_tracks = set()

    matched_detections = set()

    possible_matches = []

    # Calculate every track/detection distance.
    for track_index, track in enumerate(
        tracks
    ):

        for detection_index, detection in enumerate(
            detections
        ):

            d = distance(
                track.center,
                detection["center"]
            )

            possible_matches.append(
                (
                    d,
                    track_index,
                    detection_index
                )
            )

    # Closest matches first.
    possible_matches.sort(
        key=lambda x: x[0]
    )

    for (
        d,
        track_index,
        detection_index
    ) in possible_matches:

        if d > MAX_MATCH_DISTANCE:
            continue

        if track_index in matched_tracks:
            continue

        if detection_index in matched_detections:
            continue

        track = tracks[
            track_index
        ]

        detection = detections[
            detection_index
        ]

        track.bbox = detection[
            "bbox"
        ]

        track.center = detection[
            "center"
        ]

        track.last_seen = frame_number

        track.missed = 0

        matched_tracks.add(
            track_index
        )

        matched_detections.add(
            detection_index
        )

    # Create a new track for every
    # previously unmatched plate.
    for detection_index, detection in enumerate(
        detections
    ):

        if detection_index in matched_detections:
            continue

        tracks.append(
            PlateTrack(
                track_id=next_track_id,
                bbox=detection["bbox"],
                center=detection["center"],
                last_seen=frame_number
            )
        )

        next_track_id += 1

    # Increase missed count.
    for track_index, track in enumerate(
        tracks
    ):

        if track_index not in matched_tracks:

            track.missed += 1

    # Remove old tracks.
    tracks[:] = [
        track
        for track in tracks
        if track.missed <= MAX_MISSED_FRAMES
    ]

    return next_track_id


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Multi-car video ALPR"
    )

    parser.add_argument(
        "--video",
        default="test.mp4",
        help="Input video"
    )

    parser.add_argument(
        "--model",
        default="yolov11s.pt",
        help="YOLO license plate model"
    )

    parser.add_argument(
        "--output",
        default="alpr_output.mp4",
        help="Output video"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=YOLO_CONF,
        help="YOLO confidence"
    )

    parser.add_argument(
        "--ocr-interval",
        type=int,
        default=OCR_INTERVAL,
        help="OCR every N frames"
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Show video while processing"
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("MULTI-CAR VIDEO ALPR")
    print("=" * 60)

    print(
        f"Video        : {args.video}"
    )

    print(
        f"YOLO model   : {args.model}"
    )

    print(
        f"OCR model    : {OCR_MODEL}"
    )

    print(
        f"OCR interval : {args.ocr_interval}"
    )

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print()
    print("Loading YOLO...")

    detector = YOLO(
        args.model
    )

    # --------------------------------------------------------
    # Load PaddleOCR
    # --------------------------------------------------------

    print(
        "Loading PaddleOCR..."
    )

    recognizer = TextRecognition(
        model_name=OCR_MODEL
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        args.video
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: {args.video}"
        )

    fps_input = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps_input <= 0:
        fps_input = 25.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print()
    print(
        f"Input size   : {width}x{height}"
    )

    print(
        f"Input FPS    : {fps_input:.2f}"
    )

    print(
        f"Frames       : {frame_count}"
    )

    # --------------------------------------------------------
    # Output video
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        args.output,
        fourcc,
        fps_input,
        (
            width,
            height
        )
    )

    if not writer.isOpened():

        raise RuntimeError(
            "Cannot create output video"
        )

    # --------------------------------------------------------
    # Tracking state
    # --------------------------------------------------------

    tracks = []

    next_track_id = 1

    frame_number = 0

    processed_frames = 0

    start_time = time.time()

    # --------------------------------------------------------
    # YOLO warmup
    # --------------------------------------------------------

    print()
    print(
        "Warming up YOLO..."
    )

    dummy = np.zeros(
        (
            height,
            width,
            3
        ),
        dtype=np.uint8
    )

    detector.predict(
        dummy,
        conf=args.conf,
        verbose=False
    )

    print(
        "Processing video..."
    )

    print()

    # ========================================================
    # VIDEO LOOP
    # ========================================================

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame_number += 1

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        result = detector.predict(
            frame,
            conf=args.conf,
            verbose=False
        )[0]

        detections = []

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()

            confidences = result.boxes.conf.cpu().numpy()

            for box, confidence in zip(
                boxes,
                confidences
            ):

                x1, y1, x2, y2 = (
                    box.astype(int)
                )

                x1 = max(
                    0,
                    min(
                        x1,
                        width - 1
                    )
                )

                y1 = max(
                    0,
                    min(
                        y1,
                        height - 1
                    )
                )

                x2 = max(
                    0,
                    min(
                        x2,
                        width - 1
                    )
                )

                y2 = max(
                    0,
                    min(
                        y2,
                        height - 1
                    )
                )

                plate_width = x2 - x1

                plate_height = y2 - y1

                if plate_width < MIN_PLATE_W:
                    continue

                if plate_height < MIN_PLATE_H:
                    continue

                bbox = (
                    x1,
                    y1,
                    x2,
                    y2
                )

                detections.append(
                    {
                        "bbox": bbox,
                        "center": center_of_bbox(
                            bbox
                        ),
                        "confidence": float(
                            confidence
                        )
                    }
                )

        # ----------------------------------------------------
        # Track plates
        # ----------------------------------------------------

        next_track_id = update_tracks(
            tracks,
            detections,
            frame_number,
            next_track_id
        )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        for track in tracks:

            # Only process visible tracks.
            if track.missed != 0:
                continue

            # Don't OCR too frequently.
            if (
                frame_number -
                track.last_ocr_frame
                <
                args.ocr_interval
            ):
                continue

            x1, y1, x2, y2 = track.bbox

            bw = x2 - x1
            bh = y2 - y1

            # Small crop padding.
            pad_x = int(
                bw * 0.03
            )

            pad_y = int(
                bh * 0.03
            )

            crop_x1 = max(
                0,
                x1 - pad_x
            )

            crop_y1 = max(
                0,
                y1 - pad_y
            )

            crop_x2 = min(
                width,
                x2 + pad_x
            )

            crop_y2 = min(
                height,
                y2 + pad_y
            )

            crop = frame[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]

            if crop.size == 0:
                continue

            # -----------------------------------------------
            # Perspective correction
            # -----------------------------------------------

            quad = find_plate_quad(
                crop
            )

            if quad is None:

                quad = fallback_quad(
                    crop
                )

            rectified = rectify_plate(
                crop,
                quad
            )

            # -----------------------------------------------
            # CLAHE
            # -----------------------------------------------

            enhanced = enhance_plate(
                rectified
            )

            # -----------------------------------------------
            # OCR
            # -----------------------------------------------

            text, score = run_ocr(
                recognizer,
                enhanced
            )

            track.last_ocr_frame = (
                frame_number
            )

            # Only accept reasonable OCR.
            if (
                text and
                score >= 0.35
            ):

                track.add_ocr(
                    text,
                    score
                )

        # ----------------------------------------------------
        # Draw results
        # ----------------------------------------------------

        active_tracks = 0

        for track in tracks:

            if track.missed != 0:
                continue

            active_tracks += 1

            x1, y1, x2, y2 = track.bbox

            # Plate bounding box.
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Track ID.
            label = (
                f"#{track.track_id}"
            )

            # OCR result.
            if track.current_text:

                label += (
                    f" {track.current_text}"
                )

            # OCR confidence.
            if track.current_score > 0:

                label += (
                    f" ({track.current_score:.2f})"
                )

            (
                text_width,
                text_height
            ), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                2
            )

            label_y = max(
                25,
                y1 - 8
            )

            # Label background.
            cv2.rectangle(
                frame,
                (
                    x1,
                    label_y -
                    text_height -
                    baseline -
                    6
                ),
                (
                    x1 +
                    text_width +
                    6,
                    label_y + 2
                ),
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                label,
                (
                    x1 + 3,
                    label_y - 3
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        processed_frames += 1

        elapsed = (
            time.time() -
            start_time
        )

        current_fps = (
            processed_frames /
            max(
                elapsed,
                0.000001
            )
        )

        status = (
            f"Frame: {frame_number}   "
            f"FPS: {current_fps:.1f}   "
            f"Plates: {active_tracks}"
        )

        cv2.rectangle(
            frame,
            (0, 0),
            (
                min(
                    width,
                    540
                ),
                38
            ),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            status,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Write output
        # ----------------------------------------------------

        writer.write(
            frame
        )

        # ----------------------------------------------------
        # Optional live display
        # ----------------------------------------------------

        if args.display:

            cv2.imshow(
                "ALPR Video",
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    writer.release()

    if args.display:
        cv2.destroyAllWindows()

    elapsed = (
        time.time() -
        start_time
    )

    average_fps = (
        processed_frames /
        max(
            elapsed,
            0.000001
        )
    )

    print()
    print("=" * 60)
    print("FINISHED")
    print("=" * 60)

    print(
        f"Processed frames : {processed_frames}"
    )

    print(
        f"Processing time  : {elapsed:.2f} sec"
    )

    print(
        f"Average FPS      : {average_fps:.2f}"
    )

    print(
        f"Output video     : {args.output}"
    )

    print()
    print("Plate tracks:")

    for track in sorted(
        tracks,
        key=lambda x: x.track_id
    ):

        if track.current_text:

            print(
                f"  Track #{track.track_id}: "
                f"{track.current_text} "
                f"(OCR={track.current_score:.3f})"
            )


if __name__ == "__main__":
    main()

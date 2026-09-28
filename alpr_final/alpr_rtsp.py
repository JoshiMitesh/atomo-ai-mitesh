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

# YOLO confidence.
#
# Your screenshot contains false detections, so start higher
# than the previous 0.35.
YOLO_CONF = 0.55

# OCR every N frames.
OCR_INTERVAL = 5

# Tracking.
MAX_MATCH_DISTANCE = 90
MAX_MISSED_FRAMES = 15

# Minimum detected plate dimensions.
MIN_PLATE_W = 45
MIN_PLATE_H = 15

# Plate aspect ratio.
#
# width / height
MIN_PLATE_ASPECT = 2.0
MAX_PLATE_ASPECT = 8.0

# Rectified plate.
RECT_W = 320
RECT_H = 80

# OCR history.
MAX_HISTORY = 30

# Minimum number of consistent OCR observations
# before showing a plate.
MIN_CONFIRMATIONS = 2

# Minimum OCR confidence.
MIN_OCR_CONF = 0.45


# ============================================================
# INDIAN PLATE FORMAT
# ============================================================

NORMAL_PLATE_RE = re.compile(
    r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$"
)

BH_PLATE_RE = re.compile(
    r"^\d{2}BH\d{4}[A-Z]{1,2}$"
)


# ============================================================
# TRACK
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

    # Number of observations supporting current result.
    confirmation_count: int = 0

    # Last YOLO confidence.
    detection_confidence: float = 0.0

    def add_ocr(self, text, score):

        if not text:
            return

        self.history.append(
            (text, float(score))
        )

        if len(self.history) > MAX_HISTORY:
            self.history.pop(0)

        counts = Counter(
            t
            for t, s in self.history
        )

        candidates = []

        for text, count in counts.items():

            scores = [
                s
                for t, s in self.history
                if t == text
            ]

            avg_score = (
                sum(scores) /
                len(scores)
            )

            candidates.append(
                (
                    count,
                    avg_score,
                    text
                )
            )

        candidates.sort(
            key=lambda x: (
                x[0],
                x[1]
            ),
            reverse=True
        )

        count, avg_score, best_text = (
            candidates[0]
        )

        self.current_text = best_text

        self.current_score = avg_score

        self.confirmation_count = count


# ============================================================
# NORMALIZE OCR
# ============================================================

def normalize_plate(text):

    if not text:
        return ""

    text = str(text).upper().strip()

    # Remove spaces/dots/hyphens.
    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    if len(text) < 7:
        return ""

    if len(text) > 12:
        return ""

    return text


# ============================================================
# VALIDATE INDIAN PLATE
# ============================================================

def is_valid_indian_plate(text):

    if not text:
        return False

    text = normalize_plate(
        text
    )

    if not text:
        return False

    # Normal Indian registration.
    if NORMAL_PLATE_RE.match(text):
        return True

    # BH series.
    if BH_PLATE_RE.match(text):
        return True

    return False


# ============================================================
# POINT ORDER
# ============================================================

def order_points(points):

    points = np.asarray(
        points,
        dtype=np.float32
    )

    s = points.sum(axis=1)

    tl = points[np.argmin(s)]
    br = points[np.argmax(s)]

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
# DETECTION FILTER
# ============================================================

def plate_bbox_valid(
    bbox,
    image_width,
    image_height
):

    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    if w < MIN_PLATE_W:
        return False

    if h < MIN_PLATE_H:
        return False

    if h <= 0:
        return False

    aspect = w / h

    # Reject square-ish objects, signs, people, etc.
    if aspect < MIN_PLATE_ASPECT:
        return False

    if aspect > MAX_PLATE_ASPECT:
        return False

    # Reject extremely tiny regions relative to image.
    image_area = (
        image_width *
        image_height
    )

    plate_area = w * h

    if plate_area < image_area * 0.00005:
        return False

    return True


# ============================================================
# QUAD VALIDATION
# ============================================================

def quad_geometry_valid(
    quad,
    crop_w,
    crop_h
):

    q = order_points(
        quad
    )

    margin = 2

    # Don't accept crop boundary as plate boundary.
    if np.any(
        q[:, 0] <= margin
    ):
        return False

    if np.any(
        q[:, 1] <= margin
    ):
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

    if width < 20:
        return False

    if height < 5:
        return False

    aspect = width / height

    if aspect < 2.0:
        return False

    if aspect > 10.0:
        return False

    return True


# ============================================================
# FIND PLATE QUAD
# ============================================================

def find_plate_quad(crop):

    h, w = crop.shape[:2]

    if w < 20 or h < 10:
        return None

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

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

        area = cv2.contourArea(
            contour
        )

        if area < (
            w *
            h *
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

        quad_big = (
            approx
            .reshape(4, 2)
            .astype(np.float32)
        )

        quad = (
            quad_big /
            scale
        )

        if not quad_geometry_valid(
            quad,
            w,
            h
        ):
            continue

        candidates.append(
            (
                area,
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
# FALLBACK
# ============================================================

def fallback_quad(crop):

    h, w = crop.shape[:2]

    return np.float32(
        [
            [0.06 * w, 0.08 * h],
            [0.94 * w, 0.08 * h],
            [0.94 * w, 0.92 * h],
            [0.06 * w, 0.92 * h]
        ]
    )


# ============================================================
# PERSPECTIVE
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

    return cv2.warpPerspective(
        crop,
        matrix,
        (
            RECT_W,
            RECT_H
        ),
        flags=cv2.INTER_CUBIC
    )


# ============================================================
# CLAHE
# ============================================================

def enhance_plate(plate):

    lab = cv2.cvtColor(
        plate,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    result = cv2.merge(
        (l, a, b)
    )

    return cv2.cvtColor(
        result,
        cv2.COLOR_LAB2BGR
    )


# ============================================================
# OCR
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
                    score = float(
                        score
                    )
                except Exception:
                    score = 0.0

                return (
                    text,
                    score
                )

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
                score = float(
                    score
                )
            except Exception:
                score = 0.0

            return (
                text,
                score
            )

    except Exception as e:

        print(
            "OCR error:",
            e
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

    matches = []

    for ti, track in enumerate(
        tracks
    ):

        for di, detection in enumerate(
            detections
        ):

            d = distance(
                track.center,
                detection["center"]
            )

            matches.append(
                (
                    d,
                    ti,
                    di
                )
            )

    matches.sort(
        key=lambda x: x[0]
    )

    for (
        d,
        ti,
        di
    ) in matches:

        if d > MAX_MATCH_DISTANCE:
            continue

        if ti in matched_tracks:
            continue

        if di in matched_detections:
            continue

        track = tracks[ti]

        detection = detections[di]

        track.bbox = detection[
            "bbox"
        ]

        track.center = detection[
            "center"
        ]

        track.last_seen = frame_number

        track.missed = 0

        track.detection_confidence = (
            detection["confidence"]
        )

        matched_tracks.add(ti)

        matched_detections.add(di)

    # Create new tracks.
    for di, detection in enumerate(
        detections
    ):

        if di in matched_detections:
            continue

        tracks.append(
            PlateTrack(
                track_id=next_track_id,
                bbox=detection["bbox"],
                center=detection["center"],
                last_seen=frame_number,
                detection_confidence=detection[
                    "confidence"
                ]
            )
        )

        next_track_id += 1

    # Missed tracks.
    for ti, track in enumerate(
        tracks
    ):

        if ti not in matched_tracks:

            track.missed += 1

    # Remove old tracks.
    tracks[:] = [
        track
        for track in tracks
        if track.missed <= MAX_MISSED_FRAMES
    ]

    return next_track_id


# ============================================================
# RTSP
# ============================================================

def open_rtsp(url):

    print(
        "Connecting to RTSP..."
    )

    cap = cv2.VideoCapture(
        url,
        cv2.CAP_FFMPEG
    )

    try:
        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )
    except Exception:
        pass

    if not cap.isOpened():

        cap.release()

        cap = cv2.VideoCapture(
            url
        )

    if cap.isOpened():

        print(
            "RTSP connected."
        )

    return cap


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rtsp",
        required=True
    )

    parser.add_argument(
        "--model",
        default="yolov11s.pt"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=YOLO_CONF
    )

    parser.add_argument(
        "--ocr-interval",
        type=int,
        default=OCR_INTERVAL
    )

    args = parser.parse_args()

    print()
    print("=" * 65)
    print("INDIAN RTSP LICENSE PLATE RECOGNITION")
    print("=" * 65)

    print(
        "YOLO confidence:",
        args.conf
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print(
        "\nLoading YOLO..."
    )

    detector = YOLO(
        args.model
    )

    print(
        "Loading PaddleOCR..."
    )

    recognizer = TextRecognition(
        model_name=OCR_MODEL
    )

    # --------------------------------------------------------
    # RTSP
    # --------------------------------------------------------

    cap = open_rtsp(
        args.rtsp
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Cannot connect to RTSP."
        )

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

    print(
        f"Resolution: {width}x{height}"
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    tracks = []

    next_track_id = 1

    frame_number = 0

    processed_frames = 0

    start_time = time.time()

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    while True:

        ok, frame = cap.read()

        if not ok or frame is None:

            print(
                "RTSP frame lost. Reconnecting..."
            )

            cap.release()

            time.sleep(1)

            cap = open_rtsp(
                args.rtsp
            )

            continue

        frame_number += 1

        # ====================================================
        # YOLO
        # ====================================================

        result = detector.predict(
            frame,
            conf=args.conf,
            verbose=False
        )[0]

        detections = []

        if result.boxes is not None:

            boxes = (
                result.boxes
                .xyxy
                .cpu()
                .numpy()
            )

            confidences = (
                result.boxes
                .conf
                .cpu()
                .numpy()
            )

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

                bbox = (
                    x1,
                    y1,
                    x2,
                    y2
                )

                # --------------------------------------------
                # IMPORTANT:
                # Reject non-plate-shaped detections BEFORE OCR
                # --------------------------------------------

                if not plate_bbox_valid(
                    bbox,
                    width,
                    height
                ):
                    continue

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

        # ====================================================
        # TRACKING
        # ====================================================

        next_track_id = update_tracks(
            tracks,
            detections,
            frame_number,
            next_track_id
        )

        # ====================================================
        # OCR
        # ====================================================

        for track in tracks:

            if track.missed != 0:
                continue

            if (
                frame_number -
                track.last_ocr_frame
                <
                args.ocr_interval
            ):
                continue

            x1, y1, x2, y2 = (
                track.bbox
            )

            bw = x2 - x1
            bh = y2 - y1

            pad_x = int(
                bw * 0.03
            )

            pad_y = int(
                bh * 0.03
            )

            cx1 = max(
                0,
                x1 - pad_x
            )

            cy1 = max(
                0,
                y1 - pad_y
            )

            cx2 = min(
                width,
                x2 + pad_x
            )

            cy2 = min(
                height,
                y2 + pad_y
            )

            crop = frame[
                cy1:cy2,
                cx1:cx2
            ]

            if crop.size == 0:
                continue

            # --------------------------------------------
            # Perspective correction
            # --------------------------------------------

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

            # --------------------------------------------
            # CLAHE
            # --------------------------------------------

            enhanced = enhance_plate(
                rectified
            )

            # --------------------------------------------
            # OCR
            # --------------------------------------------

            text, score = run_ocr(
                recognizer,
                enhanced
            )

            track.last_ocr_frame = (
                frame_number
            )

            # =================================================
            # VERY IMPORTANT:
            # Reject OCR which isn't an Indian plate format.
            # =================================================

            if not text:
                continue

            if score < MIN_OCR_CONF:
                continue

            if not is_valid_indian_plate(
                text
            ):
                continue

            # Only valid Indian plate text reaches here.
            track.add_ocr(
                text,
                score
            )

        # ====================================================
        # DISPLAY
        # ====================================================

        active = 0

        for track in tracks:

            if track.missed != 0:
                continue

            active += 1

            x1, y1, x2, y2 = (
                track.bbox
            )

            # Don't display unconfirmed OCR.
            confirmed = (
                track.confirmation_count
                >= MIN_CONFIRMATIONS
            )

            if confirmed:

                # Plate bounding box.
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = (
                    f"#{track.track_id} "
                    f"{track.current_text}"
                )

                label += (
                    f" "
                    f"({track.current_score:.2f})"
                )

                (
                    tw,
                    th
                ), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    2
                )

                label_y = max(
                    30,
                    y1 - 8
                )

                cv2.rectangle(
                    frame,
                    (
                        x1,
                        label_y -
                        th -
                        baseline -
                        6
                    ),
                    (
                        x1 +
                        tw +
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

            else:

                # We know there is a possible plate,
                # but don't display OCR yet.
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 165, 255),
                    2
                )

        # ====================================================
        # FPS
        # ====================================================

        processed_frames += 1

        elapsed = (
            time.time() -
            start_time
        )

        fps = (
            processed_frames /
            max(
                elapsed,
                0.000001
            )
        )

        status = (
            f"FPS: {fps:.1f}   "
            f"Plate candidates: {active}"
        )

        cv2.rectangle(
            frame,
            (0, 0),
            (480, 40),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            status,
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "RTSP ALPR",
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

    cv2.destroyAllWindows()

    print()
    print("=" * 65)
    print("ALPR STOPPED")
    print("=" * 65)

    print(
        f"Processed frames: "
        f"{processed_frames}"
    )

    print(
        f"Average FPS: "
        f"{processed_frames / max(time.time() - start_time, 0.001):.2f}"
    )

    print()
    print("Detected plates:")

    for track in tracks:

        if (
            track.current_text
            and
            track.confirmation_count
            >= MIN_CONFIRMATIONS
        ):

            print(
                f"Track #{track.track_id}: "
                f"{track.current_text} "
                f"confidence={track.current_score:.3f} "
                f"observations={track.confirmation_count}"
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os

# ============================================================
# IMPORTANT: Set these BEFORE importing Paddle/PaddleOCR
# ============================================================

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"

import cv2
import sys
import re
import time
import math
import subprocess
import numpy as np

# ============================================================
# ASNN
# ============================================================

try:
    from asnn.api import asnn as NPU_API
    from asnn.types import output_format

    NPU_KIND = "ASNN"

except ImportError:
    print("[ERROR] ASNN module not found.")
    sys.exit(1)


# ============================================================
# PaddleOCR - ONNX Runtime
# ============================================================

try:
    from paddleocr import TextRecognition

except ImportError:
    print("[ERROR] paddleocr is not installed.")
    print("Install/use your yolov11s virtual environment.")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "/home/electron/yolov11s/yolov11s.nb"
LIBRARY = "/home/electron/yolov11s/libnn_yolov11s.so"

RTSP_URL = "rtsp://192.168.1.38:8554/video"

# RTSP stream
STREAM_W = 464
STREAM_H = 832

# NPU input
TARGET_SIZE = 640

# YOLO11s converted model
NUM_CLS = 1
LISTSIZE = 65

# Detection thresholds
OBJ_THRESH = 0.45
NMS_THRESH = 0.30

# DFL/post-processing
CANDIDATE_GATHER_THRESH = 0.18
DFL_CERTAINTY_THRESH = 0.18

# Plate geometry
MIN_PLATE_ASPECT = 2.0
MAX_PLATE_ASPECT = 8.0

MIN_PLATE_W = 35
MIN_PLATE_H = 10

# NPU processing rate
NPU_FPS = 12.0

# OCR
OCR_MIN_CONF = 0.45

# OCR input
OCR_W = 320
OCR_H = 80

# Tracking
TRACK_IOU_THRESHOLD = 0.20
TRACK_MAX_MISSED = 18

# Number of observations needed before stable output
MIN_VALID_OBSERVATIONS = 3

# Minimum weighted evidence
MIN_STABLE_SCORE = 1.8

# Maximum OCR candidates retained per track
MAX_HISTORY = 25

# Show information every frame
PRINT_EVERY_FRAME = True

# Save debug images
SAVE_DEBUG = False

DEBUG_DIR = "/tmp/alpr_debug"

# GUI disabled because this machine may be headless/SSH
SHOW_WINDOW = False


# ============================================================
# INDIAN PLATE REGEX
# ============================================================

NORMAL_PLATE_RE = re.compile(
    r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$"
)

BH_PLATE_RE = re.compile(
    r"^\d{2}BH\d{4}[A-Z]{1,2}$"
)


# ============================================================
# OCR CONFUSION TABLE
# ============================================================

OCR_CONFUSIONS = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "T": "7",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def normalize_ocr(text):
    if text is None:
        return ""

    text = str(text).upper()

    # Remove spaces, punctuation and anything except A-Z / 0-9
    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


def is_indian_plate(text):
    if not text:
        return False

    if NORMAL_PLATE_RE.fullmatch(text):
        return True

    if BH_PLATE_RE.fullmatch(text):
        return True

    return False


def plate_format_type(text):
    if not text:
        return "invalid"

    if BH_PLATE_RE.fullmatch(text):
        return "BH"

    if NORMAL_PLATE_RE.fullmatch(text):
        return "NORMAL"

    return "invalid"


# ============================================================
# OCR CANDIDATE CORRECTION
# ============================================================

def generate_corrections(text):
    """
    Generate a small number of conservative OCR alternatives.

    We do NOT blindly replace every O with 0 because letters and
    numbers can both legitimately occur in Indian plates.

    Corrections are mainly used around positions where the
    Indian registration format expects a digit/letter.
    """

    text = normalize_ocr(text)

    if not text:
        return []

    candidates = [text]

    # --------------------------------------------------------
    # Normal Indian registration:
    #
    # XX 00 XX 0000
    #
    # We try common OCR confusions in likely digit positions.
    # --------------------------------------------------------

    if len(text) >= 8:
        chars = list(text)

        # First two characters should normally be state letters.
        for i in range(min(2, len(chars))):
            if chars[i] == "0":
                chars[i] = "O"
            elif chars[i] == "1":
                chars[i] = "I"

        state_fixed = "".join(chars)
        candidates.append(state_fixed)

        # Try numeric positions after state code.
        for digit_end in (4, 5):
            if len(text) >= digit_end:
                chars = list(text)

                for i in range(2, min(digit_end, len(chars))):
                    if chars[i] in OCR_CONFUSIONS:
                        chars[i] = OCR_CONFUSIONS[chars[i]]

                candidates.append("".join(chars))

    # --------------------------------------------------------
    # Generic conservative single-character alternatives
    # --------------------------------------------------------

    base = text

    for i, ch in enumerate(base):
        if ch in OCR_CONFUSIONS:
            alt = list(base)
            alt[i] = OCR_CONFUSIONS[ch]
            candidates.append("".join(alt))

    # Remove duplicates
    unique = []

    for c in candidates:
        if c not in unique:
            unique.append(c)

    return unique[:12]


def choose_best_plate_candidate(raw_text, raw_conf):
    """
    Convert OCR output into the best plate candidate.

    Priority:
        1. Exact Indian format
        2. Corrected Indian format
        3. Otherwise normalized OCR text

    Returns:
        text, confidence, valid
    """

    raw = normalize_ocr(raw_text)

    if not raw:
        return "", 0.0, False

    # Exact match gets highest priority.
    if is_indian_plate(raw):
        return raw, float(raw_conf), True

    # Try conservative corrections.
    corrections = generate_corrections(raw)

    valid_candidates = []

    for candidate in corrections:

        if is_indian_plate(candidate):

            # Small penalty for corrected text.
            penalty = 0.96 if candidate == raw else 0.88

            score = float(raw_conf) * penalty

            valid_candidates.append(
                (
                    score,
                    candidate
                )
            )

    if valid_candidates:

        valid_candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        score, candidate = valid_candidates[0]

        return candidate, score, True

    # Not valid yet.
    return raw, float(raw_conf), False


# ============================================================
# LETTERBOX
# ============================================================

def letterbox(frame, target=640):

    h, w = frame.shape[:2]

    scale = min(
        target / float(w),
        target / float(h)
    )

    nw = int(round(w * scale))
    nh = int(round(h * scale))

    resized = cv2.resize(
        frame,
        (nw, nh),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.full(
        (target, target, 3),
        114,
        dtype=np.uint8
    )

    x0 = (target - nw) // 2
    y0 = (target - nh) // 2

    canvas[
        y0:y0 + nh,
        x0:x0 + nw
    ] = resized

    return canvas, scale, x0, y0


def unletterbox_box(box, scale, x0, y0, orig_w, orig_h):

    x1, y1, x2, y2 = box

    x1 = (x1 - x0) / scale
    y1 = (y1 - y0) / scale
    x2 = (x2 - x0) / scale
    y2 = (y2 - y0) / scale

    x1 = max(0, min(orig_w - 1, x1))
    y1 = max(0, min(orig_h - 1, y1))
    x2 = max(0, min(orig_w - 1, x2))
    y2 = max(0, min(orig_h - 1, y2))

    return [
        int(round(x1)),
        int(round(y1)),
        int(round(x2)),
        int(round(y2))
    ]


# ============================================================
# DFL DECODER
# ============================================================

def decode_tensor(
    tensor,
    grid_h,
    grid_w
):

    """
    Tensor layout:

        1 class + 64 DFL
        = 65 values/cell

    DFL:
        left  : 16 bins
        top   : 16 bins
        right : 16 bins
        bottom: 16 bins
    """

    arr = np.asarray(tensor, dtype=np.float32)

    arr = arr.reshape(
        LISTSIZE,
        grid_h,
        grid_w
    )

    cls_logits = arr[0]

    dfl_logits = arr[1:65]

    cls_scores = sigmoid(cls_logits)

    boxes = []
    scores = []

    stride = TARGET_SIZE / float(grid_h)

    for gy in range(grid_h):

        for gx in range(grid_w):

            score = float(
                cls_scores[gy, gx]
            )

            if score < CANDIDATE_GATHER_THRESH:
                continue

            dfl = dfl_logits[
                :,
                gy,
                gx
            ]

            # Four groups of 16 bins
            distances = []

            certainty_values = []

            for side in range(4):

                logits = dfl[
                    side * 16:
                    (side + 1) * 16
                ]

                probs = softmax(logits)

                bins = np.arange(
                    16,
                    dtype=np.float32
                )

                distance = float(
                    np.sum(
                        probs * bins
                    )
                )

                certainty = float(
                    np.max(probs)
                )

                distances.append(distance)
                certainty_values.append(certainty)

            dfl_certainty = float(
                np.mean(certainty_values)
            )

            if dfl_certainty < DFL_CERTAINTY_THRESH:
                continue

            left, top, right, bottom = distances

            cx = (gx + 0.5) * stride
            cy = (gy + 0.5) * stride

            x1 = cx - left * stride
            y1 = cy - top * stride

            x2 = cx + right * stride
            y2 = cy + bottom * stride

            x1 = max(0.0, min(TARGET_SIZE, x1))
            y1 = max(0.0, min(TARGET_SIZE, y1))
            x2 = max(0.0, min(TARGET_SIZE, x2))
            y2 = max(0.0, min(TARGET_SIZE, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append(
                [x1, y1, x2, y2]
            )

            scores.append(score)

    return boxes, scores


# ============================================================
# NMS
# ============================================================

def nms_boxes(boxes, scores):

    if not boxes:
        return []

    cv_boxes = []

    for b in boxes:

        x1, y1, x2, y2 = b

        cv_boxes.append([
            int(x1),
            int(y1),
            int(x2 - x1),
            int(y2 - y1)
        ])

    indices = cv2.dnn.NMSBoxes(
        cv_boxes,
        scores,
        OBJ_THRESH,
        NMS_THRESH
    )

    if indices is None:
        return []

    if len(indices) == 0:
        return []

    indices = np.array(
        indices
    ).reshape(-1)

    return [
        int(i)
        for i in indices
    ]


# ============================================================
# PLATE GEOMETRY
# ============================================================

def valid_plate_geometry(box):

    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1

    if w < MIN_PLATE_W:
        return False

    if h < MIN_PLATE_H:
        return False

    if h <= 0:
        return False

    aspect = w / float(h)

    if aspect < MIN_PLATE_ASPECT:
        return False

    if aspect > MAX_PLATE_ASPECT:
        return False

    return True


# ============================================================
# NPU DETECTOR
# ============================================================

class NPUDetector:

    def __init__(self):

        print()
        print("========================================")
        print("Initializing ASNN NPU")
        print("========================================")
        print("Runtime :", NPU_KIND)
        print("Model   :", MODEL)
        print("Library :", LIBRARY)

        if not os.path.exists(MODEL):
            raise FileNotFoundError(
                f"Model not found: {MODEL}"
            )

        if not os.path.exists(LIBRARY):
            raise FileNotFoundError(
                f"Library not found: {LIBRARY}"
            )

        self.npu = NPU_API("Electron")

        result = self.npu.nn_init(
            library=os.path.abspath(LIBRARY),
            model=os.path.abspath(MODEL),
            level=0
        )

        print("nn_init result:", result)

        if "SUCCESS" not in str(result):
            raise RuntimeError(
                f"NPU initialization failed: {result}"
            )

        print("NPU initialization successful")

        self.input_buf = np.zeros(
            (
                1,
                3,
                TARGET_SIZE,
                TARGET_SIZE
            ),
            dtype=np.float32
        )

    def detect(self, frame):

        orig_h, orig_w = frame.shape[:2]

        image, scale, x0, y0 = letterbox(
            frame,
            TARGET_SIZE
        )

        # BGR -> RGB
        image = image[:, :, ::-1]

        # HWC -> CHW
        image = image.transpose(
            2,
            0,
            1
        )

        # Normalize
        image = (
            image.astype(
                np.float32
            ) / 255.0
        )

        self.input_buf[0] = image

        data = self.npu.nn_inference(
            [self.input_buf],
            platform="ONNX",
            reorder="2 1 0",
            output_tensor=3,
            output_format=output_format.OUT_FORMAT_FLOAT32
        )

        if data is None:
            return []

        if len(data) < 3:
            return []

        # Verified output order:
        #
        # data[0] = 80x80x65
        # data[1] = 40x40x65
        # data[2] = 20x20x65

        grid_sizes = [
            80,
            40,
            20
        ]

        all_boxes = []
        all_scores = []

        for tensor, grid in zip(
            data[:3],
            grid_sizes
        ):

            boxes, scores = decode_tensor(
                tensor,
                grid,
                grid
            )

            all_boxes.extend(boxes)
            all_scores.extend(scores)

        if not all_boxes:
            return []

        keep = nms_boxes(
            all_boxes,
            all_scores
        )

        detections = []

        for idx in keep:

            box = all_boxes[idx]
            score = float(
                all_scores[idx]
            )

            box_original = unletterbox_box(
                box,
                scale,
                x0,
                y0,
                orig_w,
                orig_h
            )

            if not valid_plate_geometry(
                box_original
            ):
                continue

            detections.append({
                "box": box_original,
                "score": score,
                "class_id": 0
            })

        return detections

    def release(self):

        try:
            self.npu.nn_deinit()

        except Exception:
            pass


# ============================================================
# PERSPECTIVE CORRECTION
# ============================================================

def order_points(points):

    pts = np.asarray(
        points,
        dtype=np.float32
    )

    s = pts.sum(axis=1)
    d = np.diff(
        pts,
        axis=1
    ).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]

    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]

    return np.array(
        [
            top_left,
            top_right,
            bottom_right,
            bottom_left
        ],
        dtype=np.float32
    )


def proportional_corners(crop):

    """
    Conservative fallback.

    The detected YOLO box is already a plate region, so instead
    of allowing a contour detector to accidentally use the crop
    boundary, use a small inset trapezoid.
    """

    h, w = crop.shape[:2]

    # Small inset.
    x_margin = max(
        1,
        int(w * 0.04)
    )

    y_margin = max(
        1,
        int(h * 0.08)
    )

    # Slight trapezoid.
    pts = np.float32([
        [x_margin + w * 0.015, y_margin],
        [w - x_margin - w * 0.015, y_margin + h * 0.04],
        [w - x_margin, h - y_margin],
        [x_margin, h - y_margin - h * 0.03]
    ])

    return order_points(pts)


def find_plate_corners(crop):

    """
    Attempt contour detection, but reject unreliable contours.

    If no trustworthy quadrilateral is found, use conservative
    proportional corners.
    """

    h, w = crop.shape[:2]

    if w < 20 or h < 10:
        return proportional_corners(crop)

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        50,
        150
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_score = -1.0

    crop_area = float(w * h)

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        # Avoid tiny and full-crop contours.
        if area < crop_area * 0.08:
            continue

        if area > crop_area * 0.92:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.03 * perimeter,
            True
        )

        if len(approx) != 4:
            continue

        pts = approx.reshape(
            4,
            2
        ).astype(
            np.float32
        )

        ordered = order_points(
            pts
        )

        width_top = np.linalg.norm(
            ordered[1] - ordered[0]
        )

        width_bottom = np.linalg.norm(
            ordered[2] - ordered[3]
        )

        height_left = np.linalg.norm(
            ordered[3] - ordered[0]
        )

        height_right = np.linalg.norm(
            ordered[2] - ordered[1]
        )

        avg_width = (
            width_top +
            width_bottom
        ) / 2.0

        avg_height = (
            height_left +
            height_right
        ) / 2.0

        if avg_height <= 0:
            continue

        aspect = avg_width / avg_height

        if aspect < 1.8 or aspect > 9.0:
            continue

        # Reject quadrilaterals touching crop boundaries.
        margin = 1.5

        if np.any(
            ordered[:, 0] <= margin
        ):
            continue

        if np.any(
            ordered[:, 1] <= margin
        ):
            continue

        if np.any(
            ordered[:, 0] >= w - margin
        ):
            continue

        if np.any(
            ordered[:, 1] >= h - margin
        ):
            continue

        rectangularity = area / (
            avg_width * avg_height + 1e-6
        )

        score = (
            rectangularity *
            min(area / crop_area, 1.0)
        )

        if score > best_score:

            best_score = score
            best = ordered

    if best is not None:
        return best

    return proportional_corners(crop)


def perspective_correct(crop):

    src = find_plate_corners(
        crop
    )

    dst = np.float32([
        [0, 0],
        [OCR_W - 1, 0],
        [OCR_W - 1, OCR_H - 1],
        [0, OCR_H - 1]
    ])

    matrix = cv2.getPerspectiveTransform(
        src,
        dst
    )

    rectified = cv2.warpPerspective(
        crop,
        matrix,
        (OCR_W, OCR_H),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rectified


# ============================================================
# OCR PREPROCESSING
# ============================================================

def enhance_plate(image):

    # CLAHE on luminance
    lab = cv2.cvtColor(
        image,
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

    enhanced = cv2.merge(
        [l, a, b]
    )

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )

    # Mild sharpening
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    enhanced = cv2.filter2D(
        enhanced,
        -1,
        kernel
    )

    return enhanced


# ============================================================
# OCR
# ============================================================

class PlateOCR:

    def __init__(self):

        print()
        print("========================================")
        print("Initializing OCR")
        print("========================================")

        self.recognizer = TextRecognition(
            model_name="PP-OCRv5_mobile_rec",
            engine="onnxruntime"
        )

        print("OCR engine : ONNX Runtime")
        print("OCR model  : PP-OCRv5_mobile_rec")
        print("OCR ready")

    def recognize(self, image):

        start = time.perf_counter()

        try:

            result = self.recognizer.predict(image)

            best_text = ""
            best_conf = 0.0

            for res in result:

                data = None

                # ------------------------------------------------
                # PaddleOCR 3.x result
                #
                # Usually:
                #
                # {
                #     "res": {
                #         "rec_text": "...",
                #         "rec_score": ...
                #     }
                # }
                # ------------------------------------------------

                try:

                    if hasattr(res, "json"):

                        data = res.json

                        if callable(data):
                            data = data()

                except Exception:
                    data = None

                # ------------------------------------------------
                # Sometimes result can already be a dict
                # ------------------------------------------------

                if data is None:

                    try:

                        if isinstance(res, dict):
                            data = res

                    except Exception:
                        pass

                # ------------------------------------------------
                # Extract OCR result
                # ------------------------------------------------

                rec_text = ""
                rec_score = 0.0

                if isinstance(data, dict):

                    # Case 1:
                    # {
                    #   "rec_text": "...",
                    #   "rec_score": ...
                    # }

                    if "rec_text" in data:

                        rec_text = data.get(
                            "rec_text",
                            ""
                        )

                        rec_score = data.get(
                            "rec_score",
                            0.0
                        )

                    # Case 2:
                    # {
                    #   "res": {
                    #       "rec_text": "...",
                    #       "rec_score": ...
                    #   }
                    # }

                    elif isinstance(
                        data.get("res"),
                        dict
                    ):

                        inner = data["res"]

                        rec_text = inner.get(
                            "rec_text",
                            ""
                        )

                        rec_score = inner.get(
                            "rec_score",
                            0.0
                        )

                # ------------------------------------------------
                # Object-style access fallback
                # ------------------------------------------------

                if not rec_text:

                    try:

                        if hasattr(
                            res,
                            "rec_text"
                        ):

                            rec_text = (
                                res.rec_text
                            )

                            rec_score = (
                                res.rec_score
                            )

                    except Exception:
                        pass

                # ------------------------------------------------
                # Convert confidence
                # ------------------------------------------------

                try:

                    rec_score = float(
                        rec_score
                    )

                except Exception:

                    rec_score = 0.0

                rec_text = str(
                    rec_text
                ).strip()

                # ------------------------------------------------
                # Keep highest confidence result
                # ------------------------------------------------

                if rec_text and (
                    rec_score > best_conf
                ):

                    best_text = rec_text
                    best_conf = rec_score

            elapsed = (
                time.perf_counter()
                - start
            ) * 1000.0

            return (
                best_text,
                best_conf,
                elapsed
            )

        except Exception as e:

            print(
                "[OCR ERROR]",
                repr(e)
            )

            return (
                "",
                0.0,
                (
                    time.perf_counter()
                    - start
                ) * 1000.0
            )

# ============================================================
# IOU
# ============================================================

def bbox_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0,
        ix2 - ix1
    )

    ih = max(
        0,
        iy2 - iy1
    )

    intersection = (
        iw * ih
    )

    area_a = max(
        0,
        ax2 - ax1
    ) * max(
        0,
        ay2 - ay1
    )

    area_b = max(
        0,
        bx2 - bx1
    ) * max(
        0,
        by2 - by1
    )

    union = (
        area_a +
        area_b -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# PLATE TRACK
# ============================================================

class PlateTrack:

    def __init__(
        self,
        track_id,
        detection
    ):

        self.track_id = track_id

        self.box = detection["box"]

        self.detection_conf = detection[
            "score"
        ]

        self.age = 1
        self.missed = 0

        # OCR history:
        #
        # [
        #   {
        #       text,
        #       raw,
        #       conf,
        #       valid,
        #       weight
        #   }
        # ]
        self.history = []

        self.last_text = ""
        self.last_conf = 0.0
        self.last_valid = False

        self.stable_text = ""

        self.total_valid = 0

    def update_box(
        self,
        box,
        detection_conf
    ):

        self.box = box
        self.detection_conf = detection_conf

        self.age += 1
        self.missed = 0

    def miss(self):

        self.missed += 1

    def add_ocr(
        self,
        raw_text,
        raw_conf
    ):

        text, conf, valid = (
            choose_best_plate_candidate(
                raw_text,
                raw_conf
            )
        )

        if not text:
            return

        # Detection confidence contributes to evidence.
        det_weight = max(
            0.35,
            min(
                1.0,
                float(
                    self.detection_conf
                )
            )
        )

        # Valid Indian format receives a substantial bonus.
        valid_bonus = (
            1.65
            if valid
            else 0.30
        )

        # OCR confidence.
        ocr_weight = max(
            0.10,
            min(
                1.0,
                float(conf)
            )
        )

        weight = (
            ocr_weight *
            det_weight *
            valid_bonus
        )

        # Additional bonus for longer valid plate.
        if valid:

            if len(text) >= 10:
                weight *= 1.10

            if len(text) >= 9:
                weight *= 1.05

        item = {
            "text": text,
            "raw": normalize_ocr(
                raw_text
            ),
            "conf": float(conf),
            "valid": bool(valid),
            "weight": float(weight)
        }

        self.history.append(
            item
        )

        if len(self.history) > MAX_HISTORY:

            self.history = (
                self.history[
                    -MAX_HISTORY:
                ]
            )

        self.last_text = text
        self.last_conf = conf
        self.last_valid = valid

        if valid:
            self.total_valid += 1

    def get_best_result(self):

        if not self.history:
            return "", 0.0, False, 0.0

        scores = {}

        counts = {}

        valid_counts = {}

        for item in self.history:

            text = item["text"]

            if not text:
                continue

            weight = item["weight"]

            scores[text] = (
                scores.get(text, 0.0)
                + weight
            )

            counts[text] = (
                counts.get(text, 0)
                + 1
            )

            if item["valid"]:

                valid_counts[text] = (
                    valid_counts.get(
                        text,
                        0
                    ) + 1
                )

        if not scores:
            return "", 0.0, False, 0.0

        candidates = []

        for text, score in scores.items():

            count = counts.get(
                text,
                0
            )

            valid_count = valid_counts.get(
                text,
                0
            )

            is_valid = is_indian_plate(
                text
            )

            final_score = score

            # Repeated observations.
            if count >= 2:
                final_score *= 1.10

            if count >= 3:
                final_score *= 1.10

            # Valid format gets strong preference.
            if is_valid:
                final_score *= 1.45

            # Multiple independent valid frames.
            if valid_count >= 2:
                final_score *= 1.20

            if valid_count >= 3:
                final_score *= 1.15

            candidates.append(
                (
                    final_score,
                    text,
                    count,
                    valid_count,
                    is_valid
                )
            )

        # Valid candidates first, then score.
        candidates.sort(
            key=lambda x: (
                x[4],
                x[3],
                x[0]
            ),
            reverse=True
        )

        best = candidates[0]

        final_score = best[0]
        text = best[1]
        count = best[2]
        valid_count = best[3]
        valid = best[4]

        # Require enough evidence.
        stable = (
            valid
            and
            valid_count >= MIN_VALID_OBSERVATIONS
            and
            final_score >= MIN_STABLE_SCORE
        )

        if stable:

            self.stable_text = text

        # Find average confidence for this candidate.
        conf_values = [
            item["conf"]
            for item in self.history
            if item["text"] == text
        ]

        avg_conf = (
            sum(conf_values)
            / len(conf_values)
            if conf_values
            else 0.0
        )

        return (
            text,
            avg_conf,
            stable,
            final_score
        )


# ============================================================
# PLATE TRACKER
# ============================================================

class PlateTracker:

    def __init__(self):

        self.tracks = []

        self.next_id = 1

    def update(
        self,
        detections
    ):

        if not self.tracks:

            for detection in detections:

                self.tracks.append(
                    PlateTrack(
                        self.next_id,
                        detection
                    )
                )

                self.next_id += 1

            return self.tracks

        matched_tracks = set()
        matched_detections = set()

        pairs = []

        for ti, track in enumerate(
            self.tracks
        ):

            for di, detection in enumerate(
                detections
            ):

                iou = bbox_iou(
                    track.box,
                    detection["box"]
                )

                if iou >= TRACK_IOU_THRESHOLD:

                    pairs.append(
                        (
                            iou,
                            ti,
                            di
                        )
                    )

        # Highest IOU first.
        pairs.sort(
            reverse=True
        )

        for iou, ti, di in pairs:

            if ti in matched_tracks:
                continue

            if di in matched_detections:
                continue

            track = self.tracks[ti]

            detection = detections[di]

            track.update_box(
                detection["box"],
                detection["score"]
            )

            matched_tracks.add(ti)
            matched_detections.add(di)

        # Unmatched existing tracks.
        for ti, track in enumerate(
            self.tracks
        ):

            if ti not in matched_tracks:

                track.miss()

        # New detections.
        for di, detection in enumerate(
            detections
        ):

            if di not in matched_detections:

                self.tracks.append(
                    PlateTrack(
                        self.next_id,
                        detection
                    )
                )

                self.next_id += 1

        # Remove old tracks.
        self.tracks = [
            track
            for track in self.tracks
            if track.missed <= TRACK_MAX_MISSED
        ]

        return self.tracks


# ============================================================
# FFmpeg RTSP READER
# ============================================================

class FFmpegReader:

    def __init__(
        self,
        url,
        width,
        height
    ):

        self.url = url

        self.width = width
        self.height = height

        self.frame_size = (
            width *
            height *
            3
        )

        command = [
            "ffmpeg",

            "-hide_banner",
            "-loglevel",
            "error",

            "-rtsp_transport",
            "tcp",

            "-i",
            url,

            "-an",

            "-f",
            "rawvideo",

            "-pix_fmt",
            "bgr24",

            "-vf",
            f"scale={width}:{height}",

            "pipe:1"
        ]

        print()
        print("Starting FFmpeg:")
        print(" ".join(command))
        print()

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10 ** 8
        )

    def read(self):

        raw = self.process.stdout.read(
            self.frame_size
        )

        if len(raw) != self.frame_size:
            return None

        frame = np.frombuffer(
            raw,
            dtype=np.uint8
        )

        frame = frame.reshape(
            self.height,
            self.width,
            3
        )

        return frame

    def close(self):

        try:

            if self.process:

                self.process.terminate()

                try:
                    self.process.wait(
                        timeout=2
                    )

                except subprocess.TimeoutExpired:

                    self.process.kill()

        except Exception:
            pass


# ============================================================
# DEBUG IMAGE
# ============================================================

def save_debug_images(
    frame,
    detection,
    rectified,
    enhanced,
    frame_number,
    track_id
):

    if not SAVE_DEBUG:
        return

    os.makedirs(
        DEBUG_DIR,
        exist_ok=True
    )

    box = detection["box"]

    x1, y1, x2, y2 = box

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return

    cv2.imwrite(
        f"{DEBUG_DIR}/"
        f"frame_{frame_number}_"
        f"track_{track_id}_crop.jpg",
        crop
    )

    cv2.imwrite(
        f"{DEBUG_DIR}/"
        f"frame_{frame_number}_"
        f"track_{track_id}_rectified.jpg",
        rectified
    )

    cv2.imwrite(
        f"{DEBUG_DIR}/"
        f"frame_{frame_number}_"
        f"track_{track_id}_enhanced.jpg",
        enhanced
    )


# ============================================================
# DRAWING
# ============================================================

def draw_detection(
    frame,
    track,
    display_text,
    stable
):

    x1, y1, x2, y2 = track.box

    # Keep drawing simple for headless-safe processing.
    color = (
        (0, 255, 0)
        if stable
        else (0, 165, 255)
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    label = (
        display_text
        if display_text
        else f"Track {track.track_id}"
    )

    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("========================================")
    print("ATomo A311D NPU ALPR")
    print("========================================")
    print("Model  :", MODEL)
    print("Library:", LIBRARY)
    print("RTSP   :", RTSP_URL)
    print("Stream :", f"{STREAM_W}x{STREAM_H}")
    print("NPU FPS:", NPU_FPS)
    print("Target :", f"{TARGET_SIZE}x{TARGET_SIZE}")
    print("========================================")
    print()

    detector = None
    ocr = None
    reader = None

    try:

        # ----------------------------------------------------
        # NPU
        # ----------------------------------------------------

        detector = NPUDetector()

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        try:

            ocr = PlateOCR()

        except Exception as e:

            print()
            print(
                "[WARNING] OCR initialization failed:"
            )
            print(
                repr(e)
            )

            print(
                "Detection will continue without OCR."
            )

            ocr = None

        # ----------------------------------------------------
        # RTSP
        # ----------------------------------------------------

        reader = FFmpegReader(
            RTSP_URL,
            STREAM_W,
            STREAM_H
        )

        tracker = PlateTracker()

        frame_number = 0

        last_npu_time = 0.0

        npu_interval = (
            1.0 / NPU_FPS
        )

        processing_fps_start = (
            time.perf_counter()
        )

        processed_frames = 0

        print()
        print("ALPR running...")
        print("Press Ctrl+C to stop.")
        print()

        # ----------------------------------------------------
        # MAIN LOOP
        # ----------------------------------------------------

        while True:

            frame = reader.read()

            if frame is None:

                print(
                    "[ERROR] FFmpeg frame read failed."
                )

                time.sleep(0.2)

                continue

            frame_number += 1

            now = time.perf_counter()

            # Limit NPU inference frequency.
            if (
                now - last_npu_time
                < npu_interval
            ):

                continue

            last_npu_time = now

            # ------------------------------------------------
            # NPU DETECTION
            # ------------------------------------------------

            npu_start = (
                time.perf_counter()
            )

            try:

                detections = detector.detect(
                    frame
                )

            except Exception as e:

                print(
                    "[NPU ERROR]",
                    repr(e)
                )

                detections = []

            npu_ms = (
                time.perf_counter()
                - npu_start
            ) * 1000.0

            # ------------------------------------------------
            # TRACKING
            # ------------------------------------------------

            tracks = tracker.update(
                detections
            )

            # ------------------------------------------------
            # OCR EACH ACTIVE TRACK
            # ------------------------------------------------

            for track in tracks:

                # Do not OCR tracks that disappeared.
                if track.missed > 0:
                    continue

                x1, y1, x2, y2 = (
                    track.box
                )

                x1 = max(
                    0,
                    min(
                        STREAM_W - 1,
                        x1
                    )
                )

                y1 = max(
                    0,
                    min(
                        STREAM_H - 1,
                        y1
                    )
                )

                x2 = max(
                    0,
                    min(
                        STREAM_W,
                        x2
                    )
                )

                y2 = max(
                    0,
                    min(
                        STREAM_H,
                        y2
                    )
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[
                    y1:y2,
                    x1:x2
                ]

                if crop.size == 0:
                    continue

                # --------------------------------------------
                # Perspective correction
                # --------------------------------------------

                try:

                    rectified = perspective_correct(
                        crop
                    )

                    enhanced = enhance_plate(
                        rectified
                    )

                except Exception as e:

                    print(
                        "[PREPROCESS ERROR]",
                        repr(e)
                    )

                    continue

                # --------------------------------------------
                # OCR
                # --------------------------------------------

                raw_text = ""
                raw_conf = 0.0
                ocr_ms = 0.0

                if ocr is not None:

                    (
                        raw_text,
                        raw_conf,
                        ocr_ms
                    ) = ocr.recognize(
                        enhanced
                    )

                    if raw_text:

                        track.add_ocr(
                            raw_text,
                            raw_conf
                        )

                # --------------------------------------------
                # Get temporal best result
                # --------------------------------------------

                (
                    best_text,
                    best_conf,
                    stable,
                    evidence
                ) = track.get_best_result()

                # --------------------------------------------
                # Save debug
                # --------------------------------------------

                save_debug_images(
                    frame,
                    {
                        "box": [
                            x1,
                            y1,
                            x2,
                            y2
                        ],
                        "score":
                            track.detection_conf
                    },
                    rectified,
                    enhanced,
                    frame_number,
                    track.track_id
                )

                # --------------------------------------------
                # OUTPUT
                # --------------------------------------------

                if PRINT_EVERY_FRAME:

                    raw_display = (
                        normalize_ocr(
                            raw_text
                        )
                        if raw_text
                        else "-"
                    )

                    validity = (
                        "YES"
                        if is_indian_plate(
                            best_text
                        )
                        else "NO"
                    )

                    print(
                        f"Frame {frame_number}: "
                        f"Track {track.track_id} | "
                        f"Plate conf="
                        f"{track.detection_conf:.3f} | "
                        f"Raw={raw_display:<14} | "
                        f"Best={best_text or '-':<14} | "
                        f"OCR="
                        f"{raw_conf:.3f} | "
                        f"Indian={validity} | "
                        f"Stable="
                        f"{stable} | "
                        f"Evidence="
                        f"{evidence:.2f} | "
                        f"NPU="
                        f"{npu_ms:.1f}ms | "
                        f"OCR="
                        f"{ocr_ms:.1f}ms"
                    )

                    # Print a clear stable result.
                    if stable:

                        print(
                            f"  >>> STABLE PLATE: "
                            f"{best_text} "
                            f"(confidence "
                            f"{best_conf:.3f})"
                        )

                # Optional GUI
                if SHOW_WINDOW:

                    draw_detection(
                        frame,
                        track,
                        (
                            best_text
                            if stable
                            else ""
                        ),
                        stable
                    )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            processed_frames += 1

            elapsed = (
                time.perf_counter()
                - processing_fps_start
            )

            if elapsed >= 5.0:

                fps = (
                    processed_frames
                    / elapsed
                )

                print(
                    f"[PERFORMANCE] "
                    f"Processing FPS: "
                    f"{fps:.2f}"
                )

                processed_frames = 0

                processing_fps_start = (
                    time.perf_counter()
                )

            # ------------------------------------------------
            # GUI
            # ------------------------------------------------

            if SHOW_WINDOW:

                cv2.imshow(
                    "A311D NPU ALPR",
                    frame
                )

                key = cv2.waitKey(1)

                if key == 27:
                    break

    except KeyboardInterrupt:

        print()
        print(
            "Stopping ALPR..."
        )

    except Exception as e:

        print()
        print(
            "[FATAL ERROR]"
        )
        print(
            repr(e)
        )

        raise

    finally:

        print()
        print(
            "Cleaning up..."
        )

        if reader is not None:

            reader.close()

        if detector is not None:

            detector.release()

        if SHOW_WINDOW:

            cv2.destroyAllWindows()

        print(
            "ALPR stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import re
import time
import os

from ultralytics import YOLO
from paddleocr import TextRecognition


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolov11s.pt"

# Change this to your original vehicle image
IMAGE_PATH = "1.jpeg"

OUTPUT_DIR = "alpr_output"

# YOLO confidence
YOLO_CONF = 0.25

# OCR model
OCR_MODEL = "PP-OCRv5_mobile_rec"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}"
    )

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Input image not found: {IMAGE_PATH}"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError(
        f"Could not read image: {IMAGE_PATH}"
    )

original = image.copy()

print("=" * 60)
print("AT0MO ALPR TEST")
print("=" * 60)

print(f"Image : {IMAGE_PATH}")
print(f"Size  : {image.shape[1]}x{image.shape[0]}")


# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading YOLO model...")

yolo = YOLO(MODEL_PATH)

print("YOLO loaded.")


# ============================================================
# LOAD OCR
# ============================================================

print("\nLoading OCR model...")

ocr = TextRecognition(
    model_name=OCR_MODEL
)

print("OCR loaded.")


# ============================================================
# RUN YOLO
# ============================================================

print("\nRunning license plate detection...")

yolo_start = time.perf_counter()

results = yolo.predict(
    source=image,
    conf=YOLO_CONF,
    verbose=False
)

yolo_time = (
    time.perf_counter() - yolo_start
) * 1000


# ============================================================
# CHECK DETECTIONS
# ============================================================

if len(results) == 0:
    print("No YOLO result.")
    raise SystemExit

result = results[0]

if result.boxes is None or len(result.boxes) == 0:
    print("No license plate detected.")
    raise SystemExit


print(
    f"YOLO inference: {yolo_time:.1f} ms"
)

print(
    f"Detected plates: {len(result.boxes)}"
)


# ============================================================
# PROCESS EACH DETECTED PLATE
# ============================================================

for plate_index, box in enumerate(result.boxes):

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    x1, y1, x2, y2 = (
        box.xyxy[0]
        .cpu()
        .numpy()
        .astype(int)
    )

    confidence = float(
        box.conf[0].cpu().numpy()
    )

    # Clamp coordinates
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        continue


    # --------------------------------------------------------
    # Add small padding around YOLO box
    # --------------------------------------------------------

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * 0.03)
    pad_y = int(bh * 0.03)

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)

    cx2 = min(
        image.shape[1],
        x2 + pad_x
    )

    cy2 = min(
        image.shape[0],
        y2 + pad_y
    )


    crop = original[
        cy1:cy2,
        cx1:cx2
    ].copy()


    if crop.size == 0:
        continue


    print("\n" + "-" * 60)

    print(
        f"Plate #{plate_index}"
    )

    print(
        f"YOLO confidence: {confidence:.4f}"
    )

    print(
        f"Bounding box: "
        f"({x1},{y1}) - ({x2},{y2})"
    )

    print(
        f"Crop size: "
        f"{crop.shape[1]}x{crop.shape[0]}"
    )


    # --------------------------------------------------------
    # Save YOLO crop
    # --------------------------------------------------------

    crop_path = os.path.join(
        OUTPUT_DIR,
        f"plate_{plate_index}_crop.jpg"
    )

    cv2.imwrite(
        crop_path,
        crop
    )


    # ========================================================
    # AUTOMATIC PERSPECTIVE CORRECTION
    # ========================================================

    crop_h, crop_w = crop.shape[:2]


    def find_plate_corners(img):
        """
        Try to find the rectangular license plate
        using contours.
        """

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # Slight blur removes tiny image noise
        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            blurred,
            50,
            150
        )

        # Connect broken plate edges
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 3)
        )

        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < crop_w * crop_h * 0.08:
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

            pts = approx.reshape(4, 2).astype(
                np.float32
            )

            x, y, w, h = cv2.boundingRect(
                approx
            )

            if h == 0:
                continue

            aspect = w / float(h)

            # Typical plate-like rectangle.
            # This is deliberately broad.
            if aspect < 1.8 or aspect > 7.0:
                continue

            rectangular_area = w * h

            fill_ratio = (
                area / float(rectangular_area)
                if rectangular_area > 0
                else 0
            )

            if fill_ratio < 0.35:
                continue

            score = area * min(
                aspect,
                5.0
            )

            candidates.append(
                (score, pts)
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        return candidates[0][1]


    def order_points(points):
        """
        Return:
        top-left
        top-right
        bottom-right
        bottom-left
        """

        points = np.asarray(
            points,
            dtype=np.float32
        )

        ordered = np.zeros(
            (4, 2),
            dtype=np.float32
        )

        sums = points.sum(axis=1)
        diffs = np.diff(
            points,
            axis=1
        ).reshape(-1)

        ordered[0] = points[
            np.argmin(sums)
        ]

        ordered[2] = points[
            np.argmax(sums)
        ]

        ordered[1] = points[
            np.argmin(diffs)
        ]

        ordered[3] = points[
            np.argmax(diffs)
        ]

        return ordered


    corners = find_plate_corners(
        crop
    )


    # ========================================================
    # FALLBACK
    #
    # If contour detection fails, use a proportional
    # quadrilateral inside the YOLO crop.
    # ========================================================

    if corners is None:

        print(
            "Automatic corner detection: "
            "not found"
        )

        print(
            "Using fallback plate geometry."
        )

        corners = np.float32([
            [
                crop_w * 0.12,
                crop_h * 0.19
            ],
            [
                crop_w * 0.93,
                crop_h * 0.33
            ],
            [
                crop_w * 0.90,
                crop_h * 0.79
            ],
            [
                crop_w * 0.08,
                crop_h * 0.67
            ]
        ])

    else:

        print(
            "Automatic corner detection: OK"
        )

        corners = order_points(
            corners
        )


    print(
        "Corners:"
    )

    for i, point in enumerate(corners):

        print(
            f"  {i}: "
            f"({point[0]:.1f}, "
            f"{point[1]:.1f})"
        )


    # ========================================================
    # DRAW CORNERS FOR DEBUGGING
    # ========================================================

    debug_crop = crop.copy()

    for i, point in enumerate(corners):

        px = int(point[0])
        py = int(point[1])

        cv2.circle(
            debug_crop,
            (px, py),
            3,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            debug_crop,
            str(i),
            (px + 3, py - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )


    debug_path = os.path.join(
        OUTPUT_DIR,
        f"plate_{plate_index}_corners.jpg"
    )

    cv2.imwrite(
        debug_path,
        debug_crop
    )


    # ========================================================
    # RECTIFY PLATE
    # ========================================================

    output_width = 320
    output_height = 80

    destination = np.float32([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ])


    matrix = cv2.getPerspectiveTransform(
        corners,
        destination
    )


    rectified = cv2.warpPerspective(
        crop,
        matrix,
        (
            output_width,
            output_height
        )
    )


    rectified_path = os.path.join(
        OUTPUT_DIR,
        f"plate_{plate_index}_rectified.jpg"
    )

    cv2.imwrite(
        rectified_path,
        rectified
    )


    # ========================================================
    # CLAHE
    # ========================================================

    gray = cv2.cvtColor(
        rectified,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_gray = clahe.apply(
        gray
    )

    # Convert back to 3-channel
    enhanced = cv2.cvtColor(
        enhanced_gray,
        cv2.COLOR_GRAY2BGR
    )


    enhanced_path = os.path.join(
        OUTPUT_DIR,
        f"plate_{plate_index}_enhanced.jpg"
    )

    cv2.imwrite(
        enhanced_path,
        enhanced
    )


    # ========================================================
    # OCR
    # ========================================================

    print(
        "\nRunning OCR..."
    )

    ocr_start = time.perf_counter()

    ocr_results = ocr.predict(
        enhanced
    )

    ocr_time = (
        time.perf_counter()
        - ocr_start
    ) * 1000


    raw_text = ""
    ocr_score = 0.0


    for ocr_result in ocr_results:

        raw_text = ocr_result.get(
            "rec_text",
            ""
        )

        ocr_score = float(
            ocr_result.get(
                "rec_score",
                0.0
            )
        )


    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    normalized = raw_text.upper()

    normalized = re.sub(
        r"[^A-Z0-9]",
        "",
        normalized
    )


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print("\n" + "=" * 60)

    print("ALPR RESULT")

    print("=" * 60)

    print(
        f"Plate #{plate_index}"
    )

    print(
        f"YOLO confidence : "
        f"{confidence:.4f}"
    )

    print(
        f"Raw OCR         : "
        f"{raw_text}"
    )

    print(
        f"Normalized      : "
        f"{normalized}"
    )

    print(
        f"OCR confidence  : "
        f"{ocr_score:.4f}"
    )

    print(
        f"OCR time        : "
        f"{ocr_time:.1f} ms"
    )

    print("=" * 60)


# ============================================================
# SAVE YOLO ANNOTATED IMAGE
# ============================================================

annotated = results[0].plot()

annotated_path = os.path.join(
    OUTPUT_DIR,
    "detections.jpg"
)

cv2.imwrite(
    annotated_path,
    annotated
)


# ============================================================
# FINISHED
# ============================================================

print("\nOutput files saved in:")

print(
    f"  {OUTPUT_DIR}/"
)

print(
    "\nUseful files:"
)

print(
    "  detections.jpg"
)

print(
    "  plate_0_crop.jpg"
)

print(
    "  plate_0_corners.jpg"
)

print(
    "  plate_0_rectified.jpg"
)

print(
    "  plate_0_enhanced.jpg"
)

print("\nALPR test complete.")


import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    PROJECT_ROOT
    / "data_processing"
    / "data"
    / "bigearthnet_rgb"
    / "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_32_64_RGB.png"
)

QUERY = (
    "Can you point out the smallest contiguous "
    "area of industrial or commercial units in this image?"
)

# BigEarthNet normalized coordinates:
# [x1, y1, x2, y2]
GROUND_TRUTH_NORMALIZED = [
    0.51,
    0.90,
    0.90,
    1.00,
]


def normalized_to_pixel(
    bbox,
    width,
    height,
):
    """Convert normalized bbox to pixel coordinates."""

    x1 = round(
        bbox[0] * (width - 1)
    )

    y1 = round(
        bbox[1] * (height - 1)
    )

    x2 = round(
        bbox[2] * (width - 1)
    )

    y2 = round(
        bbox[3] * (height - 1)
    )

    return [
        x1,
        y1,
        x2,
        y2,
    ]


def calculate_iou(
    box1,
    box2,
):
    """Calculate IoU between two pixel bounding boxes."""

    x1 = max(
        box1[0],
        box2[0],
    )

    y1 = max(
        box1[1],
        box2[1],
    )

    x2 = min(
        box1[2],
        box2[2],
    )

    y2 = min(
        box1[3],
        box2[3],
    )

    intersection_width = max(
        0,
        x2 - x1 + 1,
    )

    intersection_height = max(
        0,
        y2 - y1 + 1,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area1 = (
        (box1[2] - box1[0] + 1)
        * (box1[3] - box1[1] + 1)
    )

    area2 = (
        (box2[2] - box2[0] + 1)
        * (box2[3] - box2[1] + 1)
    )

    union_area = (
        area1
        + area2
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return (
        intersection_area
        / union_area
    )


def main():

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"RGB image not found: {IMAGE_PATH}"
        )

    # Import here so the script starts cleanly.
    from models.grounding.ground_query import (
        ground_query,
    )

    print(
        "Image:",
        IMAGE_PATH,
    )

    print(
        "Query:",
        QUERY,
    )

    # -----------------------------------------------------
    # Run model
    # -----------------------------------------------------

    result = ground_query(
        str(IMAGE_PATH),
        QUERY,
    )

    print()
    print(
        "Model result:"
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    # -----------------------------------------------------
    # Image dimensions
    # -----------------------------------------------------

    width = result["image_width"]
    height = result["image_height"]

    ground_truth = normalized_to_pixel(
        GROUND_TRUTH_NORMALIZED,
        width,
        height,
    )

    print()
    print(
        "Ground-truth bbox:",
        ground_truth,
    )

    # -----------------------------------------------------
    # Check predictions
    # -----------------------------------------------------

    predictions = result[
        "bounding_boxes"
    ]

    if not predictions:

        print()
        print(
            "No predicted bounding boxes."
        )

        return

    # -----------------------------------------------------
    # Find best IoU
    # -----------------------------------------------------

    best_iou = 0.0
    best_prediction = None

    for prediction in predictions:

        predicted_bbox = prediction[
            "bbox"
        ]

        iou = calculate_iou(
            predicted_bbox,
            ground_truth,
        )

        print()
        print(
            "Prediction:",
            predicted_bbox,
        )

        print(
            "Confidence:",
            prediction[
                "confidence"
            ],
        )

        print(
            f"IoU: {iou:.4f}"
        )

        if iou > best_iou:

            best_iou = iou
            best_prediction = prediction

    # -----------------------------------------------------
    # Final evaluation
    # -----------------------------------------------------

    print()
    print(
        "=============================="
    )

    print(
        f"Best IoU: {best_iou:.4f}"
    )

    if best_iou >= 0.50:

        print(
            "Result: PASS (IoU >= 0.50)"
        )

    else:

        print(
            "Result: FAIL (IoU < 0.50)"
        )

    print(
        "=============================="
    )

    if best_prediction is not None:

        print()
        print(
            "Best prediction:"
        )

        print(
            json.dumps(
                best_prediction,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
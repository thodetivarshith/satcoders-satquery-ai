from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from models.grounding.ground_query import get_models
from segment_anything import SamAutomaticMaskGenerator


IMAGE_PATH = Path(
    "data_processing/data/bigearthnet_rgb/"
    "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_32_64_RGB.png"
)

GT_BOX = [61, 107, 107, 119]


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def union_box(box_a, box_b):
    return [
        min(box_a[0], box_b[0]),
        min(box_a[1], box_b[1]),
        max(box_a[2], box_b[2]),
        max(box_a[3], box_b[3]),
    ]


def boxes_are_close(box_a, box_b, max_gap=5):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    horizontal_gap = max(
        0,
        max(ax1, bx1) - min(ax2, bx2)
    )

    vertical_gap = max(
        0,
        max(ay1, by1) - min(ay2, by2)
    )

    return horizontal_gap <= max_gap and vertical_gap <= max_gap


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {IMAGE_PATH}"
        )

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"Image: {IMAGE_PATH.resolve()}")
    print(f"Image size: {image_rgb.shape[1]}x{image_rgb.shape[0]}")

    sam, _ = get_models()

    print("\nGenerating SAM candidates...")

    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.85,
        min_mask_region_area=100,
    )

    masks = generator.generate(image_rgb)

    print(f"SAM generated {len(masks)} candidates.")

    boxes = []

    for idx, mask_data in enumerate(masks):
        segmentation = mask_data["segmentation"]

        ys, xs = segmentation.nonzero()

        if len(xs) == 0:
            continue

        box = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ]

        boxes.append(
            {
                "id": idx,
                "bbox": box,
                "area": int(segmentation.sum()),
            }
        )

    # ---------------------------------------------------------
    # Individual candidate evaluation
    # ---------------------------------------------------------

    individual = []

    for item in boxes:
        iou = compute_iou(item["bbox"], GT_BOX)

        individual.append(
            {
                **item,
                "iou": iou,
            }
        )

    individual.sort(
        key=lambda x: x["iou"],
        reverse=True,
    )

    print("\nBest individual candidates:")
    print("=" * 70)

    for item in individual[:10]:
        print(
            f"Candidate {item['id']:3d} | "
            f"bbox={item['bbox']} | "
            f"IoU={item['iou']:.4f}"
        )

    # ---------------------------------------------------------
    # Pairwise merging
    # ---------------------------------------------------------

    merged = []

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):

            box_a = boxes[i]["bbox"]
            box_b = boxes[j]["bbox"]

            if not boxes_are_close(
                box_a,
                box_b,
                max_gap=5,
            ):
                continue

            merged_box = union_box(
                box_a,
                box_b,
            )

            iou = compute_iou(
                merged_box,
                GT_BOX,
            )

            merged.append(
                {
                    "a": boxes[i]["id"],
                    "b": boxes[j]["id"],
                    "bbox": merged_box,
                    "iou": iou,
                }
            )

    merged.sort(
        key=lambda x: x["iou"],
        reverse=True,
    )

    print("\nBest merged candidates:")
    print("=" * 70)

    for item in merged[:20]:
        print(
            f"Candidates {item['a']} + {item['b']} | "
            f"bbox={item['bbox']} | "
            f"IoU={item['iou']:.4f}"
        )

    best_merged_iou = (
        merged[0]["iou"]
        if merged
        else 0.0
    )

    print("\n" + "=" * 70)
    print(
        f"Best individual IoU : "
        f"{individual[0]['iou']:.4f}"
    )
    print(
        f"Best merged IoU     : "
        f"{best_merged_iou:.4f}"
    )
    print("=" * 70)

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(image_rgb)

    # Ground truth
    gx1, gy1, gx2, gy2 = GT_BOX

    gt_rect = plt.Rectangle(
        (gx1, gy1),
        gx2 - gx1,
        gy2 - gy1,
        fill=False,
        linewidth=3,
    )

    ax.add_patch(gt_rect)

    ax.text(
        gx1,
        max(0, gy1 - 3),
        "GROUND TRUTH",
        fontsize=9,
        backgroundcolor="white",
    )

    # Show best merged boxes
    for rank, item in enumerate(merged[:10]):

        x1, y1, x2, y2 = item["bbox"]

        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=1.5,
        )

        ax.add_patch(rect)

        ax.text(
            x1,
            y1,
            f"M{rank + 1}:{item['iou']:.2f}",
            fontsize=7,
            backgroundcolor="white",
        )

    ax.set_title(
        "SAM Candidate Merging vs Ground Truth"
    )

    ax.axis("off")

    output_path = Path(
        "data_processing/data/"
        "sam_merged_candidates.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nVisualization saved to:\n"
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
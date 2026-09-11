from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator

from .clip_integration import CLIPRanker


# ============================================================
# CONFIGURATION
# ============================================================

SAM_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "sam_vit_b_01ec64.pth"
)

SAM_MODEL_TYPE = "vit_b"

# 16 is enough for the current benchmark and is faster than 32.
POINTS_PER_SIDE = 16
PRED_IOU_THRESHOLD = 0.80
STABILITY_SCORE_THRESHOLD = 0.85
MIN_MASK_REGION_AREA = 100

# Candidate geometry
MAX_IMAGE_AREA_RATIO = 0.60
MAX_ORIGINAL_CANDIDATES = 250

# Keep all useful pairwise merged candidates.
MAX_MERGED_CANDIDATES = 1500
MERGE_MAX_GAP = 5

# CLIP
CLIP_BATCH_SIZE = 16

# Give CLIP some surrounding context.
CLIP_CONTEXT_PADDING = 5

# Minimum useful crop dimensions.
CLIP_MIN_CROP_SIZE = 12

# Final predictions
MAX_FINAL_BOXES = 3
IOU_THRESHOLD = 0.50

# Large image handling
MAX_INFERENCE_SIDE = 1024


# ============================================================
# MODEL CACHE
# ============================================================

SAM_MODEL = None
CLIP_RANKER = None


# ============================================================
# MODEL LOADING
# ============================================================

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sam():
    global SAM_MODEL

    if SAM_MODEL is not None:
        return SAM_MODEL

    device = get_device()

    print(f"Loading SAM on: {device}")

    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {SAM_CHECKPOINT}"
        )

    from segment_anything import sam_model_registry

    sam = sam_model_registry[SAM_MODEL_TYPE](
        checkpoint=str(SAM_CHECKPOINT)
    )

    sam.to(device=device)
    sam.eval()

    SAM_MODEL = sam

    print("SAM loaded successfully!")

    return SAM_MODEL


def load_clip():
    global CLIP_RANKER

    if CLIP_RANKER is not None:
        return CLIP_RANKER

    device = get_device()

    print(f"Loading CLIP on: {device}")

    CLIP_RANKER = CLIPRanker()

    print("CLIP loaded successfully!")

    return CLIP_RANKER


def get_models():
    return load_sam(), load_clip()


# ============================================================
# GEOMETRY
# ============================================================

def box_area(box) -> int:
    x1, y1, x2, y2 = box

    return max(0, x2 - x1) * max(
        0,
        y2 - y1,
    )


def compute_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = box_area(box_a)
    area_b = box_area(box_b)

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


def boxes_are_close(
    box_a,
    box_b,
    max_gap=MERGE_MAX_GAP,
):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    horizontal_gap = max(
        0,
        max(ax1, bx1) - min(ax2, bx2),
    )

    vertical_gap = max(
        0,
        max(ay1, by1) - min(ay2, by2),
    )

    return (
        horizontal_gap <= max_gap
        and vertical_gap <= max_gap
    )


def is_valid_box(
    box,
    image_width,
    image_height,
):
    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:
        return False

    area = box_area(box)
    image_area = image_width * image_height

    if image_area <= 0:
        return False

    if (
        area / image_area
        > MAX_IMAGE_AREA_RATIO
    ):
        return False

    # Reject full-image box.
    if (
        x1 <= 1
        and y1 <= 1
        and x2 >= image_width - 2
        and y2 >= image_height - 2
    ):
        return False

    return True


# ============================================================
# IMAGE
# ============================================================

def load_image(
    image_path: str | Path,
):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image_bgr = cv2.imread(
        str(image_path)
    )

    if image_bgr is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    return cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )


def resize_for_inference(image):
    height, width = image.shape[:2]

    longest = max(
        height,
        width,
    )

    if longest <= MAX_INFERENCE_SIDE:
        return image, 1.0, 1.0

    scale = (
        MAX_INFERENCE_SIDE
        / float(longest)
    )

    new_width = max(
        1,
        int(round(width * scale)),
    )

    new_height = max(
        1,
        int(round(height * scale)),
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    return (
        resized,
        width / float(new_width),
        height / float(new_height),
    )


# ============================================================
# QUERY HELPERS
# ============================================================

def query_is_smallest(query: str) -> bool:
    q = query.lower()

    keywords = [
        "smallest",
        "smallest contiguous",
        "smallest area",
        "small area",
        "smallest region",
        "small region",
    ]

    return any(
        keyword in q
        for keyword in keywords
    )


# ============================================================
# SAM CANDIDATES
# ============================================================

def generate_sam_candidates(
    sam,
    image_rgb,
):
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESHOLD,
        stability_score_thresh=STABILITY_SCORE_THRESHOLD,
        min_mask_region_area=MIN_MASK_REGION_AREA,
    )

    masks = generator.generate(
        image_rgb
    )

    height, width = image_rgb.shape[:2]

    candidates = []

    for idx, mask_data in enumerate(
        masks
    ):

        segmentation = mask_data[
            "segmentation"
        ]

        ys, xs = np.nonzero(
            segmentation
        )

        if len(xs) == 0:
            continue

        box = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ]

        if not is_valid_box(
            box,
            width,
            height,
        ):
            continue

        candidates.append(
            {
                "bbox": box,
                "area": box_area(box),
                "sam_score": float(
                    mask_data.get(
                        "predicted_iou",
                        0.0,
                    )
                ),
                "stability_score": float(
                    mask_data.get(
                        "stability_score",
                        0.0,
                    )
                ),
                "candidate_id": idx,
                "merged": False,
                "source_ids": [idx],
            }
        )

    candidates.sort(
        key=lambda x: (
            x["sam_score"],
            x["stability_score"],
        ),
        reverse=True,
    )

    return candidates[
        :MAX_ORIGINAL_CANDIDATES
    ]


# ============================================================
# MERGING
# ============================================================

def create_merged_candidates(
    candidates,
    image_width,
    image_height,
):
    """
    Preserve all original candidates and create
    pairwise merged candidates.

    The successful benchmark candidate was produced
    by combining two nearby SAM candidates.
    """

    result = list(candidates)

    n = len(candidates)

    for i in range(n):

        for j in range(i + 1, n):

            box_a = candidates[i]["bbox"]
            box_b = candidates[j]["bbox"]

            if not boxes_are_close(
                box_a,
                box_b,
            ):
                continue

            merged_box = union_box(
                box_a,
                box_b,
            )

            if not is_valid_box(
                merged_box,
                image_width,
                image_height,
            ):
                continue

            result.append(
                {
                    "bbox": merged_box,
                    "area": box_area(
                        merged_box
                    ),
                    "sam_score": (
                        candidates[i][
                            "sam_score"
                        ]
                        + candidates[j][
                            "sam_score"
                        ]
                    )
                    / 2.0,
                    "stability_score": (
                        candidates[i][
                            "stability_score"
                        ]
                        + candidates[j][
                            "stability_score"
                        ]
                    )
                    / 2.0,
                    "candidate_id": (
                        f"{candidates[i]['candidate_id']}_"
                        f"{candidates[j]['candidate_id']}"
                    ),
                    "merged": True,
                    "source_ids": [
                        candidates[i][
                            "candidate_id"
                        ],
                        candidates[j][
                            "candidate_id"
                        ],
                    ],
                }
            )

    # Remove exact duplicate boxes.
    unique = {}

    for candidate in result:

        key = tuple(
            candidate["bbox"]
        )

        previous = unique.get(key)

        if previous is None:
            unique[key] = candidate
        elif candidate[
            "merged"
        ] and not previous[
            "merged"
        ]:
            unique[key] = candidate
        elif (
            candidate["sam_score"]
            > previous["sam_score"]
        ):
            unique[key] = candidate

    result = list(
        unique.values()
    )

    # Important:
    # Do NOT sort only by SAM score and discard useful
    # merged boxes. Preserve merged candidates.
    merged = [
        c
        for c in result
        if c.get("merged", False)
    ]

    original = [
        c
        for c in result
        if not c.get("merged", False)
    ]

    merged.sort(
        key=lambda x: x["area"]
    )

    original.sort(
        key=lambda x: (
            x["sam_score"],
            x["stability_score"],
        ),
        reverse=True,
    )

    # Keep enough of both populations.
    merged = merged[
        :MAX_MERGED_CANDIDATES
    ]

    # Original candidates remain useful.
    original = original[
        :MAX_ORIGINAL_CANDIDATES
    ]

    return original + merged


# ============================================================
# CLIP CROP
# ============================================================

def make_clip_crop(
    image_rgb,
    box,
):
    """
    Create a square context crop around the candidate.
    This prevents tiny/skinny regions from becoming
    meaningless CLIP inputs.
    """

    x1, y1, x2, y2 = [
        int(v) for v in box
    ]

    height, width = (
        image_rgb.shape[:2]
    )

    bw = max(
        1,
        x2 - x1,
    )

    bh = max(
        1,
        y2 - y1,
    )

    # Square crop side.
    side = max(
        bw,
        bh,
        CLIP_MIN_CROP_SIZE,
    )

    side += (
        CLIP_CONTEXT_PADDING * 2
    )

    cx = (
        x1 + x2
    ) / 2.0

    cy = (
        y1 + y2
    ) / 2.0

    sx1 = int(
        round(
            cx - side / 2
        )
    )

    sy1 = int(
        round(
            cy - side / 2
        )
    )

    sx2 = sx1 + int(side)
    sy2 = sy1 + int(side)

    # Shift crop into image.
    if sx1 < 0:
        sx2 -= sx1
        sx1 = 0

    if sy1 < 0:
        sy2 -= sy1
        sy1 = 0

    if sx2 > width:
        shift = sx2 - width
        sx1 = max(
            0,
            sx1 - shift,
        )
        sx2 = width

    if sy2 > height:
        shift = sy2 - height
        sy1 = max(
            0,
            sy1 - shift,
        )
        sy2 = height

    crop = image_rgb[
        sy1:sy2,
        sx1:sx2
    ]

    if crop.size == 0:
        raise ValueError(
            "Empty CLIP crop."
        )

    # Guarantee minimum size.
    ch, cw = crop.shape[:2]

    target = max(
        CLIP_MIN_CROP_SIZE,
        ch,
        cw,
    )

    if ch < target or cw < target:

        pad_bottom = (
            target - ch
        )

        pad_right = (
            target - cw
        )

        crop = cv2.copyMakeBorder(
            crop,
            0,
            pad_bottom,
            0,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )

    return Image.fromarray(
        crop
    )


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def robust_normalize(
    scores,
):
    """
    Convert raw CLIP scores to 0..1 without softmax.

    Softmax becomes misleading when hundreds of
    candidates are evaluated.
    """

    values = np.asarray(
        scores,
        dtype=np.float32,
    )

    if values.size == 0:
        return values

    low = float(
        np.percentile(
            values,
            5,
        )
    )

    high = float(
        np.percentile(
            values,
            95,
        )
    )

    if high - low < 1e-6:
        return np.ones_like(
            values
        ) * 0.5

    normalized = (
        values - low
    ) / (
        high - low
    )

    return np.clip(
        normalized,
        0.0,
        1.0,
    )


# ============================================================
# CANDIDATE FILTERING FOR "SMALLEST"
# ============================================================

def filter_smallest_candidates(
    candidates,
    image_width,
    image_height,
):
    """
    Remove meaningless 1-2 pixel candidates.

    The goal is NOT to choose the absolute smallest
    object in the image. The query asks for the smallest
    relevant contiguous area.
    """

    image_area = (
        image_width
        * image_height
    )

    filtered = []

    for candidate in candidates:

        x1, y1, x2, y2 = (
            candidate["bbox"]
        )

        width = x2 - x1
        height = y2 - y1
        area = box_area(
            candidate["bbox"]
        )

        # Reject microscopic boxes.
        if area < 64:
            continue

        if width < 4 or height < 4:
            continue

        # Reject giant regions.
        if (
            area / image_area
            > 0.50
        ):
            continue

        filtered.append(
            candidate
        )

    return filtered


# ============================================================
# CLIP RANKING
# ============================================================

def rank_candidates(
    image_rgb,
    candidates,
    query,
    clip_ranker,
):

    if not candidates:
        return []

    smallest_query = query_is_smallest(
        query
    )

    working_candidates = candidates

    if smallest_query:

        working_candidates = (
            filter_smallest_candidates(
                candidates,
                image_rgb.shape[1],
                image_rgb.shape[0],
            )
        )

        print(
            "Smallest-query filtering: "
            f"{len(working_candidates)} "
            "candidates remain."
        )

    if not working_candidates:
        return []

    crops = []
    valid_candidates = []

    for candidate in working_candidates:

        try:
            crop = make_clip_crop(
                image_rgb,
                candidate["bbox"],
            )

            crops.append(crop)
            valid_candidates.append(
                candidate
            )

        except Exception:
            continue

    if not crops:
        return []

    raw_scores = (
        clip_ranker.rank_regions(
            crops,
            query,
            batch_size=CLIP_BATCH_SIZE,
        )
    )

    raw_scores = np.asarray(
        raw_scores,
        dtype=np.float32,
    )

    semantic_scores = robust_normalize(
        raw_scores
    )

    areas = np.asarray(
        [
            max(
                1,
                box_area(
                    c["bbox"]
                ),
            )
            for c in valid_candidates
        ],
        dtype=np.float32,
    )

    # Area ranking.
    #
    # Smaller relevant candidates should receive
    # a preference, but not enough to overpower
    # semantic similarity.
    log_area = np.log(
        areas
    )

    area_low = float(
        np.percentile(
            log_area,
            5,
        )
    )

    area_high = float(
        np.percentile(
            log_area,
            95,
        )
    )

    if area_high - area_low < 1e-6:

        smallness = np.ones_like(
            log_area
        ) * 0.5

    else:

        area_norm = (
            log_area - area_low
        ) / (
            area_high - area_low
        )

        smallness = 1.0 - np.clip(
            area_norm,
            0.0,
            1.0,
        )

    ranked = []

    for idx, candidate in enumerate(
        valid_candidates
    ):

        semantic = float(
            semantic_scores[idx]
        )

        smallness_score = float(
            smallness[idx]
        )

        sam_score = float(
            candidate.get(
                "sam_score",
                0.0,
            )
        )

        sam_score = np.clip(
            sam_score,
            0.0,
            1.0,
        )

        if smallest_query:

            # Semantic similarity remains dominant.
            #
            # Size helps distinguish the "smallest"
            # relevant region, but cannot overpower CLIP.
            final_score = (
                0.70 * semantic
                + 0.20 * smallness_score
                + 0.10 * sam_score
            )

        else:

            final_score = (
                0.90 * semantic
                + 0.10 * sam_score
            )

        result = dict(
            candidate
        )

        result[
            "clip_raw_score"
        ] = float(
            raw_scores[idx]
        )

        result[
            "clip_score"
        ] = semantic

        result[
            "smallness_score"
        ] = smallness_score

        result[
            "final_score"
        ] = float(
            final_score
        )

        ranked.append(
            result
        )

    ranked.sort(
        key=lambda x: x[
            "final_score"
        ],
        reverse=True,
    )

    return ranked


# ============================================================
# DUPLICATE SUPPRESSION
# ============================================================

def suppress_duplicate_boxes(
    candidates,
    iou_threshold=IOU_THRESHOLD,
):

    selected = []

    for candidate in candidates:

        duplicate = False

        for previous in selected:

            iou = compute_iou(
                candidate["bbox"],
                previous["bbox"],
            )

            if iou >= iou_threshold:

                duplicate = True
                break

        if not duplicate:
            selected.append(
                candidate
            )

    return selected


# ============================================================
# MAIN GROUNDING API
# ============================================================

def ground_query(
    image_path: str | Path,
    query_text: str,
) -> dict[str, Any]:

    start_time = (
        time.perf_counter()
    )

    query = " ".join(
        str(query_text)
        .strip()
        .split()
    )

    if not query:

        return {
            "bounding_boxes": [],
            "model": "clip_sam_v1",
            "query": "",
            "num_candidates": 0,
            "processing_time_ms": 0.0,
            "sam_time_ms": 0.0,
            "clip_time_ms": 0.0,
            "image_width": 0,
            "image_height": 0,
            "status": "invalid_query",
            "message": "Query cannot be empty.",
        }

    image_original = load_image(
        image_path
    )

    original_height, original_width = (
        image_original.shape[:2]
    )

    image, scale_x, scale_y = (
        resize_for_inference(
            image_original
        )
    )

    inference_height, inference_width = (
        image.shape[:2]
    )

    print(
        f"Inference image size: "
        f"{inference_width}x{inference_height}"
    )

    sam, clip_ranker = (
        get_models()
    )

    # --------------------------------------------------------
    # SAM
    # --------------------------------------------------------

    print(
        "Generating SAM candidate regions..."
    )

    sam_start = (
        time.perf_counter()
    )

    sam_candidates = (
        generate_sam_candidates(
            sam,
            image,
        )
    )

    sam_time_ms = (
        time.perf_counter()
        - sam_start
    ) * 1000.0

    print(
        f"SAM generated "
        f"{len(sam_candidates)} "
        f"valid candidates."
    )

    print(
        f"SAM time: "
        f"{sam_time_ms:.2f} ms"
    )

    if not sam_candidates:

        total_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        return {
            "bounding_boxes": [],
            "model": "clip_sam_v1",
            "query": query,
            "num_candidates": 0,
            "processing_time_ms": round(
                total_ms,
                2,
            ),
            "sam_time_ms": round(
                sam_time_ms,
                2,
            ),
            "clip_time_ms": 0.0,
            "image_width": original_width,
            "image_height": original_height,
            "status": "no_detections",
        }

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    candidates = (
        create_merged_candidates(
            sam_candidates,
            inference_width,
            inference_height,
        )
    )

    merged_count = sum(
        1
        for c in candidates
        if c.get("merged", False)
    )

    print(
        f"Candidates after merging: "
        f"{len(candidates)} "
        f"(merged: {merged_count})"
    )

    # --------------------------------------------------------
    # CLIP
    # --------------------------------------------------------

    print(
        "Ranking candidates using CLIP..."
    )

    clip_start = (
        time.perf_counter()
    )

    ranked = rank_candidates(
        image,
        candidates,
        query,
        clip_ranker,
    )

    clip_time_ms = (
        time.perf_counter()
        - clip_start
    ) * 1000.0

    print(
        f"CLIP time: "
        f"{clip_time_ms:.2f} ms"
    )

    if not ranked:

        total_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        return {
            "bounding_boxes": [],
            "model": "clip_sam_v1",
            "query": query,
            "num_candidates": len(
                candidates
            ),
            "processing_time_ms": round(
                total_ms,
                2,
            ),
            "sam_time_ms": round(
                sam_time_ms,
                2,
            ),
            "clip_time_ms": round(
                clip_time_ms,
                2,
            ),
            "image_width": original_width,
            "image_height": original_height,
            "status": "no_detections",
        }

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    ranked = (
        suppress_duplicate_boxes(
            ranked
        )
    )

    # --------------------------------------------------------
    # FINAL BOXES
    # --------------------------------------------------------

    final_boxes = []

    for candidate in ranked[
        :MAX_FINAL_BOXES
    ]:

        x1, y1, x2, y2 = (
            candidate["bbox"]
        )

        original_box = [
            int(
                round(
                    x1 * scale_x
                )
            ),
            int(
                round(
                    y1 * scale_y
                )
            ),
            int(
                round(
                    x2 * scale_x
                )
            ),
            int(
                round(
                    y2 * scale_y
                )
            ),
        ]

        original_box[0] = max(
            0,
            min(
                original_width - 1,
                original_box[0],
            ),
        )

        original_box[1] = max(
            0,
            min(
                original_height - 1,
                original_box[1],
            ),
        )

        original_box[2] = max(
            0,
            min(
                original_width - 1,
                original_box[2],
            ),
        )

        original_box[3] = max(
            0,
            min(
                original_height - 1,
                original_box[3],
            ),
        )

        result = {
            "label": "candidate_region",
            "bbox": original_box,
            "confidence": round(
                float(
                    candidate[
                        "final_score"
                    ]
                ),
                4,
            ),
            "candidate_id": candidate[
                "candidate_id"
            ],
        }

        if candidate.get(
            "merged",
            False,
        ):

            result["merged"] = True

            result["source_ids"] = (
                candidate[
                    "source_ids"
                ]
            )

        final_boxes.append(
            result
        )

    total_ms = (
        time.perf_counter()
        - start_time
    ) * 1000.0

    return {
        "bounding_boxes": final_boxes,
        "model": "clip_sam_v1",
        "query": query,
        "num_candidates": len(
            candidates
        ),
        "processing_time_ms": round(
            total_ms,
            2,
        ),
        "sam_time_ms": round(
            sam_time_ms,
            2,
        ),
        "clip_time_ms": round(
            clip_time_ms,
            2,
        ),
        "image_width": original_width,
        "image_height": original_height,
        "status": "success",
    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    test_image = (
        Path(__file__).resolve().parents[2]
        / "data_processing"
        / "data"
        / "satellite_test.jpeg"
    )

    test_query = (
        "Where is the river?"
    )

    import json

    print(
        "\n" + "=" * 70
    )

    print(
        "SATQUERY AI - SAM + CLIP "
        "GROUNDING TEST"
    )

    print(
        "=" * 70
    )

    result = ground_query(
        test_image,
        test_query,
    )

    print(
        "\nModel result:"
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )
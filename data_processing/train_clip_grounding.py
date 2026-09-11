from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "openai/clip-vit-base-patch32"

DEFAULT_IMAGES = Path(
    "data/vrsbench/images/Images_train"
)

DEFAULT_ANNOTATIONS = Path(
    "data/vrsbench/annotations/Annotations_train"
)

DEFAULT_OUTPUT = Path(
    "models/grounding/checkpoints/clip_grounding_vrsbench"
)

DEFAULT_MAX_SAMPLES = 1000
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 4
DEFAULT_LR = 1e-5
DEFAULT_SEED = 42


# ============================================================
# SEED
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# IMAGE INDEX
# ============================================================

def build_image_index(root: Path) -> dict[str, Path]:

    if not root.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{root}"
        )

    print("Indexing VRSBench images...")

    image_files = []

    for pattern in (
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.PNG",
        "*.JPG",
        "*.JPEG",
    ):
        image_files.extend(
            root.glob(pattern)
        )

    index = {
        path.name: path
        for path in image_files
    }

    print(
        f"Indexed {len(index)} images."
    )

    return index


# ============================================================
# BOUNDING BOX
# ============================================================

def convert_bbox(obj: dict):

    bbox = obj.get("obj_coord")

    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
    ):
        return None

    try:
        values = [
            float(value)
            for value in bbox
        ]
    except (TypeError, ValueError):
        return None

    # The verified VRSBench format uses normalized
    # coordinates in [0, 1].
    if max(values) > 1.0:
        values = [
            value / 100.0
            for value in values
        ]

    x1, y1, x2, y2 = values

    x1 = max(
        0.0,
        min(1.0, x1),
    )
    y1 = max(
        0.0,
        min(1.0, y1),
    )
    x2 = max(
        0.0,
        min(1.0, x2),
    )
    y2 = max(
        0.0,
        min(1.0, y2),
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return [
        x1,
        y1,
        x2,
        y2,
    ]


# ============================================================
# BUILD SAMPLES
# ============================================================

def build_samples(
    annotation_root: Path,
    image_index: dict[str, Path],
    max_samples: int,
    seed: int,
):

    if not annotation_root.exists():
        raise FileNotFoundError(
            f"Annotation directory not found:\n"
            f"{annotation_root}"
        )

    json_files = sorted(
        annotation_root.glob(
            "*.json"
        )
    )

    print(
        f"Annotation files found: "
        f"{len(json_files)}"
    )

    if not json_files:
        raise RuntimeError(
            "No annotation JSON files found."
        )

    samples = []

    missing_images = 0
    invalid_boxes = 0

    # Each JSON file is one complete image record.
    for json_file in json_files:

        try:
            with open(
                json_file,
                "r",
                encoding="utf-8",
            ) as file:
                record = json.load(file)

        except Exception as exc:
            print(
                f"Skipping invalid JSON: "
                f"{json_file.name} "
                f"({exc})"
            )
            continue

        if not isinstance(
            record,
            dict,
        ):
            continue

        # Image filename is the same stem as the JSON.
        #
        # Example:
        # 00002_0000.json
        # 00002_0000.png
        image_name = (
            json_file.stem
            + ".png"
        )

        image_path = image_index.get(
            image_name
        )

        if image_path is None:

            # Try JPG/JPEG just in case.
            for extension in (
                ".jpg",
                ".jpeg",
                ".JPG",
                ".JPEG",
            ):

                candidate_name = (
                    json_file.stem
                    + extension
                )

                if candidate_name in image_index:
                    image_path = (
                        image_index[
                            candidate_name
                        ]
                    )
                    break

        if image_path is None:

            missing_images += 1
            continue

        objects = record.get(
            "objects",
            []
        )

        if not isinstance(
            objects,
            list,
        ):
            continue

        for obj in objects:

            if not isinstance(
                obj,
                dict,
            ):
                continue

            text = obj.get(
                "referring_sentence"
            )

            if not isinstance(
                text,
                str,
            ):
                continue

            text = text.strip()

            if not text:
                continue

            bbox = convert_bbox(
                obj
            )

            if bbox is None:

                invalid_boxes += 1
                continue

            samples.append(
                {
                    "image_path": image_path,
                    "text": text,
                    "bbox": bbox,
                }
            )

    print(
        f"Grounding samples found: "
        f"{len(samples)}"
    )

    print(
        f"Missing images: "
        f"{missing_images}"
    )

    print(
        f"Invalid boxes: "
        f"{invalid_boxes}"
    )

    if not samples:
        raise RuntimeError(
            "No valid grounding samples found."
        )

    random.Random(
        seed
    ).shuffle(samples)

    if (
        max_samples > 0
        and len(samples) > max_samples
    ):
        samples = samples[
            :max_samples
        ]

    print(
        f"Samples selected for training: "
        f"{len(samples)}"
    )

    return samples


# ============================================================
# DATASET
# ============================================================

class VRSBenchGroundingDataset(Dataset):

    def __init__(
        self,
        samples,
    ):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ):

        sample = self.samples[
            index
        ]

        image = Image.open(
            sample["image_path"]
        ).convert("RGB")

        width, height = image.size

        x1, y1, x2, y2 = sample[
            "bbox"
        ]

        # Normalized -> pixels.
        px1 = int(
            round(x1 * width)
        )
        py1 = int(
            round(y1 * height)
        )
        px2 = int(
            round(x2 * width)
        )
        py2 = int(
            round(y2 * height)
        )

        px1 = max(
            0,
            min(
                width - 1,
                px1,
            ),
        )

        py1 = max(
            0,
            min(
                height - 1,
                py1,
            ),
        )

        px2 = max(
            px1 + 1,
            min(
                width,
                px2,
            ),
        )

        py2 = max(
            py1 + 1,
            min(
                height,
                py2,
            ),
        )

        # Add 20% context around target.
        object_width = max(
            1,
            px2 - px1,
        )

        object_height = max(
            1,
            py2 - py1,
        )

        pad_x = max(
            4,
            int(
                object_width * 0.20
            ),
        )

        pad_y = max(
            4,
            int(
                object_height * 0.20
            ),
        )

        crop = image.crop(
            (
                max(
                    0,
                    px1 - pad_x,
                ),
                max(
                    0,
                    py1 - pad_y,
                ),
                min(
                    width,
                    px2 + pad_x,
                ),
                min(
                    height,
                    py2 + pad_y,
                ),
            )
        )

        return {
            "image": crop,
            "text": sample["text"],
        }


# ============================================================
# COLLATE
# ============================================================

def create_collate_fn(
    processor,
):

    def collate(batch):

        images = [
            item["image"]
            for item in batch
        ]

        texts = [
            item["text"]
            for item in batch
        ]

        return processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

    return collate


# ============================================================
# TRAINABLE PARAMETERS
# ============================================================

def configure_trainable_parameters(
    model: CLIPModel,
):

    # Freeze all parameters.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Projection layers.
    for parameter in (
        model.visual_projection.parameters()
    ):
        parameter.requires_grad = True

    for parameter in (
        model.text_projection.parameters()
    ):
        parameter.requires_grad = True

    # Last two vision layers.
    vision_layers = (
        model.vision_model.encoder.layers
    )

    for layer in vision_layers[-2:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    # Last two text layers.
    text_layers = (
        model.text_model.encoder.layers
    )

    for layer in text_layers[-2:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True


def parameter_counts(
    model,
):

    total = 0
    trainable = 0

    for parameter in model.parameters():

        count = parameter.numel()

        total += count

        if parameter.requires_grad:
            trainable += count

    return total, trainable


# ============================================================
# TRAIN
# ============================================================

def train(args):

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Build indexes
    # --------------------------------------------------------

    image_index = build_image_index(
        args.images
    )

    # --------------------------------------------------------
    # Load CLIP
    # --------------------------------------------------------

    print(
        f"\nLoading CLIP:\n"
        f"{MODEL_NAME}"
    )

    processor = (
        CLIPProcessor.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        CLIPModel.from_pretrained(
            MODEL_NAME
        )
    )

    model.to(device)

    configure_trainable_parameters(
        model
    )

    total, trainable = (
        parameter_counts(
            model
        )
    )

    print(
        f"Total parameters: "
        f"{total:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    # --------------------------------------------------------
    # Samples
    # --------------------------------------------------------

    samples = build_samples(
        annotation_root=args.annotations,
        image_index=image_index,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = (
        VRSBenchGroundingDataset(
            samples
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=create_collate_fn(
            processor
        ),
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print(
        f"\nTraining batches: "
        f"{len(loader)}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=0.01,
    )

    use_amp = (
        device.type == "cuda"
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda"
        )
        if use_amp
        else None
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    for epoch in range(
        args.epochs
    ):

        epoch_loss = 0.0
        steps = 0

        print(
            f"\n========== "
            f"Epoch {epoch + 1}/"
            f"{args.epochs} "
            f"=========="
        )

        for step, batch in enumerate(
            loader
        ):

            batch = {
                key: value.to(device)
                for key, value in batch.items()
                if isinstance(
                    value,
                    torch.Tensor,
                )
            }

            if use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):

                    outputs = model(
                        **batch,
                        return_loss=True,
                    )

                    loss = outputs.loss

            else:

                outputs = model(
                    **batch,
                    return_loss=True,
                )

                loss = outputs.loss

            scaled_loss = (
                loss
                / args.grad_accum
            )

            if scaler is not None:

                scaler.scale(
                    scaled_loss
                ).backward()

            else:

                scaled_loss.backward()

            epoch_loss += float(
                loss.item()
            )

            steps += 1

            should_update = (
                (step + 1)
                % args.grad_accum
                == 0
                or
                (step + 1)
                == len(loader)
            )

            if should_update:

                if scaler is not None:
                    scaler.unscale_(
                        optimizer
                    )

                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=1.0,
                )

                if scaler is not None:

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            if (
                step == 0
                or
                (step + 1) % 25 == 0
                or
                (step + 1) == len(loader)
            ):

                avg_loss = (
                    epoch_loss
                    / max(
                        1,
                        steps,
                    )
                )

                print(
                    f"Step "
                    f"{step + 1}/"
                    f"{len(loader)} | "
                    f"Loss: "
                    f"{avg_loss:.4f}"
                )

        avg_epoch_loss = (
            epoch_loss
            / max(
                1,
                steps,
            )
        )

        print(
            f"\nEpoch "
            f"{epoch + 1} complete | "
            f"Average loss: "
            f"{avg_epoch_loss:.4f}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        args.output
    )

    processor.save_pretrained(
        args.output
    )

    metadata = {
        "base_model": MODEL_NAME,
        "samples": len(samples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "learning_rate": args.lr,
        "trainable_parameters": trainable,
        "total_parameters": total,
    }

    with open(
        args.output
        / "training_config.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        "\n============================================================"
    )

    print(
        "CLIP GROUNDING TRAINING COMPLETE"
    )

    print(
        f"Model saved to:\n"
        f"{args.output.resolve()}"
    )

    print(
        "============================================================"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGES,
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=DEFAULT_GRAD_ACCUM,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
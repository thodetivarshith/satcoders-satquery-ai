import torch
from segment_anything import sam_model_registry


CHECKPOINT_PATH = "models/grounding/checkpoints/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"


def load_sam():
    """Load the SAM model using GPU when available."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading SAM on: {device}")

    sam = sam_model_registry[MODEL_TYPE](
        checkpoint=CHECKPOINT_PATH
    )

    sam.to(device=device)

    print("SAM loaded successfully!")
    print(f"Device: {device}")

    return sam


if __name__ == "__main__":
    model = load_sam()
# Training configuration - Varshith
"""
config.py

Central config for GeoChat fine-tuning on BigEarthNet.txt.
Keep every tunable in one place so train_geochat.py and
inference_geochat.py both import from here instead of hardcoding values.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    # Point these at your actual BigEarthNet.txt download location
    root_dir: str = os.environ.get("BIGEARTH_ROOT", "./data/bigearthnet")
    annotations_file: str = os.environ.get(
        "BIGEARTH_ANNOTATIONS", "./data/bigearthnet/annotations.jsonl"
    )
    train_split_file: str = "./data/bigearthnet/splits/train.txt"
    val_split_file: str = "./data/bigearthnet/splits/val.txt"
    test_split_file: str = "./data/bigearthnet/splits/test.txt"

    image_size: int = 224
    use_sar: bool = True

    # Team data contract — see bigearth_dataset.py header. Don't change
    # this without telling Kunchala/Abhinay/Akshaya, it's the shared format.
    bands: List[str] = field(
        default_factory=lambda: ["red", "green", "blue", "nir", "vv", "vh"]
    )


@dataclass
class ModelConfig:
    # HuggingFace repo for pretrained GeoChat (per execution plan Day 1 setup)
    pretrained_model_name: str = "linjie/geochat"
    checkpoint_dir: str = "./checkpoints"

    # Vision encoder is 3-channel (CLIP-based) — see bigearth_dataset.py
    # header for why we only feed RGB in this version.
    vision_input_channels: int = 3

    # LoRA fine-tuning (recommended given time/compute constraints —
    # full fine-tuning is slower and needs more GPU memory than a
    # 20-day hackathon timeline comfortably allows)
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    max_text_length: int = 512


@dataclass
class TrainConfig:
    output_dir: str = "./checkpoints/geochat_v1_bigearth"

    batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4  # effective batch size = 32

    learning_rate: float = 2e-4  # higher LR is typical/expected for LoRA
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03

    num_epochs: int = 3  # plan targets 3-5 epochs for Week 3 "final" run
    num_epochs_week2_checkpoint: int = 1  # Week 2 target is just 1 epoch

    eval_every_n_steps: int = 200
    save_every_n_steps: int = 200
    logging_every_n_steps: int = 20

    mixed_precision: str = "bf16"  # falls back to fp16 if bf16 unsupported
    seed: int = 42

    # Week-by-week accuracy targets from the execution plan, kept here
    # so train_geochat.py can log progress against them
    target_accuracy_week2: float = 0.65
    target_accuracy_week3: float = 0.75
    target_inference_time_ms: int = 2000


@dataclass
class InferenceConfig:
    checkpoint_path: str = "./checkpoints/geochat_v1_bigearth/best.pt"
    device: str = "cuda"  # falls back to "cpu" automatically if unavailable
    max_new_tokens: int = 128
    temperature: float = 0.2  # low temperature — factual VQA, not creative generation


data_config = DataConfig()
model_config = ModelConfig()
train_config = TrainConfig()
inference_config = InferenceConfig()


if __name__ == "__main__":
    # Quick sanity print — run this after editing paths above
    print("=== Data ===")
    print(data_config)
    print("\n=== Model ===")
    print(model_config)
    print("\n=== Train ===")
    print(train_config)
    print("\n=== Inference ===")
    print(inference_config)

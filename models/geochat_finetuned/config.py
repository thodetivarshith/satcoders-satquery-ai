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
    # CORRECTED: the execution plan's "linjie/geochat" doesn't exist.
    # Real checkpoint: https://huggingface.co/MBZUAI/geochat-7B
    # (LLaVA-1.5 architecture, 7B params, CLIP ViT-L/14 vision tower
    # extended to 504x504). We fine-tune further FROM this checkpoint
    # rather than rebuilding from base LLaVA + projector — much simpler
    # and skips a second multi-GB download.
    pretrained_model_name: str = "MBZUAI/geochat-7B"
    geochat_repo_path: str = "./GeoChat"  # where you `git clone` their repo
    checkpoint_dir: str = "./checkpoints"

    # Official training uses their own train_mem.py (LLaVA-forked), NOT
    # a from_pretrained()/custom forward loop — see train_geochat.py notes.
    # This vision_input_channels / use_lora / lora_* config below still
    # applies since their script also uses PEFT LoRA under the hood.
    vision_input_channels: int = 3

    # QLoRA — REQUIRED on a single free-tier T4 (15GB). Official docs
    # assume 3x A100 40GB; loading the 7B base in 4-bit (via bitsandbytes)
    # is what makes single-GPU LoRA fine-tuning survive at all.
    # Confirmed supported: LLaVA/GeoChat's train_mem.py accepts --bits 4
    # directly (same flag used in LLaVA's own finetune_qlora.sh).
    use_lora: bool = True
    load_in_4bit: bool = True
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

    # Realistic subset size for free-tier T4 QLoRA — NOT the full 318k
    # GeoChat_Instruct set (~100GB, needs multi-GPU days to train on).
    # A few hundred to ~1-2k BigEarthNet.txt examples is enough to show
    # real domain adaptation within the Sep 19 deadline.
    target_dataset_size: int = 500

    batch_size: int = 1  # per_device_train_batch_size — T4 forces this low
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16  # effective batch size = 16

    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03

    num_epochs: int = 1  # start with 1 epoch on the small subset, extend if time allows

    eval_every_n_steps: int = 200
    save_every_n_steps: int = 100  # more frequent given Colab disconnect risk
    logging_every_n_steps: int = 10

    mixed_precision: str = "fp16"  # falls back to fp16 if fp16 unsupported
    seed: int = 42

    # Week-by-week accuracy targets from the execution plan, kept here
    # so we can log progress against them (adjusted down given the
    # much smaller training set vs. the plan's original assumption)
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
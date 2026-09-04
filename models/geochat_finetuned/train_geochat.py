# GeoChat training script - Varshith
"""
train_geochat.py

Fine-tune GeoChat-7B with QLoRA on the prepared BigEarthNet GeoChat dataset.

Expected Colab environment:
    /content/satcoders-satquery-ai/
        models/
            geochat_finetuned/
                images/
                    *.jpg
                conversations.json
                config.py
                train_geochat.py
                GeoChat/

Training:
    - GeoChat-7B
    - 4-bit QLoRA
    - LoRA
    - Single NVIDIA GPU
    - FP16
"""

import importlib
import os
import subprocess
import sys

from config import model_config, train_config


# ============================================================
# COLAB / DRIVE PATH HELPERS
# ============================================================

def _mount_drive_if_colab(path: str) -> str:
    """
    Mount Google Drive when running in Colab.

    For Colab, checkpoints are redirected to Drive so they
    survive runtime disconnects.
    """

    try:
        importlib.import_module("google.colab")
        drive = importlib.import_module("google.colab.drive")

        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
            print("Google Drive mounted.")

        if path.startswith("/content/drive"):
            return path

        redirected = os.path.join(
            "/content/drive/MyDrive/satquery/checkpoints",
            os.path.basename(path.rstrip("/")),
        )

        print(
            f"Redirecting checkpoint path:\n"
            f"  {path}\n"
            f"→ {redirected}"
        )

        return redirected

    except ImportError:
        return path


# ============================================================
# DATA VALIDATION
# ============================================================

def _check_prereqs(
    images_dir: str,
    conversations_json: str,
) -> None:

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Image folder not found:\n{images_dir}"
        )

    if not os.path.isfile(conversations_json):
        raise FileNotFoundError(
            f"Conversations JSON not found:\n{conversations_json}"
        )

    image_files = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    if not image_files:
        raise RuntimeError(
            f"No images found in:\n{images_dir}"
        )

    print(
        f"Found {len(image_files)} images in:\n"
        f"{images_dir}"
    )

    print(
        f"Found conversations file:\n"
        f"{conversations_json}"
    )


# ============================================================
# BUILD GE0CHAT TRAINING COMMAND
# ============================================================

def build_train_command(
    geochat_repo_path: str,
    images_dir: str,
    conversations_json: str,
    output_dir: str,
) -> list:

    train_script = os.path.join(
        geochat_repo_path,
        "geochat",
        "train",
        "train_mem.py",
    )

    if not os.path.isfile(train_script):
        raise FileNotFoundError(
            f"GeoChat training script not found:\n"
            f"{train_script}\n\n"
            f"Expected GeoChat repository at:\n"
            f"{geochat_repo_path}"
        )

    cmd = [
        sys.executable,
        train_script,

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        "--model_name_or_path",
        model_config.pretrained_model_name,

        "--version",
        "v1",

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        "--data_path",
        conversations_json,

        "--image_folder",
        images_dir,

        # ----------------------------------------------------
        # VISION TOWER
        # ----------------------------------------------------

        "--vision_tower",
        "openai/clip-vit-large-patch14-336",

        "--mm_projector_type",
        "mlp2x_gelu",

        "--mm_vision_select_layer",
        "-2",

        "--mm_use_im_start_end",
        "False",

        "--mm_use_im_patch_token",
        "False",

        "--image_aspect_ratio",
        "pad",

        # ----------------------------------------------------
        # QLORA / LORA
        # ----------------------------------------------------

        "--lora_enable",
        "True",

        "--bits",
        "4",

        # ----------------------------------------------------
        # PRECISION
        # ----------------------------------------------------

        "--fp16",
        "True",

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        "--output_dir",
        output_dir,

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        "--num_train_epochs",
        str(train_config.num_epochs),

        "--per_device_train_batch_size",
        str(train_config.batch_size),

        "--per_device_eval_batch_size",
        str(train_config.eval_batch_size),

        "--gradient_accumulation_steps",
        str(
            train_config.gradient_accumulation_steps
        ),

        # ----------------------------------------------------
        # OPTIMIZER / LR
        # ----------------------------------------------------

        "--learning_rate",
        str(train_config.learning_rate),

        "--weight_decay",
        str(train_config.weight_decay),

        "--warmup_ratio",
        str(train_config.warmup_ratio),

        "--lr_scheduler_type",
        "cosine",

        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        "--evaluation_strategy",
        "no",

        # ----------------------------------------------------
        # CHECKPOINTS
        # ----------------------------------------------------

        "--save_strategy",
        "steps",

        "--save_steps",
        str(train_config.save_every_n_steps),

        "--save_total_limit",
        "2",

        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------

        "--logging_steps",
        str(train_config.logging_every_n_steps),

        "--report_to",
        "none",

        # ----------------------------------------------------
        # MEMORY
        # ----------------------------------------------------

        "--gradient_checkpointing",
        "True",

        "--lazy_preprocess",
        "True",

        "--model_max_length",
        str(model_config.max_text_length),

        # ----------------------------------------------------
        # DATALOADER
        # ----------------------------------------------------

        "--dataloader_num_workers",
        "2",
    ]

    return cmd


# ============================================================
# TRAIN
# ============================================================

def train() -> None:

    # --------------------------------------------------------
    # CHECK ENVIRONMENT
    # --------------------------------------------------------

    print("=" * 70)
    print("GE0CHAT BIGEARTHNET QLORA TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECKPOINT DIRECTORY
    # --------------------------------------------------------

    output_dir = _mount_drive_if_colab(
        train_config.output_dir
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DATA PATHS
    # --------------------------------------------------------

    images_dir = os.environ.get(
        "GEOCHAT_IMAGES_DIR",
        "/content/satcoders-satquery-ai/models/geochat_finetuned/images",
    )

    conversations_json = os.environ.get(
        "GEOCHAT_CONVERSATIONS_JSON",
        "/content/satcoders-satquery-ai/models/geochat_finetuned/conversations.json",
    )

    # --------------------------------------------------------
    # VALIDATE DATA
    # --------------------------------------------------------

    _check_prereqs(
        images_dir,
        conversations_json,
    )

    # --------------------------------------------------------
    # GEOCHAT REPOSITORY
    # --------------------------------------------------------

    geochat_repo_path = model_config.geochat_repo_path

    print()
    print("GeoChat repository:")
    print(geochat_repo_path)

    print()
    print("Model:")
    print(model_config.pretrained_model_name)

    print()
    print("Images:")
    print(images_dir)

    print()
    print("Conversations:")
    print(conversations_json)

    print()
    print("Checkpoint output:")
    print(output_dir)

    # --------------------------------------------------------
    # BUILD COMMAND
    # --------------------------------------------------------

    cmd = build_train_command(
        geochat_repo_path=geochat_repo_path,
        images_dir=images_dir,
        conversations_json=conversations_json,
        output_dir=output_dir,
    )

    print()
    print("=" * 70)
    print("TRAINING COMMAND")
    print("=" * 70)

    print(
        " ".join(cmd)
    )

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # RUN TRAINING
    # --------------------------------------------------------

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(
            line,
            end=""
        )

    process.wait()

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if process.returncode != 0:

        print()
        print("=" * 70)
        print("TRAINING FAILED")
        print("=" * 70)

        print(
            f"train_mem.py exited with code "
            f"{process.returncode}."
        )

        print()
        print(
            "Check the error printed above."
        )

        sys.exit(
            process.returncode
        )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"LoRA checkpoint saved to:\n"
        f"{output_dir}"
    )

    print()
    print(
        "Next step: merge/use the LoRA adapter "
        "with the GeoChat base checkpoint for inference."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train()
# GeoChat training script - Varshith
"""
train_geochat.py

Fine-tunes GeoChat (LoRA) on BigEarthNet.txt using bigearth_dataset.py
and config.py. Written for Google Colab free tier (single T4 GPU,
~15GB VRAM, sessions can disconnect at any time) — so it:
  - Mounts Google Drive and saves ALL checkpoints there, not to Colab's
    local disk (local disk is wiped when the session ends/disconnects)
  - Checkpoints frequently (every `save_every_n_steps`, from config.py)
    so a dropped session loses minutes, not hours
  - Auto-resumes from the latest checkpoint on Drive if one exists
  - Uses LoRA + bf16/fp16 + gradient accumulation to fit T4's memory

===========================================================================
COLAB SETUP (run these in a Colab cell before this script):
===========================================================================
    # 1. Runtime > Change runtime type > T4 GPU

    # 2. Mount Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # 3. Install deps
    !pip install torch transformers peft accelerate rasterio bitsandbytes -q

    # 4. Put BigEarthNet.txt data + this repo under Drive, e.g.:
    #    /content/drive/MyDrive/satquery/data/bigearthnet/
    #    /content/drive/MyDrive/satquery/checkpoints/
    #    then point config.py's paths there (or override with env vars
    #    BIGEARTH_ROOT / BIGEARTH_ANNOTATIONS before importing config)

    # 5. Run:
    !python train_geochat.py
===========================================================================
"""

import os
import time
import glob
import importlib

torch = importlib.import_module("torch")
DataLoader = importlib.import_module("torch.utils.data").DataLoader

from config import data_config, model_config, train_config
from bigearth_dataset import BigEarthDataset, collate_fn


def _mount_drive_if_colab():
    """No-op outside Colab. Inside Colab, mounts Drive so checkpoints
    survive a disconnect. Call this before touching any Drive path."""
    try:
        # Use dynamic imports so local environments and static analyzers do
        # not require the Colab-only package to be installed.
        importlib.import_module("google.colab")
        drive = importlib.import_module("google.colab.drive")

        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
            print("Google Drive mounted.")
    except ImportError:
        pass  # not running in Colab — assume local paths are fine


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: No GPU detected — falling back to CPU. This will be "
          "very slow for GeoChat fine-tuning. Check Runtime > Change "
          "runtime type > GPU in Colab.")
    return torch.device("cpu")


def _latest_checkpoint(output_dir: str):
    """Finds the most recent checkpoint under output_dir, or None."""
    if not os.path.isdir(output_dir):
        return None
    ckpts = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*.pt")),
        key=os.path.getmtime,
    )
    return ckpts[-1] if ckpts else None


def build_model():
    """
    Loads pretrained GeoChat and wraps it with a LoRA adapter.

    NOTE: GeoChat isn't a standard HF AutoModel — you'll likely load it
    via the repo you cloned in Day-1 setup (huggingface.co/linjie/geochat)
    rather than `transformers.AutoModel.from_pretrained`. Swap the
    placeholder loading line below for GeoChat's actual loading API once
    you've cloned it — the LoRA wrapping and optimizer setup below don't
    need to change.
    """
    # PEFT is optional until LoRA is enabled. Import it dynamically so this
    # module remains importable in environments where PEFT is not installed.
    peft = importlib.import_module("peft")
    LoraConfig = peft.LoraConfig
    get_peft_model = peft.get_peft_model

    # --- Placeholder: replace with GeoChat's real model-loading call ---
    # e.g. from geochat.model import GeoChatForConditionalGeneration
    #      base_model = GeoChatForConditionalGeneration.from_pretrained(
    #          model_config.pretrained_model_name
    #      )
    raise NotImplementedError(
        "Plug in GeoChat's actual model-loading call here once you've "
        "cloned github.com/linjie/geochat (or its HF repo) — see the "
        "docstring above build_model(). Everything else in this script "
        "(LoRA wrapping, training loop, checkpointing) is ready to go."
    )
    # --- end placeholder ---

    if model_config.use_lora:
        lora_config = LoraConfig(
            r=model_config.lora_rank,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            target_modules=model_config.lora_target_modules,
            bias="none",
        )
        base_model = get_peft_model(base_model, lora_config)
        base_model.print_trainable_parameters()

    return base_model


def train():
    _mount_drive_if_colab()
    device = _pick_device()
    torch.manual_seed(train_config.seed)

    os.makedirs(train_config.output_dir, exist_ok=True)

    # -- Data --
    train_dataset = BigEarthDataset(
        root_dir=data_config.root_dir,
        annotations_file=data_config.annotations_file,
        split="train",
        split_ids_file=data_config.train_split_file,
        image_size=data_config.image_size,
        use_sar=data_config.use_sar,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,  # Colab free tier has limited CPU — keep this low
        pin_memory=(device.type == "cuda"),
    )
    print(f"Train samples: {len(train_dataset)}")

    # -- Model --
    model = build_model().to(device)

    dtype = torch.bfloat16 if train_config.mixed_precision == "bf16" else torch.float16
    use_amp = device.type == "cuda"

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    # -- Resume from latest checkpoint, if any (Colab session may have dropped) --
    start_step = 0
    latest_ckpt = _latest_checkpoint(train_config.output_dir)
    if latest_ckpt is not None:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
    else:
        print("No checkpoint found — starting fresh.")

    model.train()
    global_step = start_step
    accum_steps = train_config.gradient_accumulation_steps
    t0 = time.time()

    for epoch in range(train_config.num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                # --- Placeholder forward/loss call ---
                # Replace with GeoChat's actual forward signature, e.g.:
                # outputs = model(pixel_values=pixel_values,
                #                  prompts=batch["prompt"],
                #                  labels=batch["answer"])
                # loss = outputs.loss
                raise NotImplementedError(
                    "Plug in GeoChat's forward()/loss call here — depends "
                    "on its exact API once cloned. pixel_values, "
                    "batch['prompt'], batch['answer'] are already prepared "
                    "for you by bigearth_dataset.py."
                )
                # --- end placeholder ---

            loss = loss / accum_steps
            loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % train_config.logging_every_n_steps == 0:
                    elapsed = time.time() - t0
                    print(f"epoch={epoch} step={global_step} "
                          f"loss={loss.item() * accum_steps:.4f} "
                          f"elapsed={elapsed:.0f}s")

                if global_step % train_config.save_every_n_steps == 0:
                    ckpt_path = os.path.join(
                        train_config.output_dir, f"checkpoint-{global_step}.pt"
                    )
                    torch.save({
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, ckpt_path)
                    print(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(train_config.output_dir, "best.pt")
    torch.save({"step": global_step, "model_state_dict": model.state_dict()}, final_path)
    print(f"Training complete. Final model: {final_path}")


if __name__ == "__main__":
    train()

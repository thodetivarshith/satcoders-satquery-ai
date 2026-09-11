from __future__ import annotations

from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor


BASE_MODEL = "openai/clip-vit-base-patch32"

FINETUNED_MODEL = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "clip_grounding_vrsbench"
)


class CLIPRanker:
    """
    CLIP ranking module for SAM candidate regions.

    Uses the fine-tuned VRSBench checkpoint when available.
    """

    def __init__(
        self,
        model_name: str = BASE_MODEL,
    ):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if FINETUNED_MODEL.exists():
            model_path = str(FINETUNED_MODEL)
            print(
                "Loading fine-tuned VRSBench CLIP..."
            )
        else:
            model_path = model_name
            print(
                "Fine-tuned VRSBench CLIP not found."
            )
            print(
                "Loading base CLIP..."
            )

        self.processor = (
            CLIPProcessor.from_pretrained(
                model_path
            )
        )

        self.model = (
            CLIPModel.from_pretrained(
                model_path
            )
        )

        self.model.to(self.device)
        self.model.eval()

        print("CLIP loaded successfully!")
        print(f"CLIP device: {self.device}")

    @torch.inference_mode()
    def _get_text_features(
        self,
        query: str,
    ) -> torch.Tensor:
        """
        Extract normalized CLIP text embeddings.

        Transformers versions may return a
        BaseModelOutputWithPooling object from
        model.text_model(), so we explicitly use
        .pooler_output.
        """

        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = inputs["input_ids"].to(
            self.device
        )

        attention_mask = inputs.get(
            "attention_mask"
        )

        if attention_mask is not None:
            attention_mask = attention_mask.to(
                self.device
            )

        outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = outputs.pooler_output

        text_features = (
            self.model.text_projection(
                pooled_output
            )
        )

        text_features = (
            text_features
            / text_features.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        )

        return text_features

    @torch.inference_mode()
    def _get_image_features(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract normalized CLIP image embeddings.
        """

        outputs = self.model.vision_model(
            pixel_values=pixel_values
        )

        pooled_output = outputs.pooler_output

        image_features = (
            self.model.visual_projection(
                pooled_output
            )
        )

        image_features = (
            image_features
            / image_features.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        )

        return image_features

    @torch.inference_mode()
    def rank_regions(
        self,
        images,
        query: str,
        batch_size: int = 16,
    ):
        """
        Rank candidate image regions against a query.

        Returns one cosine-similarity score per image.
        """

        if not images:
            return []

        text_features = (
            self._get_text_features(
                query
            )
        )

        scores = []

        for start in range(
            0,
            len(images),
            batch_size,
        ):
            batch = images[
                start:start + batch_size
            ]

            encoded = self.processor(
                images=batch,
                return_tensors="pt",
            )

            pixel_values = encoded[
                "pixel_values"
            ].to(self.device)

            image_features = (
                self._get_image_features(
                    pixel_values
                )
            )

            similarity = (
                image_features
                @ text_features.T
            )

            scores.extend(
                similarity[:, 0]
                .detach()
                .float()
                .cpu()
                .tolist()
            )

        return scores

    def rank_region(
        self,
        image,
        query: str,
    ):
        """Rank a single image region."""

        scores = self.rank_regions(
            [image],
            query,
            batch_size=1,
        )

        if not scores:
            raise RuntimeError(
                "CLIP returned no score."
            )

        return scores[0]

    def rank_regions_batch(
        self,
        images,
        query: str,
        batch_size: int = 16,
    ):
        """
        Backward-compatible API.
        """

        return self.rank_regions(
            images,
            query,
            batch_size=batch_size,
        )
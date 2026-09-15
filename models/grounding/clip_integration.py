from pathlib import Path
from typing import List, Union, Optional, Any

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPRanker:
    """
    CLIP ranker used by the existing ground_query.py pipeline.

    Loads:
        models/grounding/checkpoints/clip_grounding_ranker
    """

    BASE_MODEL = "openai/clip-vit-base-patch32"

    CHECKPOINT_DIR = (
        Path(__file__).resolve().parent / "checkpoints"
    )

    FINETUNED_MODEL = (
        CHECKPOINT_DIR / "clip_grounding_ranker"
    )

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        device: Optional[str] = None
    ):
        if getattr(self, "_initialized", False):
            return

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Loading CLIP on: {self.device}")

        self.processor = (
            CLIPProcessor.from_pretrained(
                self.BASE_MODEL
            )
        )

        if self.FINETUNED_MODEL.exists():

            print(
                "Loading hard-negative VRSBench CLIP ranker..."
            )

            print(
                f"Checkpoint: {self.FINETUNED_MODEL}"
            )

            self.model = (
                CLIPModel.from_pretrained(
                    str(self.FINETUNED_MODEL)
                )
            )

        else:

            print(
                "WARNING: ranker checkpoint not found."
            )

            print(
                f"Expected: {self.FINETUNED_MODEL}"
            )

            print(
                "Falling back to base CLIP."
            )

            self.model = (
                CLIPModel.from_pretrained(
                    self.BASE_MODEL
                )
            )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        self._initialized = True

        print(
            "CLIP loaded successfully!"
        )

        print(
            f"CLIP device: {self.device}"
        )

    # =========================================================
    # TEXT FEATURES
    # =========================================================

    @torch.no_grad()
    def get_text_features(
        self,
        texts: Union[str, List[str]]
    ) -> torch.Tensor:

        if isinstance(texts, str):
            texts = [texts]

        cleaned_texts = []

        for text in texts:

            if isinstance(text, str):
                cleaned_texts.append(text)

            else:
                cleaned_texts.append(
                    str(text)
                )

        inputs = self.processor(
            text=cleaned_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        input_ids = (
            inputs["input_ids"]
            .to(self.device)
        )

        attention_mask = (
            inputs["attention_mask"]
            .to(self.device)
        )

        outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = (
            outputs.pooler_output
        )

        text_features = (
            self.model.text_projection(
                pooled_output
            )
        )

        text_features = (
            text_features
            / (
                text_features.norm(
                    dim=-1,
                    keepdim=True
                )
                + 1e-8
            )
        )

        return text_features

    # =========================================================
    # IMAGE FEATURES
    # =========================================================

    @torch.no_grad()
    def get_image_features(
        self,
        images: Union[
            Image.Image,
            List[Image.Image]
        ]
    ) -> torch.Tensor:

        if isinstance(
            images,
            Image.Image
        ):
            images = [images]

        processed_images = []

        for image in images:

            # Direct PIL image
            if isinstance(
                image,
                Image.Image
            ):

                processed_images.append(
                    image.convert("RGB")
                )

                continue

            # Dictionary candidate
            if isinstance(
                image,
                dict
            ):

                found = None

                for key in (
                    "image",
                    "crop",
                    "region",
                    "pil_image"
                ):

                    value = image.get(key)

                    if isinstance(
                        value,
                        Image.Image
                    ):

                        found = value
                        break

                if found is not None:

                    processed_images.append(
                        found.convert("RGB")
                    )

                    continue

            # Tuple/list candidate
            if isinstance(
                image,
                (tuple, list)
            ):

                found = None

                for value in image:

                    if isinstance(
                        value,
                        Image.Image
                    ):

                        found = value
                        break

                if found is not None:

                    processed_images.append(
                        found.convert("RGB")
                    )

                    continue

            raise TypeError(
                "Could not extract PIL image "
                f"from candidate type "
                f"{type(image).__name__}"
            )

        inputs = self.processor(
            images=processed_images,
            return_tensors="pt"
        )

        pixel_values = (
            inputs["pixel_values"]
            .to(self.device)
        )

        outputs = self.model.vision_model(
            pixel_values=pixel_values
        )

        pooled_output = (
            outputs.pooler_output
        )

        image_features = (
            self.model.visual_projection(
                pooled_output
            )
        )

        image_features = (
            image_features
            / (
                image_features.norm(
                    dim=-1,
                    keepdim=True
                )
                + 1e-8
            )
        )

        return image_features

    # =========================================================
    # REGION RANKING
    # =========================================================

    @torch.no_grad()
    def rank_regions(
        self,
        *args,
        **kwargs
    ):
        """
        Rank candidate regions against a query.

        IMPORTANT:
        The existing ground_query.py expects this method
        to return a plain list of FLOAT scores.

        Supported calls:

            rank_regions(query, regions)

            rank_regions(regions, query)

            rank_regions(
                query_text=query,
                region_images=regions
            )
        """

        query_text = kwargs.get(
            "query_text"
        )

        if query_text is None:
            query_text = kwargs.get(
                "query"
            )

        region_images = kwargs.get(
            "region_images"
        )

        if region_images is None:
            region_images = kwargs.get(
                "regions"
            )

        if region_images is None:
            region_images = kwargs.get(
                "candidate_images"
            )

        batch_size = int(
            kwargs.get(
                "batch_size",
                32
            )
        )

        # -----------------------------------------------------
        # Parse positional arguments
        # -----------------------------------------------------

        if len(args) >= 2:

            first = args[0]
            second = args[1]

            # rank_regions(query, regions)
            if isinstance(
                first,
                str
            ):

                query_text = first
                region_images = second

            # rank_regions(regions, query)
            elif isinstance(
                second,
                str
            ):

                region_images = first
                query_text = second

        elif len(args) == 1:

            value = args[0]

            if isinstance(
                value,
                str
            ):

                query_text = value

            else:

                region_images = value

        # -----------------------------------------------------
        # Validate query
        # -----------------------------------------------------

        if isinstance(
            query_text,
            (list, tuple)
        ):

            strings = [
                x for x in query_text
                if isinstance(x, str)
            ]

            if len(strings) == 1:
                query_text = strings[0]

        if not isinstance(
            query_text,
            str
        ):

            raise TypeError(
                "rank_regions could not obtain "
                "a valid query string."
            )

        query_text = query_text.strip()

        if not query_text:

            raise ValueError(
                "rank_regions received an empty query."
            )

        # -----------------------------------------------------
        # Validate candidate regions
        # -----------------------------------------------------

        if region_images is None:

            raise ValueError(
                "rank_regions did not receive "
                "candidate regions."
            )

        if not isinstance(
            region_images,
            (list, tuple)
        ):

            raise TypeError(
                "region_images must be a "
                "list or tuple."
            )

        if len(region_images) == 0:
            return []

        # -----------------------------------------------------
        # Text embedding
        # -----------------------------------------------------

        text_features = (
            self.get_text_features(
                query_text
            )
        )

        # -----------------------------------------------------
        # Image embeddings
        # -----------------------------------------------------

        all_scores = []

        for start in range(
            0,
            len(region_images),
            batch_size
        ):

            batch = region_images[
                start:start + batch_size
            ]

            image_features = (
                self.get_image_features(
                    batch
                )
            )

            similarity = (
                image_features
                @ text_features.T
            )

            similarity = (
                similarity
                .squeeze(-1)
                .detach()
                .float()
                .cpu()
                .tolist()
            )

            # Make sure we ALWAYS append floats.
            for score in similarity:

                all_scores.append(
                    float(score)
                )

        # -----------------------------------------------------
        # IMPORTANT
        # -----------------------------------------------------
        # Return ONLY floats because ground_query.py does:
        #
        # np.asarray(raw_scores, dtype=float)
        #
        return all_scores

    # =========================================================
    # BACKWARD COMPATIBILITY
    # =========================================================

    @torch.no_grad()
    def rank_candidates(
        self,
        query: str,
        candidate_images: List[Any]
    ):

        return self.rank_regions(
            query,
            candidate_images
        )

    # =========================================================
    # SIMILARITY
    # =========================================================

    @torch.no_grad()
    def similarity(
        self,
        images,
        texts
    ):

        image_features = (
            self.get_image_features(
                images
            )
        )

        text_features = (
            self.get_text_features(
                texts
            )
        )

        return (
            image_features
            @ text_features.T
        )


# Compatibility aliases
CLIPGrounding = CLIPRanker
CLIPIntegration = CLIPRanker


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CLIP Ranker Integration Test")
    print("=" * 60)

    ranker = CLIPRanker()

    print()
    print("Checkpoint:")
    print(
        ranker.FINETUNED_MODEL
    )

    print()
    print("Checkpoint exists:")
    print(
        ranker.FINETUNED_MODEL.exists()
    )

    text_features = (
        ranker.get_text_features(
            "industrial or commercial area"
        )
    )

    print()
    print("Text feature shape:")
    print(
        tuple(
            text_features.shape
        )
    )

    print()
    print("Text feature norm:")
    print(
        float(
            text_features.norm().item()
        )
    )

    print()
    print("rank_regions available:")
    print(
        hasattr(
            ranker,
            "rank_regions"
        )
    )

    print("=" * 60)
    print("CLIP ranker test completed.")
    print("=" * 60)
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import tempfile
import time
from pathlib import Path

from models.router.router import route_query

router = APIRouter()


@router.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    query: str = Form(...)
):
    start_time = time.perf_counter()
    temp_image_path = None

    try:
        # Validate image
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )

        # Validate query
        query = query.strip()
        if not query:
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )

        # -----------------------------
        # 1. QUERY ROUTING
        # -----------------------------
        routing_start = time.perf_counter()

        routing_result = route_query(query)

        routing_time = time.perf_counter() - routing_start

        if routing_result["status"] != "success":
            raise HTTPException(
                status_code=400,
                detail=routing_result.get(
                    "message",
                    "Query routing failed"
                )
            )

        task = routing_result["task"]
        module = routing_result["module"]

        # -----------------------------
        # 2. SAVE TEMPORARY IMAGE
        # -----------------------------
        image_save_start = time.perf_counter()

        suffix = Path(image.filename or ".jpg").suffix or ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            image_bytes = await image.read()
            temp_file.write(image_bytes)
            temp_image_path = Path(temp_file.name)

        preprocessing_time = time.perf_counter() - image_save_start

        # -----------------------------
        # 3. RUN SELECTED AI PIPELINE
        # -----------------------------
        inference_start = time.perf_counter()

        result = {}

        # ---- Grounding: SAM + CLIP ----
        if task == "grounding":

            from models.grounding.ground_query import ground_query

            result = ground_query(
                temp_image_path,
                query
            )
            boxes = result.get(
                "bounding_boxes",
                []
            )

            if boxes:
                top_confidence = max(
                    float(box.get("confidence", 0.0))
                    for box in boxes
                )

                answer = (
                    f"{len(boxes)} candidate regions were identified for this "
                    f"grounding query. "
                    f"The highest-confidence region scored "
                    f"{top_confidence * 100:.2f}%. "
                    f"Highlighted boxes provide the visual evidence."
                )
            else:
                answer = "No matching regions were identified in the image."

            confidence_values = [
                float(box.get("confidence", 0.0))
                for box in boxes
            ]

            confidence = (
                max(confidence_values)
                if confidence_values
                else 0.0
            )

            model_name = result.get(
                "model",
                "clip_sam_v1"
            )

        # ---- VQA: GeoChat ----
        elif task == "vqa":

            from models.geochat_finetuned.inference_geochat import query_image

            result = query_image(
                temp_image_path,
                query
            )

            answer = result.get(
                "answer",
                ""
            )

            boxes = result.get(
                "bounding_boxes",
                []
            )

            confidence = float(
                result.get(
                    "confidence",
                    0.0
                )
            )

            model_name = result.get(
                "model",
                "geochat"
            )

        # ---- Other pipelines ----
        else:

            raise HTTPException(
                status_code=501,
                detail=(
                    f"Task '{task}' is routed to "
                    f"'{module}', but its inference pipeline "
                    f"is not yet connected to /analyze."
                )
            )

        inference_time = time.perf_counter() - inference_start
        total_time = time.perf_counter() - start_time

        # -----------------------------
        # 4. CLEAN TEMPORARY FILE
        # -----------------------------
        if temp_image_path is not None:
            try:
                temp_image_path.unlink(missing_ok=True)
            except Exception:
                pass

        # -----------------------------
        # 5. API RESPONSE
        # -----------------------------
        return {
            "query": query,
            "answer": answer,
            "confidence": round(confidence, 4),
            "task_type": task,
            "module": module,
            "bounding_boxes": boxes,
            "execution_trace": {
                "preprocessing_time": round(
                    preprocessing_time,
                    4
                ),
                "routing_time": round(
                    routing_time,
                    4
                ),
                "inference_time": round(
                    inference_time,
                    4
                ),
                "total_time": round(
                    total_time,
                    4
                )
            },
            "metadata": {
                "model": model_name,
                "image_name": image.filename,
                "routing_confidence": routing_result[
                    "confidence"
                ],
                "status": "success"
            }
        }

    except HTTPException:
        raise

    except Exception as e:

        if temp_image_path is not None:
            try:
                temp_image_path.unlink(missing_ok=True)
            except Exception:
                pass

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

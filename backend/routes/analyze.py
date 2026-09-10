from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

router = APIRouter()

UPLOAD_DIR = Path("/tmp/satquery_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    query: str = Form(...),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No image filename provided.",
        )

    if not query or not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query must not be empty.",
        )

    suffix = Path(file.filename).suffix or ".jpg"
    temp_path = UPLOAD_DIR / f"satquery_upload{suffix}"

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        temp_path.write_bytes(contents)

        from models.geochat_finetuned.inference_geochat import query_image

        result = query_image(
            str(temp_path),
            query.strip(),
        )

        return {
            "filename": file.filename,
            "query": query.strip(),
            "ai_answer": result["answer"],
            "confidence": result["confidence"],
            "model": result["model"],
            "processing_time_ms": result["processing_time_ms"],
            "status": "success",
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"AI inference failed: {exc}",
        ) from exc

    finally:
        if temp_path.exists():
            temp_path.unlink()

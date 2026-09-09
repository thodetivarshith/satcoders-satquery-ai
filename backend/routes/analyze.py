from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io
router = APIRouter()
@router.post("/api/analyze")
async def analyze_image (file: UploadFile = File(...), query: str = Form(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    w, h = image.size
    answer = f"Query: '{query}'. Image {w}x(h) shows a village with houses and agriculture fields.[AI model connecting...]"
    return {
        "filename": file.filename,
        "query": query,
        "ai_answer": answer,
        "status": "success"
    }


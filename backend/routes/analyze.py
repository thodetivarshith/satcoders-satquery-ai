from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io
router = APIRouter()
@router.post("/api/analyze")
async def analyze_image(file:UploadFile = File(...), query: str = Form(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    width,height = image.size
    answer = f"your query was: '{query}' . Image size is{width}x{height}. The image shows a village with houses and agriculture fields."

    return{"message": "Image received successfully","status": "success", "filename": file.filename, "query": query, "ai_answer": answer, "image_info": f"{width}x{height}"}

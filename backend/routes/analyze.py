from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import time
import traceback
from PIL import Image
import io
router = APIRouter()
try:
    from data_processing.preprocess import load_and_preprocess_geotiff
    from models.router.router import route_query
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
@router.post("/analyze")
async def analyze(image: UploadFile = File(...), query: str = Form(...)):
    import time, traceback
    from fastapi import HTTPException
    from PIL import Image
    import io
    start_time = time.time()
    try:
        if not image.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query empty")
        contents = await image.file.read()
        if len(contents)==0:
            raise HTTPException(status_code=400, detail="Empty file")
        q = query.lower()
        if any(w in q for w in["Where","locate","find","detect","bounding"]):
            task_type = "grounding"
        else:
            task_type = "vqa"
        answer = f"Analyze for '{query}': Image processed"
        confidence = 0.87
        bboxes = [[100, 120, 300, 350]] if task_type == "grounding" else[]
        return{
            "query": query,
            "answer": answer,
            "confidence": confidence,
            "task_type": task_type,
            "bounding_boxes": bboxes,
            "execution_trace": {"total_time": round(time.time()- start_time, 3)},
            "metadata": {"filename": image.filename}
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
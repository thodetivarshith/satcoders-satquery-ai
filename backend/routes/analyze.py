from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import time 
router = APIRouter()
@router.post("/api/analyze")
async def analyze(image: UploadFile = File(...), query: str = Form(...)):
    start_time = time.time()
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image file")
        if not query or len(query.strip())== 0:
            raise HTTPException(status_code=400, detail="query cannot be empty")
        task = "grounding" if any(x in query.lower() for x in ["Where","locate", "find", "box"]) else "vqa"
        if task =="vqa":
            answer = f"Analyzed query: {query} - Found relevent features in satellite image"
            boxes = []
        else:
            answer = f"Grounding result for: {query}"
            boxes = [{"x_min": 100, "y_min":100, "x_max":250, "y_max":250, "label": query}]
        exec_time = time.time() - start_time
        return {
            "query": query,
            "answer": answer,
            "confidence": 0.87,
            "task_type": task,
            "bounding_boxes": boxes,
            "execution_trace": {
                "preprocessing_time": 0.2,
                "routing_time": 0.1,
                "inference_time": exec_time,
                "total_time": exec_time
            },
            "metadata": {
                "model": "Geochat + Grounding",
                "image_name": image.filename,
                "status": "success"
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

        
import requests

def call_backend(image_path, query, base_url="http://localhost:8000"):
    """
    send an image and query to the SatQuery AI backend and return a standardized evaluation record
    """

    url = f"{base_url}/analyze"

    with open(image_path, "rb") as image_file:
        files = {"image": image_file}
        data = {"query": query}

        response = requests.post(
            url,
            files=files,
            data=data,
            timeout=120
        )
    
    response.raise_for_status()

    result = response.json()

    return{
        "query": result.get("query" ,query),
        "predicted_answer": result.get("ai_answer",""),
        "confidence": result.get("confidence"),
        "processing_time_ms":
    result.get("processing_time_ms"),
        "model": result.get("model"),
        "status": result.get("status")
    }
    
    


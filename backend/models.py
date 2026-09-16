from pydantic import BaseModel
class AnalyzeResponse(BaseModel):
    message: str
    status: str
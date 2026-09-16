from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.health import router as health_router
from backend.routes.analyze import router as analyze_router
app = FastAPI(title="SatQuery AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"],allow_methods=["*"],allow_headers=["*"],)
app.include_router(health_router , prefix="/api")
app.include_router(analyze_router , prefix="/api")
@app.get("/")
def root():
    return{"message":"SatQuery AI Backend Running"}
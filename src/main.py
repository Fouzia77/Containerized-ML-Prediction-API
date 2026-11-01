"""FastAPI application for image classification."""
from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src import model

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once when the service starts, not on each prediction request.
    model.get_model()
    yield


app = FastAPI(title="Containerized ML Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported content type. Upload a JPEG or PNG image.")
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=422, detail="Uploaded image is empty.")
        result = model.predict(content)
        if not isinstance(result, dict) or not isinstance(result.get("class_label"), str) or not isinstance(result.get("probabilities"), list):
            raise RuntimeError("Prediction function returned an invalid response format")
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal prediction error") from exc
    finally:
        await file.close()


@app.exception_handler(Exception)
async def unexpected_exception_handler(request, exc):
    logger.exception("Unhandled application error", exc_info=exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})

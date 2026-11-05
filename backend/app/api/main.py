
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.services.model_service import seismopulse_service
from backend.app.schemas.prediction import PredictionRequest


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seismopulse")

ROOT = Path(__file__).resolve().parents[3]

# Maximum incoming JSON request size: 25 MB
MAX_REQUEST_SIZE = 25 * 1024 * 1024


class RequestSizeLimitMiddleware:
    """Reject requests that exceed the configured body-size limit."""

    def __init__(self, app, max_bytes):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = None

        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    content_length = int(value)
                except ValueError:
                    response = JSONResponse(
                        status_code=400,
                        content={"detail": "Invalid Content-Length header."}
                    )
                    await response(scope, receive, send)
                    return
                break

        if content_length is not None and content_length > self.max_bytes:
            response = JSONResponse(
                status_code=413,
                content={"detail": "Upload too large. Maximum size is 25 MB."}
            )
            await response(scope, receive, send)
            return

        received = 0

        async def limited_receive():
            nonlocal received

            message = await receive()

            if message["type"] == "http.request":
                received += len(message.get("body", b""))

                if received > self.max_bytes:
                    raise RequestTooLargeError()

            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestTooLargeError:
            response = JSONResponse(
                status_code=413,
                content={"detail": "Upload too large. Maximum size is 25 MB."}
            )
            await response(scope, receive, send)


class RequestTooLargeError(Exception):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading SeismoPulse model...")
    seismopulse_service.load_model()
    logger.info("SeismoPulse is ready.")
    yield


app = FastAPI(
    title="SeismoPulse API",
    description="AI-powered seismic P-wave and S-wave detection.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    RequestSizeLimitMiddleware,
    max_bytes=MAX_REQUEST_SIZE
)


@app.get("/")
def root():
    return {
        "message": "SeismoPulse API is running",
        "status": "healthy"
    }


@app.get("/api/model-info")
def model_info():
    return seismopulse_service.model_info()


@app.post("/api/predict")
def predict(request: PredictionRequest):
    try:
        result = seismopulse_service.predict(
            request.waveform,
            request.sample_rate_hz
        )
        return result

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    except Exception:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal server error."
        )


app.mount(
    "/dashboard",
    StaticFiles(directory=ROOT / "frontend", html=True),
    name="dashboard"
)

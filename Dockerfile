
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Install CPU-only PyTorch first to avoid downloading CUDA packages.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

# Install API and numerical dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and trained model.
COPY backend/ backend/
COPY src/ src/
COPY frontend/ frontend/
VOLUME ["/models"]
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
# FREUID Challenge 2026 — reproducibility Dockerfile (template)
#
# Teams: replace the base image, dependencies, and COPY steps with your stack.
# Organizers run with: --network none -v DATA:/data:ro -v OUT:/submissions

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Default paths inside the sandbox (do not change without updating the contract)
    FREUID_DATA_DIR=/data \
    FREUID_OUTPUT_DIR=/submissions \
    FREUID_SUBMISSION_PATH=/submissions/submission.csv

WORKDIR /app

# System libraries for OpenCV / Pillow-style image loading (trim if unused)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY prepare_submission.py .

# --- Team-specific section (examples) ---
# COPY models/ /models/
# COPY src/ /app/src/
# ENV MODEL_PATH=/models/best.ckpt

RUN useradd --create-home --uid 1000 runner
USER runner

ENTRYPOINT ["python", "/app/prepare_submission.py"]
CMD []

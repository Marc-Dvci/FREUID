# FREUID Challenge 2026 reproducibility image.
# The organizers provide an NVIDIA A100 and execute this image with --network none.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FREUID_DATA_DIR=/data \
    FREUID_OUTPUT_DIR=/submissions \
    FREUID_SUBMISSION_PATH=/submissions/submission.csv \
    OMP_NUM_THREADS=24 \
    MKL_NUM_THREADS=24

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY docker-requirements.txt /app/docker-requirements.txt
RUN pip install --no-cache-dir -r /app/docker-requirements.txt

COPY src /app/src
COPY docker/prepare_submission.py /app/prepare_submission.py
COPY checkpoints/cnxb384_full.pth /models/cnxb384_full.pth
COPY checkpoints/dinov2b_full.pth /models/dinov2b_full.pth
COPY checkpoints/fnoise_full.pth /models/fnoise_full.pth

ENTRYPOINT ["python", "/app/prepare_submission.py"]

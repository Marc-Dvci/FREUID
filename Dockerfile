# FREUID Challenge 2026 reproducibility image.
# The organizers provide an NVIDIA A100 and execute this image with --network none.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FREUID_DATA_DIR=/data \
    FREUID_OUTPUT_DIR=/submissions \
    FREUID_SUBMISSION_PATH=/submissions/submission.csv \
    CUDA_CACHE_DISABLE=1 \
    TORCH_HOME=/nonexistent \
    HF_HOME=/nonexistent \
    OMP_NUM_THREADS=24 \
    MKL_NUM_THREADS=24

WORKDIR /app

COPY docker-requirements.txt /app/docker-requirements.txt
RUN pip install --no-cache-dir -r /app/docker-requirements.txt

COPY src /app/src
COPY docker/prepare_submission.py /app/prepare_submission.py
COPY checkpoints/cnxb512_MAURITIUS-ID.pth /models/cnxb512_MAURITIUS-ID.pth
COPY checkpoints/dinov2b_full.pth /models/dinov2b_full.pth
RUN echo "aebf36acdf23dfe7a1542e9b07ba94e782910b547d6244f3fbb8d84083066510  /models/cnxb512_MAURITIUS-ID.pth" | sha256sum -c - \
    && echo "f333c566bef9d6c2e9be8ba4a5b5efb5e708492e71162a135856a5e08ee1e8ca  /models/dinov2b_full.pth" | sha256sum -c -

ENTRYPOINT ["python", "/app/prepare_submission.py"]

# torch._dynamo calls tempfile.gettempdir() at import; an anonymous volume keeps /tmp
# writable even under `docker run --read-only`, so the image needs no extra caller flags.
VOLUME ["/tmp"]

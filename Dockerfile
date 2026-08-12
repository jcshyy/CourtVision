FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt backend/api-requirements.txt /app/backend/
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
RUN python -m pip install --no-cache-dir \
        --index-url "$PYTORCH_INDEX_URL" \
        torch==2.12.1 torchvision==0.27.1 \
    && python -m pip install --no-cache-dir -r /app/backend/requirements.txt

COPY . /app

# Model weights are deliberately excluded from the image. Mount the five files
# into /app/backend/models or provision them in the deployment platform.
# Keep the default useful for one-off CLI analysis while allowing AWS Batch to
# replace the command with `python -m backend.app.batch_job`.
CMD ["python", "main.py"]

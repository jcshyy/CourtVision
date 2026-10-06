FROM python:3.14-slim AS system

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt backend/api-requirements.txt /app/backend/
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
# Keep the largest CUDA wheels in separate image layers. ECR can resume these
# independently if a multi-gigabyte upload is interrupted.
RUN python -m pip install --no-cache-dir --no-deps --index-url "$PYTORCH_INDEX_URL" nvidia-cublas==13.1.1.3

FROM system AS cudnn
RUN python -m pip install --no-cache-dir --no-deps --index-url "$PYTORCH_INDEX_URL" nvidia-cudnn-cu13==9.20.0.48

FROM cudnn AS cuda_math
RUN python -m pip install --no-cache-dir --no-deps --index-url "$PYTORCH_INDEX_URL" \
        nvidia-cufft==12.0.0.61 nvidia-curand==10.4.0.35

FROM cuda_math AS cuda_solver
RUN python -m pip install --no-cache-dir --no-deps --index-url "$PYTORCH_INDEX_URL" \
        nvidia-cusolver==12.0.4.66 nvidia-cusparse==12.6.3.3 nvidia-cusparselt-cu13==0.8.1

FROM cuda_solver AS nccl
RUN python -m pip install --no-cache-dir --no-deps --index-url "$PYTORCH_INDEX_URL" nvidia-nccl-cu13==2.29.7

FROM nccl AS cuda_runtime
RUN python -m pip install --no-cache-dir --no-deps --index-url "$PYTORCH_INDEX_URL" \
        nvidia-cuda-cupti==13.0.85 nvidia-cuda-nvrtc==13.0.88 \
        nvidia-cuda-runtime==13.0.96 nvidia-cufile==1.15.1.6 \
        nvidia-nvjitlink==13.0.88 nvidia-nvshmem-cu13==3.4.5 nvidia-nvtx==13.0.85

FROM cuda_runtime AS torch_runtime
RUN python -m pip install --no-cache-dir --no-deps --index-url "$PYTORCH_INDEX_URL" torch==2.12.1

FROM torch_runtime AS vision_runtime
RUN python -m pip install --no-cache-dir --no-deps --index-url "$PYTORCH_INDEX_URL" \
        torchvision==0.27.1 triton==3.7.1
# Keep the remaining Python dependency groups in resumable layers as well.
# Docker Desktop's proxy can reset long ECR uploads, so no individual layer
# should contain the entire application dependency set.

FROM vision_runtime AS api_runtime
RUN python -m pip install --no-cache-dir -r /app/backend/api-requirements.txt

FROM api_runtime AS scientific_runtime
RUN python -m pip install --no-cache-dir \
        numpy==2.5.0 opencv-python==5.0.0.93 imageio-ffmpeg==0.6.0 \
        pandas==3.0.3 pillow==12.3.0 scipy==1.18.0

FROM scientific_runtime AS transformer_runtime
RUN python -m pip install --no-cache-dir transformers==5.13.0

FROM transformer_runtime AS vision_tools
RUN python -m pip install --no-cache-dir supervision==0.29.1 ultralytics==8.4.87

FROM vision_tools AS worker
RUN python -m pip install --no-cache-dir -r /app/backend/requirements.txt \
    && python -m pip check

COPY . /app

# Model weights are deliberately excluded from the image. Mount the four files
# into /app/backend/models or provision them in the deployment platform.
# Keep the default useful for one-off CLI analysis while allowing AWS Batch to
# replace the command with `python -m backend.app.batch_job`.
CMD ["python", "main.py"]

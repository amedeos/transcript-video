# Containerfile (OCI / Podman convention).
#
# Builds an image with whisperX + pyannote pre-installed, ready to run
# `transcribe-video` and `transcript-to-md` on host audio/video files.
#
# Build (with podman):
#     podman build -t transcript-video -f Containerfile .
#
# Run (GPU, NVIDIA CDI):
#     podman run --rm \
#         --device nvidia.com/gpu=all \
#         --security-opt=label=disable \
#         -e HF_TOKEN="$HF_TOKEN" \
#         -v "$(pwd):/data:Z" \
#         -v "$HOME/.cache/huggingface:/root/.cache/huggingface:Z" \
#         transcript-video \
#         meeting.mp4 --md --tag openshift
#
# Run (CPU only, --device cpu):
#     podman run --rm \
#         -e HF_TOKEN="$HF_TOKEN" \
#         -v "$(pwd):/data:Z" \
#         -v "$HOME/.cache/huggingface:/root/.cache/huggingface:Z" \
#         transcript-video \
#         meeting.mp4 --device cpu --md
#
# The default working directory inside the container is /data, so paths
# in the command line are relative to the host directory mounted there.
# The HuggingFace cache is shared with the host so the diarization model
# is downloaded only once across runs.

FROM docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Tooling: Python 3.12 (via deadsnakes), ffmpeg (mandatory for whisperX
# audio loading), curl (for the uv installer), and a small set of build
# helpers. Then clean up apt lists to keep the image lean.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# uv is the primary package manager for this project; falling back to pip
# would also work. Installed into /usr/local/bin so it's on $PATH for any
# user.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /opt/transcript-video

# Install dependencies first (layer cache) — copy only what the build
# needs, not the whole tree, so source-only edits don't invalidate the
# slow whisperX + torch install.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN uv venv --python 3.12 /opt/transcript-video/.venv && \
    . /opt/transcript-video/.venv/bin/activate && \
    uv pip install -e ".[cuda]"

ENV PATH="/opt/transcript-video/.venv/bin:${PATH}"

# Sensible defaults for the runtime UX:
# - /data is where the user's audio/video lives (bind-mount target).
# - The HF cache is shared with the host so model downloads persist.
WORKDIR /data
VOLUME ["/data", "/root/.cache/huggingface"]

# The image's purpose is the transcription pipeline. `--help` is a safe
# no-op when no arguments are provided.
ENTRYPOINT ["transcribe-video"]
CMD ["--help"]

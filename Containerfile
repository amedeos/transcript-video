# Containerfile (OCI / Podman convention).
#
# Builds an image with whisperX + pyannote pre-installed, ready to run
# `transcribe-video` and `transcript-to-md` on host audio/video files.
#
# Base choice: nvcr.io/nvidia/cuda over docker.io to use NVIDIA's own
# registry (more authoritative than the Docker Hub mirror), and ubi9
# over ubuntu so the stack is Red Hat-aligned. CUDA tag is 12.8.2 to
# match PyTorch's `torch==X.Y.Z+cu128` wheel — the version pip installs
# from PyPI today. Going to CUDA 13 base would mismatch with the
# wheel's bundled libs; revisit once PyTorch publishes `+cu13x`.
# UBI10 is not yet an option: NVIDIA only publishes ubi10 tags for
# CUDA 13.2.1, not for the 12.x line.
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

FROM nvcr.io/nvidia/cuda:12.8.2-cudnn-runtime-ubi9

# Tooling: Python 3.12 (UBI9 ships it as a direct package since 9.4),
# curl + xz (for fetching the static ffmpeg build), tar (for extraction).
# UBI9 does NOT include ffmpeg by default — it's omitted from RHEL/UBI
# for licensing reasons. We pull a static GPL build from John Van
# Sickle's canonical mirror; single binary, no system deps.
RUN dnf install -y --setopt=install_weak_deps=False \
        python3.12 \
        python3.12-devel \
        curl \
        tar \
        xz && \
    dnf clean all && \
    rm -rf /var/cache/dnf

# Make `python` and `python3` resolve to 3.12 so uv/pip helpers work
# without explicit version selection.
RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python

# Static ffmpeg build (johnvansickle.com is the canonical mirror linked
# from the official ffmpeg site).
RUN curl -fsSL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
        -o /tmp/ffmpeg.tar.xz && \
    mkdir -p /tmp/ffmpeg && \
    tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1 && \
    install -m 0755 /tmp/ffmpeg/ffmpeg /usr/local/bin/ffmpeg && \
    install -m 0755 /tmp/ffmpeg/ffprobe /usr/local/bin/ffprobe && \
    rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz

# uv is the primary package manager for this project (pip still works
# as a fallback). Installed under /usr/local/bin so it's on $PATH for
# any user.
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

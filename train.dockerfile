# # Base image
# FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# GPU-enabled base image
FROM nvcr.io/nvidia/pytorch:22.07-py3

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install uv manually
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /

COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock

ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked

COPY src/ src/
COPY data/ data/

RUN uv run src/data.py data/corruptmnist_v1 data/processed

ENTRYPOINT ["uv", "run", "src/train.py"]
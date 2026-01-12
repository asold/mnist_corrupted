# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock

ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked

COPY src/ src/
COPY data/ data/
COPY models/ models/

ENTRYPOINT ["uv", "run", "src/evaluate.py"]
CMD ["models/model.pth"]

FROM ghcr.io/astral-sh/uv:0.13.7 AS uv
FROM python:3.14-slim
COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev
EXPOSE 8080 9400 8002
CMD ["/app/.venv/bin/gpu-sim"]

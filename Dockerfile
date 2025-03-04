FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-22.04

USER vscode
WORKDIR /home/vscode/workspace

ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/home/vscode/workspace/.venv

RUN mkdir -p /home/vscode/.local/bin
COPY --from=ghcr.io/astral-sh/uv:0.6.4 /uv /uvx /home/vscode/.local/bin/

COPY . .

# RUN /home/vscode/.local/bin/uv lock
RUN /home/vscode/.local/bin/uv sync --frozen

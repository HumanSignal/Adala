# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11
ARG POETRY_VERSION=1.8.1

FROM python:${PYTHON_VERSION}-slim
ARG POETRY_VERSION
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_CACHE_DIR="/.cache" \
    POETRY_HOME="/opt/poetry" \
    POETRY_CACHE_DIR="/.poetry-cache" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    VENV_PATH="/usr/src/app/.venv"

RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install --no-install-recommends -y git gcc python3-dev; \
    apt-get autoremove -y

# Install Poetry
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install poetry==${POETRY_VERSION}

# Set the working directory in the container to where the source is mounted as
# a volume
WORKDIR /usr/src/app

COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN --mount=type=cache,target=${POETRY_CACHE_DIR} \
    poetry install --no-interaction --no-ansi --no-root --with label-studio

COPY . .

# Install adala
RUN --mount=type=cache,target=${POETRY_CACHE_DIR} \
    poetry install --no-interaction --no-ansi --only-root

# Set the working directory in the container to where the app will be run from
WORKDIR /usr/src/app/server

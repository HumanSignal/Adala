# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12
ARG POETRY_VERSION=2.1.4

################################ Overview

# This Dockerfile builds an Adala environment.
# It consists of three main stages:
# 1. "base-image" - Prepares common env variables and installs Poetry.
# 2. "venv-builder" - Prepares the virtualenv environment.
# 3. "prod" - Creates the final production image with the Adala source and its venv.

################################ Stage: base image
# Creating a python base with shared environment variables
FROM python:${PYTHON_VERSION}-slim AS python-base
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

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install --no-install-recommends -y procps; \
    apt-get autoremove -y

RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install poetry==${POETRY_VERSION}

################################ Stage: venv-builder (prepare the virtualenv)
FROM python-base AS venv-builder
ARG PYTHON_VERSION

RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install --no-install-recommends -y git gcc python3-dev; \
    apt-get autoremove -y

# Set the working directory in the container to where the source is mounted as
# a volume
WORKDIR /usr/src/app

COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN --mount=type=cache,target=${POETRY_CACHE_DIR} \
    poetry install --no-interaction --no-ansi --no-root --without dev

COPY . .

# Install adala
RUN --mount=type=cache,target=${POETRY_CACHE_DIR} \
    poetry install --no-interaction --no-ansi --only-root

################################### Stage: prod
FROM python-base AS production

# Copy artifacts from other stages
COPY --from=venv-builder --chown=1001:0 /usr/src/app /usr/src/app

ENV LITELLM_LOG=WARNING

# Set the working directory in the container to where the app will be run from
WORKDIR /usr/src/app/server

# Install netcat for connection checking
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install --no-install-recommends -y netcat-openbsd; \
    apt-get autoremove -y

# Copy and setup entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user
USER 1001

# Set the entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

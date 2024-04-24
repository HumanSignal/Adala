# Use an official lightweight Python image
FROM python:3.11-slim

# Install git
RUN apt-get update && apt-get install -y git gcc

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install Poetry
RUN pip install poetry==1.8.1

# Set the working directory in the container to where the source is mounted as
# a volume
WORKDIR /usr/src/app

COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --with label-studio

COPY . .

# Install adala and the app
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --with label-studio

# Set the working directory in the container to where the app will be run from
WORKDIR /usr/src/app/server

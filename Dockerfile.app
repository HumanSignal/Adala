# Use an official lightweight Python image
FROM python:3.11-slim

# Install git
RUN apt-get update && apt-get install -y git gcc

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip

# Install Poetry
RUN pip install poetry==1.8.1

# Set the working directory in the container to where the source is mounted as
# a volume
WORKDIR /usr/src/app

COPY pyproject.toml poetry.lock ./

# fix error:
# KeyError: 'PEP517_BUILD_BACKEND'
# Note: This error originates from the build backend, and is likely not a problem with poetry but with paginate (0.5.6) not supporting PEP 517 builds. You can verify this by running 'pip wheel --no-cache-dir --use-pep517 "paginate (==0.5.6)"'.
RUN poetry run pip install --upgrade pip setuptools wheel
RUN poetry run python -m pip install paginate==0.5.6 --no-cache-dir --no-use-pep517 

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --with label-studio

COPY . .

# Install adala and the app
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --with label-studio

# Set the working directory in the container to where the app will be run from
WORKDIR /usr/src/app/server

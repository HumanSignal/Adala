# Use an official lightweight Python image
FROM python:3.11-slim

# Install git
RUN apt-get update && apt-get install -y git

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install Poetry
RUN pip install poetry==1.8.1

# Set the working directory in the container to where the source is mounted as
# a volume
WORKDIR /usr/src/app

COPY pyproject.toml poetry.lock ./

# Install dependencies and install adala editable
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

COPY . .

# Install the app
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Set the working directory in the container to where the app will be run from
WORKDIR /usr/src/app/server

# Command to run on container start
CMD ["poetry", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

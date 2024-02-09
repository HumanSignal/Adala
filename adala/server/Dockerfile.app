# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies from workers env
COPY workers/requirements.txt ./requirements-workers.txt
RUN pip install --no-cache-dir -r requirements-workers.txt

# Install dependencies from the main env
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's source code from your host to your image filesystem
COPY . .

# Command to run on container start
CMD ["uvicorn", "app", "--host", "0.0.0.0", "--port", "8000"]

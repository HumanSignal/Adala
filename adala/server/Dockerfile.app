# Use an official lightweight Python image
FROM python:3.11-slim

# Install git
RUN apt-get update && apt-get install -y git

# Set the working directory in the container
WORKDIR /app

# Install dependencies from the main env
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's source code from your host to your image filesystem
COPY . .

# Command to run on container start
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

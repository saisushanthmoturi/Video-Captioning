# Use a slim python base image
FROM python:3.11-slim

# Install system dependencies (including ffmpeg for video utility)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory inside the container
WORKDIR /app

# Copy python dependencies file and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the agent, captioner, and styles scripts
COPY *.py /app/

# Declare the input and output volume mount points
VOLUME [ "/input", "/output" ]

# Set execution command
ENTRYPOINT [ "python", "/app/agent.py" ]

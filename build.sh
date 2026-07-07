#!/bin/bash
# Exit on error
set -e

IMAGE_NAME="video-captioning-agent"
TAG="latest"

echo "Building Docker image targeting linux/amd64..."
docker build --platform linux/amd64 -t "${IMAGE_NAME}:${TAG}" .

echo "Docker image built successfully: ${IMAGE_NAME}:${TAG}"
echo "To run the container locally with mounted folders:"
echo "docker run --rm -v \$(pwd)/input:/input -v \$(pwd)/output:/output ${IMAGE_NAME}:${TAG}"

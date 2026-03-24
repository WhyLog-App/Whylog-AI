#!/usr/bin/env bash

set -euo pipefail

IMAGE_NAME="whylog/whylog-fastapi"
TAG="latest"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_REF="${IMAGE_NAME}:${TAG}"

echo "Building ${IMAGE_REF}"
docker build --platform linux/amd64 -t "${IMAGE_REF}" "${SCRIPT_DIR}"

echo "Pushing ${IMAGE_REF}"
docker push "${IMAGE_REF}"

echo "Done: ${IMAGE_REF}"

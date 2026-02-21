#!/usr/bin/env bash
# Build Docker image for basic-chatbot: project name + version tag.
# Usage: ./build.sh <dockerhub_user> <version> <arch>

set -euo pipefail

usage() {
  echo "Usage: $0 <dockerhub_user> <version> <arch>" >&2
  echo "  dockerhub_user  Docker Hub username (e.g. yourusername)" >&2
  echo "  version         Image version tag (e.g. 1.0.0)" >&2
  echo "  arch            Architecture: arm64 or amd64" >&2
  echo "" >&2
  echo "Example: $0 dleon24 1.0.0 amd64" >&2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$SCRIPT_DIR")"

# Parse arguments (show help if -h/--help)
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

if [ $# -lt 3 ]; then
  usage
  exit 1
fi

DOCKERHUB_USER="$1"
VERSION="$2"
ARCH="$3"

# Validate required arguments
if [ -z "$DOCKERHUB_USER" ]; then
  echo "ERROR: dockerhub_user is required." >&2
  usage
  exit 1
fi

if [ -z "$VERSION" ]; then
  echo "ERROR: version is required." >&2
  usage
  exit 1
fi

case "$ARCH" in
  arm64)  PLATFORM="linux/arm64" ;;
  amd64)  PLATFORM="linux/amd64" ;;
  *)
    echo "ERROR: arch must be 'arm64' or 'amd64', got: $ARCH" >&2
    usage
    exit 1
    ;;
esac

# Validate Docker is available
if ! command -v docker &>/dev/null; then
  echo "ERROR: docker is not installed or not in PATH." >&2
  exit 1
fi

# Validate Dockerfile exists in build context
if [ ! -f "$SCRIPT_DIR/Dockerfile" ]; then
  echo "ERROR: Dockerfile not found in $SCRIPT_DIR" >&2
  exit 1
fi

IMAGE_TAG="${DOCKERHUB_USER}/${PROJECT_NAME}:${VERSION}"

echo "Building image: $IMAGE_TAG ($PLATFORM)"
docker build --pull --platform "$PLATFORM" -t "$IMAGE_TAG" "$SCRIPT_DIR"

echo "Done. Image tagged as: $IMAGE_TAG"

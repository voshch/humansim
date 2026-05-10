#!/usr/bin/env bash
# Build the arena_humansim docker image (idempotent, layer-cached) and drop
# into a shell with the package built and sourced. The host repo is bind-mounted
# at /opt/arena_ws/src so paths line up 1:1 with the in-image build, edits to
# Python sources / YAML scenarios take effect immediately, and outputs written
# under $ARENA_DATA_DIR (./data on host) are visible without copying.

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-arena_humansim:review}"

docker build -t "$IMAGE" "$DIR"

mkdir -p "$DIR/data" "$DIR/renders"

exec docker run --rm -it \
  -v "$DIR:/opt/arena_ws/src" \
  -v "$DIR/data:/data" \
  --network host \
  --workdir /opt/arena_ws/src \
  "$IMAGE"

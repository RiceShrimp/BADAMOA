#!/usr/bin/env bash
# BADAMOA node installer (Linux). Served by the Hub with HUB_URL filled in.
#   curl -fsSL <hub>/install/node.sh | sudo bash -s -- --token <join-token> [--roles compute,storage]
set -euo pipefail

HUB_URL="{{HUB_URL}}"
VERSION="{{VERSION}}"
TOKEN=""
ROLES=""

while [ $# -gt 0 ]; do
  case "$1" in
    --token) TOKEN="$2"; shift 2 ;;
    --roles) ROLES="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$TOKEN" ]; then echo "--token is required" >&2; exit 1; fi
if [ "$(id -u)" -ne 0 ]; then echo "run as root (sudo)" >&2; exit 1; fi

ROOT=/opt/badamoa
export UV_INSTALL_DIR="$ROOT/bin"
export UV_TOOL_DIR="$ROOT/tools"
export UV_TOOL_BIN_DIR="$ROOT/bin"
export UV_PYTHON_INSTALL_DIR="$ROOT/python"
export PATH="$ROOT/bin:$PATH"
mkdir -p "$ROOT/bin"

if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi

echo "==> installing badamoa node $VERSION from $HUB_URL"
uv tool install --force --python 3.12 --index "$HUB_URL/install/simple/" "badamoa==$VERSION"
ln -sf "$ROOT/bin/badamoa" /usr/local/bin/badamoa

echo "==> joining hub"
if [ -n "$ROLES" ]; then
  badamoa node join --hub "$HUB_URL" --token "$TOKEN" --name "$(hostname)"
else
  badamoa node join --hub "$HUB_URL" --token "$TOKEN"
fi

echo "==> installing systemd service"
badamoa node install-service

echo "BADAMOA node installed. Check the Hub UI → 노드."

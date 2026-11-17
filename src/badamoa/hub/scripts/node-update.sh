#!/usr/bin/env bash
# BADAMOA node update (Linux): the program only; this machine stays registered as the same node.
# For nodes installed before they could update themselves (0.30.0 and earlier). Served by the Hub:
#   curl -fsSL <hub>/install/node-update.sh | sudo bash
set -euo pipefail

HUB_URL="{{HUB_URL}}"
VERSION="{{VERSION}}"

if [ "$(id -u)" -ne 0 ]; then echo "run as root (sudo)" >&2; exit 1; fi

ROOT=/opt/badamoa
export UV_INSTALL_DIR="$ROOT/bin"
export UV_TOOL_DIR="$ROOT/tools"
export UV_TOOL_BIN_DIR="$ROOT/bin"
export UV_PYTHON_INSTALL_DIR="$ROOT/python"
export PATH="$ROOT/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found: this machine was not set up with the node installer (node.sh)" >&2
  exit 1
fi

echo "==> updating badamoa node to $VERSION from $HUB_URL"
uv tool install --force --python 3.12 --index "$HUB_URL/install/simple/" "badamoa==$VERSION"

echo "==> opening TCP 8475 (other nodes take this node's registered images from here, not through the Hub)"
if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -q "Status: active"; then
  ufw allow 8475/tcp comment "BADAMOA node data" >/dev/null || true
elif command -v firewall-cmd >/dev/null 2>&1 && firewall-cmd --state >/dev/null 2>&1; then
  { firewall-cmd --permanent --add-port=8475/tcp && firewall-cmd --reload; } >/dev/null 2>&1 || true
fi

echo "==> restarting the service"
systemctl restart badamoa-node

echo "BADAMOA node updated to $VERSION (same registration). From now on it follows the Hub by itself."

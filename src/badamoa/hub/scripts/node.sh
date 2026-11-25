#!/usr/bin/env bash
# BADAMOA node installer (Linux). Served by the Hub with HUB_URL filled in.
#   curl -fsSL <hub>/install/node.sh | sudo bash -s -- --token <join-token>
# Without sudo (no root on that server) the same line without `sudo` installs for this account only:
# the program goes to ~/.badamoa and the node starts at boot from this account's crontab.
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

if [ "$(id -u)" -eq 0 ]; then
  ROOT=/opt/badamoa
else
  ROOT="$HOME/.badamoa"
  echo "==> no root: installing for $(id -un) only, in $ROOT (data you register only needs to be readable by $(id -un))"
fi
export UV_INSTALL_DIR="$ROOT/bin"
export UV_TOOL_DIR="$ROOT/tools"
export UV_TOOL_BIN_DIR="$ROOT/bin"
export UV_PYTHON_INSTALL_DIR="$ROOT/python"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/uv-cache}"  # downloads in the program's folder: gone with it
export PATH="$ROOT/bin:$PATH"
mkdir -p "$ROOT/bin"

if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi

echo "==> installing badamoa node $VERSION from $HUB_URL"
uv tool install --force --python 3.12 --index "$HUB_URL/install/simple/" "badamoa==$VERSION"
if [ "$(id -u)" -eq 0 ]; then
  mkdir -p /usr/local/bin  # not there on a minimal system
  ln -sf "$ROOT/bin/badamoa" /usr/local/bin/badamoa
else
  mkdir -p "$HOME/.local/bin"
  ln -sf "$ROOT/bin/badamoa" "$HOME/.local/bin/badamoa"
fi

echo "==> joining hub"
if [ -n "$ROLES" ]; then
  badamoa node join --hub "$HUB_URL" --token "$TOKEN" --name "$(hostname)"
else
  badamoa node join --hub "$HUB_URL" --token "$TOKEN"
fi

if [ "$(id -u)" -eq 0 ]; then
  echo "==> opening TCP 8475 (other nodes take this node's registered images from here, not through the Hub)"
  if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -q "Status: active"; then
    ufw allow 8475/tcp comment "BADAMOA node data" >/dev/null || true
  elif command -v firewall-cmd >/dev/null 2>&1 && firewall-cmd --state >/dev/null 2>&1; then
    { firewall-cmd --permanent --add-port=8475/tcp && firewall-cmd --reload; } >/dev/null 2>&1 || true
  fi
  echo "==> installing systemd service (starts at boot)"
else
  echo "==> TCP 8475: other nodes take this node's registered images from here. Without root the firewall"
  echo "    is left as it is: if one is on, ask the server's admin to open 8475/tcp."
  echo "==> starting the node, and at every boot (this account's crontab)"
fi
badamoa node install-service

if command -v docker >/dev/null 2>&1; then
  echo "==> Docker: the Hub's in-house registries without HTTPS (insecure-registries; needs root)"
  badamoa node trust-registries --hub "$HUB_URL" || true
fi

echo "BADAMOA node installed. Check the Hub UI → 노드."

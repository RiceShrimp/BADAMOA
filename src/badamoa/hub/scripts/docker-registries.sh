#!/usr/bin/env bash
# Let this server's Docker pull from the Hub's in-house registries that have no HTTPS (insecure-registries in
# /etc/docker/daemon.json; Docker reads it again, running containers go on). For a node installed without sudo:
# it cannot change Docker's settings itself. Served by the Hub:
#   curl -fsSL <hub>/install/docker-registries.sh | sudo bash
set -euo pipefail

HUB_URL="{{HUB_URL}}"

if [ "$(id -u)" -ne 0 ]; then echo "run as root (sudo)" >&2; exit 1; fi
if ! command -v docker >/dev/null 2>&1; then echo "Docker is not installed on this server: nothing to do"; exit 0; fi

# The node program: installed with sudo (/opt/badamoa), or by the account that ran sudo (~/.badamoa)
BIN=""
OWNER_HOME=""
if [ -n "${SUDO_USER:-}" ]; then OWNER_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6 || true)"; fi
for candidate in "$(command -v badamoa || true)" /opt/badamoa/bin/badamoa "${OWNER_HOME:+$OWNER_HOME/.badamoa/bin/badamoa}"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then BIN="$candidate"; break; fi
done
if [ -z "$BIN" ]; then
  echo "badamoa is not installed on this server: connect it as a node first (Hub UI → 노드 → 서버 추가)" >&2
  exit 1
fi

"$BIN" node trust-registries --hub "$HUB_URL"

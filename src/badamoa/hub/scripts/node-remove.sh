#!/usr/bin/env bash
# Take the BADAMOA node off this machine: the service, the program, its settings.
# Data folders (images and labels registered from here) are not touched. Served by the Hub:
#   curl -fsSL <hub>/install/node-remove.sh | sudo bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then echo "run as root (sudo)" >&2; exit 1; fi

echo "==> stopping and unregistering the service"
systemctl disable --now badamoa-node 2>/dev/null || true
rm -f /etc/systemd/system/badamoa-node.service
systemctl daemon-reload 2>/dev/null || true

echo "==> removing the program and its settings"
rm -rf /opt/badamoa /var/lib/badamoa/node
if command -v ufw >/dev/null 2>&1; then ufw delete allow 8475/tcp >/dev/null 2>&1 || true; fi
if command -v firewall-cmd >/dev/null 2>&1 && firewall-cmd --state >/dev/null 2>&1; then
  { firewall-cmd --permanent --remove-port=8475/tcp && firewall-cmd --reload; } >/dev/null 2>&1 || true
fi
if [ -L /usr/local/bin/badamoa ]; then rm -f /usr/local/bin/badamoa; fi
rmdir /var/lib/badamoa 2>/dev/null || true

echo "BADAMOA node removed from this machine. Data folders were not touched."

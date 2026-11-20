#!/usr/bin/env bash
# Take the BADAMOA node off this machine: the service, the program, its settings.
# Data folders (images and labels registered from here) are not touched. Served by the Hub:
#   curl -fsSL <hub>/install/node-remove.sh | sudo bash
# A node installed without sudo (in ~/.badamoa) is removed by the same line without `sudo`, as that account.
set -euo pipefail

# The node's work folders BADAMOA made on other disks (environments, work folders of jobs, fetched files):
# the ones it remembers in its home that are named badamoa-work (a folder an admin named otherwise stays).
work_folders() {
  local home="$1"
  [ -f "$home/work-dirs.json" ] || return 0
  python3 - "$home/work-dirs.json" 2>/dev/null <<'PY' || true
import json, os, sys
for path in json.load(open(sys.argv[1])):
    if os.path.basename(os.path.normpath(path)) == "badamoa-work" and os.path.isdir(path):
        print(path)
PY
}

if [ "$(id -u)" -ne 0 ]; then
  ROOT="$HOME/.badamoa"
  if [ ! -d "$ROOT" ]; then
    echo "no BADAMOA node installed for $(id -un) (~/.badamoa). One installed with sudo: run this with sudo." >&2
    exit 1
  fi
  echo "==> stopping the node and taking it out of this account's crontab"
  "$ROOT/bin/badamoa" node uninstall-service 2>/dev/null || true
  echo "==> removing the program and its settings"
  work_folders "${XDG_DATA_HOME:-$HOME/.local/share}/badamoa/node" | while IFS= read -r folder; do rm -rf "$folder"; done
  rm -rf "$ROOT" "${XDG_DATA_HOME:-$HOME/.local/share}/badamoa/node"
  if [ -L "$HOME/.local/bin/badamoa" ]; then rm -f "$HOME/.local/bin/badamoa"; fi
  rmdir "${XDG_DATA_HOME:-$HOME/.local/share}/badamoa" 2>/dev/null || true
  echo "BADAMOA node removed for $(id -un). Data folders were not touched."
  exit 0
fi

echo "==> stopping and unregistering the service"
systemctl disable --now badamoa-node 2>/dev/null || true
rm -f /etc/systemd/system/badamoa-node.service
systemctl daemon-reload 2>/dev/null || true

echo "==> removing the program and its settings"
work_folders /var/lib/badamoa/node | while IFS= read -r folder; do rm -rf "$folder"; done
rm -rf /opt/badamoa /var/lib/badamoa/node
if command -v ufw >/dev/null 2>&1; then ufw delete allow 8475/tcp >/dev/null 2>&1 || true; fi
if command -v firewall-cmd >/dev/null 2>&1 && firewall-cmd --state >/dev/null 2>&1; then
  { firewall-cmd --permanent --remove-port=8475/tcp && firewall-cmd --reload; } >/dev/null 2>&1 || true
fi
if [ -L /usr/local/bin/badamoa ]; then rm -f /usr/local/bin/badamoa; fi
rmdir /var/lib/badamoa 2>/dev/null || true

echo "BADAMOA node removed from this machine. Data folders were not touched."

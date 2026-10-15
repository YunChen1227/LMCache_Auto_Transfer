#!/usr/bin/env bash
# Create github.com/YunChen1227/LMCache_Auto_Transfer (if needed) and push main.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GH="${GH_BIN:-gh}"
if ! command -v "$GH" >/dev/null 2>&1; then
  if [[ -x /tmp/gh_2.102.0_macOS_arm64/bin/gh ]]; then
    GH=/tmp/gh_2.102.0_macOS_arm64/bin/gh
  else
    echo "Install GitHub CLI: https://cli.github.com/" >&2
    exit 1
  fi
fi

if ! "$GH" auth status >/dev/null 2>&1; then
  echo "Log in to GitHub (device flow opens in the browser):"
  "$GH" auth login --hostname github.com --git-protocol ssh --skip-ssh-key --web -c
fi

REPO="YunChen1227/LMCache_Auto_Transfer"
if "$GH" repo view "$REPO" >/dev/null 2>&1; then
  echo "Repository $REPO already exists."
else
  "$GH" repo create "$REPO" --public --description "LMCache MP NAS KV journal and startup GPU warmup plugin" --source=. --remote=origin
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "git@github.com:${REPO}.git"
fi

git push -u origin main
echo "Done: https://github.com/${REPO}"

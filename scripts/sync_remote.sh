#!/usr/bin/env bash
set -euo pipefail

# Sync repository files (except lockfiles, Markdown, and JSON) to a remote host via rsync.
#
# Preserves directory structure and prunes caches/vendored trees that would
# otherwise ship caches and environments.

REMOTE_USER="${REMOTE_USER:-unilab}"
REMOTE_HOST="${REMOTE_HOST:-165.245.137.171}"
REMOTE_PATH="${REMOTE_PATH:-~/jiaxi/Unilab_fork_offpolicy_sonic}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

DELETE_FLAG=()
DRY_RUN=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --delete | -d)
      DELETE_FLAG=(--delete)
      ;;
    --dry-run | -n)
      DRY_RUN=(--dry-run)
      ;;
    --help | -h)
      cat <<'EOF'
Usage: sync_remote.sh [--delete] [--dry-run]

Sync all files except *.lock, *.md, and *.json, preserving directory structure
and pruning the root data/logs trees plus .git / .venv / __pycache__ and other
caches.

Options:
  --delete    mirror included runtime files that no longer exist locally
  --dry-run   show what would be transferred without doing it

Environment:
  REMOTE_USER  (default: unilab)
  REMOTE_HOST  (default: 165.245.137.171)
  REMOTE_PATH  (default: ~/jiaxi/Unilab_fork_offpolicy_sonic)
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
  shift
done

echo "[sync_remote] ${REPO_ROOT} -> ${DEST}"
echo "[sync_remote] transferring files except *.lock/*.md/*.json (delete=${DELETE_FLAG:+yes} dry-run=${DRY_RUN:+yes})"

# Rule order matters: prune unwanted trees first, include directories so their
# contents can be considered, exclude the requested extensions, then include
# every remaining file.
rsync -avz \
  "${DELETE_FLAG[@]}" \
  "${DRY_RUN[@]}" \
  --prune-empty-dirs \
  --exclude='.git/' \
  --exclude='.git' \
  --exclude='/data' \
  --exclude='/logs' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.ruff_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.pytest_cache/' \
  --exclude='.tox/' \
  --exclude='*.egg-info/' \
  --include='*/' \
  --exclude='*.lock' \
  --exclude='*.toml' \
  --exclude='*.yaml' \
  --exclude='*.md' \
  --exclude='*.json' \
  "${REPO_ROOT}/" "${DEST}/"

echo "[sync_remote] done."

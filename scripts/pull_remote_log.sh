#!/usr/bin/env bash
set -euo pipefail

# Pull one training run from the remote UniLab workspace. Checkpoint files are
# intentionally excluded so TensorBoard metadata can be transferred cheaply.

REMOTE_USER="${REMOTE_USER:-unilab}"
REMOTE_HOST="${REMOTE_HOST:-165.245.137.171}"
REMOTE_REPO_PATH="${REMOTE_REPO_PATH:-~/jiaxi/Unilab_fork_offpolicy_sonic}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_LOG_PATH=""
LOCAL_DIR=""
DELETE_FLAG=()
DRY_RUN=()

usage() {
  cat <<'EOF'
Usage: pull_remote_log.sh <remote-log-path> [options]

Pull one remote training run into the local repository, excluding all *.pt
checkpoint files. The remote path may be absolute or relative to the remote
repository root.

Examples:
  pull_remote_log.sh /home/unilab/jiaxi/Unilab_fork_offpolicy_sonic/logs/flash_sac/G1SonicManager/2026-08-31_06-33-39_mujoco_gpux4
  pull_remote_log.sh logs/flash_sac/G1SonicManager/2026-08-31_06-33-39_mujoco_gpux4 --dry-run

Options:
  --local-dir <path>  Override the local destination directory
  --delete, -d        Delete destination files absent from the remote run
  --dry-run, -n       Show transfers without changing local files
  --help, -h          Show this help

Environment:
  REMOTE_USER       default: unilab
  REMOTE_HOST       default: 165.245.137.171
  REMOTE_REPO_PATH  default: ~/jiaxi/Unilab_fork_offpolicy_sonic

By default, logs/<run-relative-path> is written below the current repository.
Only files matching *.pt are excluded; event files, run_config.json, and other
diagnostic files are included.
EOF
}

error() {
  echo "[pull_remote_log] error: $*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --local-dir)
      [ "$#" -ge 2 ] || error "--local-dir requires a path"
      LOCAL_DIR="$2"
      shift 2
      ;;
    --delete | -d)
      DELETE_FLAG=(--delete)
      shift
      ;;
    --dry-run | -n)
      DRY_RUN=(--dry-run --itemize-changes)
      shift
      ;;
    --help | -h)
      usage
      exit 0
      ;;
    -*)
      error "unknown option: $1"
      ;;
    *)
      [ -z "$REMOTE_LOG_PATH" ] || error "only one remote log path may be provided"
      REMOTE_LOG_PATH="$1"
      shift
      ;;
  esac
done

[ -n "$REMOTE_LOG_PATH" ] || {
  usage >&2
  exit 2
}

# Reject traversal before composing either the ssh source or local target.
case "/${REMOTE_LOG_PATH}/" in
  */../* | */.. | ../*) error "remote log path must not contain '..'" ;;
esac

if [[ "$REMOTE_LOG_PATH" == logs/* ]]; then
  LOG_RELATIVE_PATH="${REMOTE_LOG_PATH#logs/}"
  REMOTE_SOURCE="${REMOTE_REPO_PATH%/}/logs/${LOG_RELATIVE_PATH}"
elif [[ "$REMOTE_LOG_PATH" == */logs/* ]]; then
  LOG_RELATIVE_PATH="${REMOTE_LOG_PATH#*/logs/}"
  REMOTE_SOURCE="$REMOTE_LOG_PATH"
else
  error "remote log path must be a logs/<...> path or contain /logs/"
fi

[ -n "$LOG_RELATIVE_PATH" ] || error "remote log path must identify one run directory"

if [ -z "$LOCAL_DIR" ]; then
  LOCAL_DIR="${REPO_ROOT}/logs/${LOG_RELATIVE_PATH}"
elif [[ "$LOCAL_DIR" != /* ]]; then
  LOCAL_DIR="${REPO_ROOT}/${LOCAL_DIR}"
fi

# Resolve the destination lexically so --dry-run does not create directories.
LOCAL_DIR="$(realpath -m "$LOCAL_DIR")"
[ "$LOCAL_DIR" != "$REPO_ROOT" ] || error "local destination may not be the repository root"

DEST="${LOCAL_DIR%/}/"
SOURCE="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_SOURCE%/}/"

echo "[pull_remote_log] ${SOURCE} -> ${DEST}"
echo "[pull_remote_log] excluding *.pt (delete=${DELETE_FLAG:+yes} dry-run=${DRY_RUN:+yes})"

rsync -avz \
  "${DELETE_FLAG[@]}" \
  "${DRY_RUN[@]}" \
  --prune-empty-dirs \
  --exclude='.git/' \
  --exclude='.git' \
  --exclude='__pycache__/' \
  --exclude='.ruff_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.pytest_cache/' \
  --exclude='*.pt' \
  "$SOURCE" "$DEST"

echo "[pull_remote_log] done."

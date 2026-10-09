#!/usr/bin/env bash
set -euo pipefail

# Forward a local TensorBoard port to a TensorBoard instance on the remote host.
# The remote TensorBoard process is intentionally managed separately.

REMOTE_USER="${REMOTE_USER:-unilab}"
REMOTE_HOST="${REMOTE_HOST:-165.245.137.171}"
LOCAL_PORT="${LOCAL_PORT:-6007}"
REMOTE_PORT="${REMOTE_PORT:-6007}"

usage() {
  cat <<'EOF'
Usage: sync_tensorboard.sh [options]

Forward a local port to a TensorBoard server on the remote host. The remote
TensorBoard process must already be running; this script only keeps the SSH
tunnel open in the foreground.

Options:
  --local-port <port>   Local listening port (default: LOCAL_PORT or 6007)
  --remote-port <port>  Remote TensorBoard port (default: REMOTE_PORT or 6007)
  --help, -h            Show this help

Environment:
  REMOTE_USER  SSH user (default: unilab)
  REMOTE_HOST  SSH host (default: 165.245.137.171)
  LOCAL_PORT   Local listening port (default: 6007)
  REMOTE_PORT  Remote TensorBoard port (default: 6007)

After the tunnel is established, open http://127.0.0.1:<local-port> locally.
EOF
}

error() {
  echo "[sync_tensorboard] error: $*" >&2
  exit 1
}

validate_port() {
  local name="$1"
  local value="$2"

  [[ "$value" =~ ^[0-9]{1,5}$ ]] || error "$name must be an integer from 1 to 65535 (got '$value')"
  (( 10#$value >= 1 && 10#$value <= 65535 )) || {
    error "$name must be an integer from 1 to 65535 (got '$value')"
  }
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --local-port)
      [ "$#" -ge 2 ] || error "--local-port requires a port"
      LOCAL_PORT="$2"
      shift 2
      ;;
    --remote-port)
      [ "$#" -ge 2 ] || error "--remote-port requires a port"
      REMOTE_PORT="$2"
      shift 2
      ;;
    --help | -h)
      usage
      exit 0
      ;;
    -* )
      error "unknown option: $1"
      ;;
    *)
      error "unexpected argument: $1"
      ;;
  esac
done

[ -n "$REMOTE_USER" ] || error "REMOTE_USER must not be empty"
[ -n "$REMOTE_HOST" ] || error "REMOTE_HOST must not be empty"
validate_port LOCAL_PORT "$LOCAL_PORT"
validate_port REMOTE_PORT "$REMOTE_PORT"

REMOTE_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
FORWARD_SPEC="${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}"

echo "[sync_tensorboard] forwarding localhost:${LOCAL_PORT} -> ${REMOTE_TARGET}:127.0.0.1:${REMOTE_PORT}"
echo "[sync_tensorboard] press Ctrl-C to close the tunnel"

ssh \
  -N \
  -T \
  -o ExitOnForwardFailure=yes \
  -L "$FORWARD_SPEC" \
  "$REMOTE_TARGET"

#!/usr/bin/env bash
# Stop the server that ./start.sh started on both Sparks and remove its containers, freeing their GPU memory. The
# ranks get STOP_TIMEOUT seconds (default 30) to shut down; requests still running are cut off (it does not drain
# them), so stop.sh says when there are any.
# Usage: ./stop.sh      Env: CONTAINER_NAME, PORT, WORKER (see scripts/config.sh), STOP_TIMEOUT
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh
source ./scripts/nodes.sh

# Where the containers are: here (rank 0) and on the worker (rank 1)
here=0; docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME" && here=1
there=0
if [[ -z "${WORKER:-}" ]]; then
  warn "WORKER is not set (scripts/local.sh): rank 1 was left alone"
elif ! worker true 2>/dev/null; then
  warn "cannot reach the worker ($WORKER) over ssh: rank 1 was left alone"
elif worker "docker ps -a --format '{{.Names}}' | grep -qx '$CONTAINER_NAME'"; then
  there=1
fi
if (( here == 0 && there == 0 )); then
  log "No container named $CONTAINER_NAME on either Spark: nothing to stop"
  exit 0
fi
where="on both Sparks"; (( there )) || where="here"; (( here )) || where="on the worker"

if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; then
  api_host="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && api_host=127.0.0.1
  [[ "$api_host" == *:* ]] && api_host="[$api_host]"
  busy=$(curl -s --max-time 3 "http://$api_host:$PORT/health" 2>/dev/null |
         python3 -c 'import json,sys; print(json.load(sys.stdin).get("requests_running", 0))' 2>/dev/null || echo 0)
  (( busy == 0 )) || warn "$busy request(s) still running will be cut off"
fi
log "Stopping $CONTAINER_NAME $where (up to ${STOP_TIMEOUT:-30}s)"

stop_here() {
  docker stop -t "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null
  log "Stopped and removed rank 0 here"
}
stop_worker() {
  worker "docker stop -t ${STOP_TIMEOUT:-30} '$CONTAINER_NAME' >/dev/null 2>&1; docker rm -f '$CONTAINER_NAME' >/dev/null" ||
    { warn "could not stop rank 1 on $WORKER"; return 1; }
  log "Stopped and removed rank 1 on $WORKER"
}
wpid=""; if (( there )); then stop_worker & wpid=$!; fi
if (( here )); then stop_here; fi
[[ -z "$wpid" ]] || wait "$wpid" || exit 0             # a warning above said what is left
log "Stopped and removed $CONTAINER_NAME $where; its GPU memory is free again"

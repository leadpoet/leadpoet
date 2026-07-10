#!/usr/bin/env bash
# Size-based rotation for the gateway process log.
#
# The gateway writes gateway.log through an O_APPEND stdout redirection, so
# rotation must copy-then-truncate: moving the file would leave the process
# writing into an unlinked inode, and restarting it just to reopen a log
# loses the evidence a restart is usually trying to capture. copy+truncate
# keeps the writer untouched and preserves the tail across restarts.
#
# Usage (operator, on the gateway box):
#   bash /home/ec2-user/gateway/rotate_gateway_log.sh
# Cron (hourly check, rotates only past the size threshold):
#   0 * * * * bash /home/ec2-user/gateway/rotate_gateway_log.sh >> /home/ec2-user/gateway/rotate.log 2>&1

set -euo pipefail

LOG_FILE="${GATEWAY_LOG_FILE:-/home/ec2-user/gateway/gateway.log}"
ARCHIVE_DIR="${GATEWAY_LOG_ARCHIVE_DIR:-/home/ec2-user/gateway/log-archive}"
MAX_BYTES="${GATEWAY_LOG_MAX_BYTES:-524288000}"   # rotate past ~500MB
KEEP_ARCHIVES="${GATEWAY_LOG_KEEP_ARCHIVES:-14}"  # newest N compressed archives

if [[ ! -f "$LOG_FILE" ]]; then
    echo "rotate_gateway_log: no log file at $LOG_FILE, nothing to do"
    exit 0
fi

size_bytes=$(stat -c %s "$LOG_FILE" 2>/dev/null || stat -f %z "$LOG_FILE")
if (( size_bytes < MAX_BYTES )); then
    echo "rotate_gateway_log: ${size_bytes} bytes < ${MAX_BYTES}, skipping"
    exit 0
fi

mkdir -p "$ARCHIVE_DIR"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
archive="$ARCHIVE_DIR/gateway.log.$stamp"

# Copy first, truncate second: a failure between the two leaves the original
# intact (double data beats lost data for incident evidence).
cp "$LOG_FILE" "$archive"
truncate -s 0 "$LOG_FILE"
gzip "$archive"
echo "rotate_gateway_log: rotated ${size_bytes} bytes -> ${archive}.gz"

# Prune oldest archives beyond the retention count.
ls -1t "$ARCHIVE_DIR"/gateway.log.*.gz 2>/dev/null | tail -n "+$((KEEP_ARCHIVES + 1))" | while read -r old; do
    rm -f -- "$old"
    echo "rotate_gateway_log: pruned $old"
done

#!/bin/bash
AUTOSYNC_SCRIPT="/data/hyperion/mlops/autosync.sh"
LOG_FILE="/data/hyperion/logs/autosync_loop.log" # Separate log for the loop itself
SLEEP_INTERVAL=300 # 5 minutes (300 seconds)

echo "--- Autosync loop starting at $(date -u +'%Y-%m-%d %H:%M:%S UTC') ---" >> "$LOG_FILE"
while true; do
    echo "[$(date -u +'%Y-%m-%d %H:%M:%S UTC')] Running autosync script..." >> "$LOG_FILE"
    bash "$AUTOSYNC_SCRIPT" >> "$LOG_FILE" 2>&1
    echo "[$(date -u +'%Y-%m-%d %H:%M:%S UTC')] Autosync finished. Sleeping for $SLEEP_INTERVAL seconds..." >> "$LOG_FILE"
    sleep "$SLEEP_INTERVAL"
done

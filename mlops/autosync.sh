#!/bin/bash
LOG_FILE="/data/hyperion/logs/autosync.log"
REPO_DIR="/data/hyperion"
GH_REPO_SPEC="adaptnova/hyperion"
HF_DATA_REPO_SPEC="LevelUp2x/hyperion-data"

echo "--- Autosync started at $(date -u +'%Y-%m-%d %H:%M:%S UTC') ---" >> "$LOG_FILE"
cd "$REPO_DIR" || { echo "ERROR: Cannot cd to $REPO_DIR" >> "$LOG_FILE"; exit 1; }

# --- Part 1: Sync Git Repository (Code, Docs) ---
echo "Syncing Git repository..." >> "$LOG_FILE"
git add . >> "$LOG_FILE" 2>&1
if ! git diff --staged --quiet; then
    git commit -m "Autosync: $(date -u +'%Y-%m-%d %H:%M:%S UTC')" >> "$LOG_FILE" 2>&1
    echo "Changes committed locally." >> "$LOG_FILE"
else
    echo "No local changes to commit." >> "$LOG_FILE"
fi
git push origin main >> "$LOG_FILE" 2>&1
echo "Git push attempt completed." >> "$LOG_FILE"

# --- Part 2: Sync HF Dataset Repository (Logs, Receipts) ---
echo "Syncing HF dataset repository..." >> "$LOG_FILE"
LOG_ARCHIVE="latest_logs.tar.gz"
RECEIPTS_ARCHIVE="latest_receipts.tar.gz"

# Archive logs and receipts
tar -czf "$LOG_ARCHIVE" -C "$REPO_DIR" logs
tar -czf "$RECEIPTS_ARCHIVE" -C "$REPO_DIR" receipts

# Upload logs archive
echo "Uploading logs..." >> "$LOG_FILE"
huggingface-cli upload "$HF_DATA_REPO_SPEC" "$LOG_ARCHIVE" "$LOG_ARCHIVE" --repo-type=dataset --quiet >> "$LOG_FILE" 2>&1
# Upload receipts archive
echo "Uploading receipts..." >> "$LOG_FILE"
huggingface-cli upload "$HF_DATA_REPO_SPEC" "$RECEIPTS_ARCHIVE" "$RECEIPTS_ARCHIVE" --repo-type=dataset --quiet >> "$LOG_FILE" 2>&1

# Clean up local archives
rm -f "$LOG_ARCHIVE" "$RECEIPTS_ARCHIVE"

echo "HF dataset push attempt completed." >> "$LOG_FILE"
echo "--- Autosync finished ---" >> "$LOG_FILE"

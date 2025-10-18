#!/bin/bash
LOG_FILE="/data/hyperion/logs/autosync.log"
REPO_DIR="/data/hyperion"
REPO_SPEC="adaptnova/hyperion" # Explicit repo spec for gh

echo "--- Autosync started at $(date -u +'%Y-%m-%d %H:%M:%S UTC') ---" >> "$LOG_FILE"
cd "$REPO_DIR" || { echo "ERROR: Cannot cd to $REPO_DIR" >> "$LOG_FILE"; exit 1; }

# Add changes, commit with timestamp
git add . >> "$LOG_FILE" 2>&1
# Check if there are changes to commit
if ! git diff --staged --quiet; then
    git commit -m "Autosync: $(date -u +'%Y-%m-%d %H:%M:%S UTC')" >> "$LOG_FILE" 2>&1
    echo "Changes committed locally." >> "$LOG_FILE"
else
    echo "No local changes to commit." >> "$LOG_FILE"
fi

# Sync using gh cli (pull first, then push)
echo "Attempting sync with remote..." >> "$LOG_FILE"
gh repo sync "$REPO_SPEC" --branch main >> "$LOG_FILE" 2>&1
echo "Sync attempt completed." >> "$LOG_FILE"
echo "--- Autosync finished ---" >> "$LOG_FILE"

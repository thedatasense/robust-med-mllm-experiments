#!/bin/bash

# Path to the log file
LOG_FILE="/NAS/bsada1/coderepo/CARES/data/Harvard-FairVLMed/fundus_factuality_proc_unc_ans.jsonl"

# Start run_llava.py in the background
nohup python run_llava.py > llava_run.log 2>&1 &
RUN_PID=$!
echo "Started run_llava.py with PID ${RUN_PID}"

# Function to monitor the JSONL file
monitor_jsonl() {
    echo "Monitoring $LOG_FILE..."
    echo "Last 10 lines:"
    tail -n 10 "$LOG_FILE"
    echo -e "\nWatching for new changes (press Ctrl+C to stop)..."
    tail -f "$LOG_FILE"
}

# Wait a moment for the process to start
sleep 2

# Monitor the JSONL file
monitor_jsonl

# Note: The script will continue to show new lines until interrupted with Ctrl+C
# The run_llava.py process will continue running in the background
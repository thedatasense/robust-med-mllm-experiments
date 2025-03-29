#!/bin/bash

# Default values
DEFAULT_INPUT="/NAS/bsada1/coderepo/CARES/data/HAM10000/HAM10000_factuality_pro-jb-1.jsonl"
DEFAULT_OUTPUT="/NAS/bsada1/coderepo/CARES/data/HAM10000/HAM10000_factuality_pro-jb-1_ans.jsonl"

# Help function
print_usage() {
    echo "Usage: $0 [-i INPUT_FILE] [-o OUTPUT_FILE]"
    echo "  -i: Input JSONL file path (default: $DEFAULT_INPUT)"
    echo "  -o: Output JSONL file path (default: $DEFAULT_OUTPUT)"
    exit 1
}

# Parse command line arguments
INPUT_FILE="$DEFAULT_INPUT"
OUTPUT_FILE="$DEFAULT_OUTPUT"

while getopts "i:o:h" opt; do
    case $opt in
        i) INPUT_FILE="$OPTARG";;
        o) OUTPUT_FILE="$OPTARG";;
        h) print_usage;;
        ?) print_usage;;
    esac
done

# Start run_llava.py in the background with parameters
nohup python run_llava.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" > llava_run.log 2>&1 &
RUN_PID=$!
echo "Started run_llava.py with PID ${RUN_PID}"

# Function to monitor the output file
monitor_jsonl() {
    local output_file=$1
    echo "Monitoring $output_file..."
    echo "Last 10 lines:"
    tail -n 10 "$output_file" 2>/dev/null || echo "Waiting for output file to be created..."
    echo -e "\nWatching for new changes (press Ctrl+C to stop)..."
    tail -f "$output_file"
}

# Wait a moment for the process to start
sleep 2

# Monitor the output file
monitor_jsonl "$OUTPUT_FILE"

# Save PID for later use
echo "$RUN_PID" > llava_run.pid
echo "PID saved to llava_run.pid"
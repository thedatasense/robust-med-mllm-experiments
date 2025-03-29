#!/bin/bash

# Array of input files
INPUT_FILES=(
    "/NAS/bsada1/coderepo/CARES/data/MIMIC-CXR/mimic_factuality_proc_jb-1.jsonl"
    "/NAS/bsada1/coderepo/CARES/data/MIMIC-CXR/mimic_factuality_proc_jb-2.jsonl"
    "/NAS/bsada1/coderepo/CARES/data/MIMIC-CXR/mimic_factuality_proc_jb-3.jsonl"
    "/NAS/bsada1/coderepo/CARES/data/MIMIC-CXR/mimic_factuality_proc_oc-3.jsonl"
    "/NAS/bsada1/coderepo/CARES/data/MIMIC-CXR/mimic_factuality_proc_tox.jsonl"
    "/NAS/bsada1/coderepo/CARES/data/MIMIC-CXR/mimic_factuality_proc_priv.jsonl"
    "/NAS/bsada1/coderepo/CARES/data/MIMIC-CXR/mimic_factuality_proc_rob.jsonl"
    "/NAS/bsada1/coderepo/CARES/data/MIMIC-CXR/mimic_factuality_proc_unc.jsonl"
)

# Remove old PID file if exists
rm -f llava_runs.pid

# Start all processes in parallel
for input_file in "${INPUT_FILES[@]}"; do
    output_file="${input_file%.jsonl}_ans.jsonl"
    bash monitor_llava-v2.sh -i "$input_file" -o "$output_file" &
done

# Wait a moment for all processes to start
sleep 5

# Monitor all output files
echo "All processes started. Monitoring all output files:"
while true; do
    clear
    echo "=== Status at $(date) ==="
    for input_file in "${INPUT_FILES[@]}"; do
        output_file="${input_file%.jsonl}_ans.jsonl"
        echo -e "\nLast few lines of $(basename "$output_file"):"
        tail -n 5 "$output_file" 2>/dev/null || echo "Waiting for file..."
    done
    sleep 10
done
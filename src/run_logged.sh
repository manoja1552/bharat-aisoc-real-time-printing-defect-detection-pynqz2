#!/bin/bash

LOG="$HOME/printing_defect_project/project_execution_log.txt"

{
    echo
    echo "=================================================="
    echo "PROJECT EXECUTION REPORT"
    echo "=================================================="
    echo "Date: $(date)"
    echo "Directory: $(pwd)"
    echo "Command: $*"
    echo "Board: $(uname -a)"
    echo

    echo "---------- DETECTOR PARAMETERS ----------"
    python3 - <<'PY'
from defect_detector import Params
from dataclasses import asdict, is_dataclass

p = Params()

if is_dataclass(p):
    params = asdict(p)
else:
    params = vars(p)

for key, value in params.items():
    print(f"{key}: {value}")
PY

    echo
    echo "---------- EXECUTION OUTPUT ----------"
} >> "$LOG" 2>&1

"$@" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}

{
    echo
    echo "Exit status: $STATUS"
    echo "Execution completed: $(date)"
    echo "=================================================="
} >> "$LOG"

echo "Report updated: $LOG"
exit "$STATUS"

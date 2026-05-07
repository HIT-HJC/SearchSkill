#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/teacher_trajectory}"
OUT_ROOT="$ROOT/runs/coverage_supplement/teacher_run_2way"
mkdir -p "$OUT_ROOT"

cd "$ROOT"
setsid srun --jobid=1313825 --overlap --ntasks=1 \
  bash -lc "cd '$ROOT' && WAIT_FOR_SHARDS=1 bash bin/run_coverage_supplement_teacher_2way.sh" \
  > "$OUT_ROOT/master_srun.log" 2>&1 < /dev/null &

echo "$!" > "$OUT_ROOT/master_srun.pid"
echo "started 2way master pid=$(cat "$OUT_ROOT/master_srun.pid") log=$OUT_ROOT/master_srun.log"

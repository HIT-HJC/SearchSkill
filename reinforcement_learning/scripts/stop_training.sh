#!/usr/bin/env bash
set +e
${RAY_BIN:-/path/to/conda/env/bin/ray} stop --force >/tmp/searchskill_ray_stop_final.log 2>&1
pkill -9 -f main_ppo_searchskill
pkill -9 -f ray::
exit 0

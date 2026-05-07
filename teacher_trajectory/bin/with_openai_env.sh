#!/usr/bin/env bash
set -euo pipefail

source "${SEARCHSKILL_ROOT:-/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code}"/config/.openai_searchskill_env
exec "$@"

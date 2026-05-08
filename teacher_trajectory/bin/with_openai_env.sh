#!/usr/bin/env bash
set -euo pipefail

source "${SEARCHSKILL_ROOT:-/path/to/SearchSkill Code}"/config/.openai_searchskill_env
exec "$@"

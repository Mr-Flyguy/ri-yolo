#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Master runner for PIERE-2026 conference revision experiments.

set -e

# Navigate to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "================================================================================"
echo "  LAUNCHING PIERE-2026 REVISION SUITE (Mr-Flyguy/ri-yolo)"
echo "  Repository root: ${REPO_ROOT}"
echo "================================================================================"

# Execute Python orchestrator with all passed arguments
python "${SCRIPT_DIR}/run_all.py" "$@"

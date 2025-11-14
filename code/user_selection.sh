#!/usr/bin/env bash
set -euo pipefail

# 用法示例：
#   ./run_selection.sh select --input-csv MARS/data/output/2025-06-14/user_profiles.csv --output-file selected.txt gender=female min_followers_count=1000
#   ./run_selection.sh collect --user-list selected.txt --input-root MARS/data/output --out-dir MARS/code/user_selection/subset
#   ./run_selection.sh convert_relationships --input-dir MARS/data/output --pattern "attention_matrix*.csv" --out-dir MARS/code/user_selection/subset --quantile 0.25
#   ./run_selection.sh convert_posts --source-db path/to/source.db --oasis-db data/oasis/oasis_database_subset.db --calibration-end 2025-06-02T16:30:00 --ground-truth-end 2025-06-02T16:45:00
#   ./run_selection.sh all --input-csv MARS/data/output/2025-06-14/user_profiles.csv --output-file selected.txt --calibration-end 2025-06-02T16:30:00 --ground-truth-end 2025-06-02T16:45:00

MODE="${1:-help}"; shift || true

PROJECT_ROOT="/remote-home/JuelinW/oasis_project"
CONDA_ACTIVATE="/remote-home/JuelinW/anaconda3/bin/activate"
CONDA_ENV="oasis"

cd "$PROJECT_ROOT"

# 尝试激活 conda（若不可用会忽略）
if [ -f "$CONDA_ACTIVATE" ]; then
  source "$CONDA_ACTIVATE" "$CONDA_ENV" 2>/dev/null || true
fi

PYTHON="$(command -v python || echo python)"

# 支持通过环境变量预设两个时间（ISO 格式），例如：
#   export CALIBRATION_END="2025-06-02T16:30:00"
#   export GROUND_TRUTH_END="2025-06-02T16:45:00"
# 若设置了，则会自动追加到命令行参数中（仅在包含 convert_posts 或 all 时有意义）
CAL_END="${CALIBRATION_END:-}"
GT_END="${GROUND_TRUTH_END:-}"

# 将剩余参数传递给模块
ARGS=( "$MODE" "$@" )

# 若环境变量提供，则追加对应参数（避免重复）
if [ -n "$CAL_END" ]; then
  skip=false
  for a in "${ARGS[@]}"; do
    if [ "$a" = "--calibration-end" ]; then skip=true; break; fi
  done
  if [ "$skip" = false ]; then
    ARGS+=( --calibration-end "$CAL_END" )
  fi
fi

if [ -n "$GT_END" ]; then
  skip=false
  for a in "${ARGS[@]}"; do
    if [ "$a" = "--ground-truth-end" ]; then skip=true; break; fi
  done
  if [ "$skip" = false ]; then
    ARGS+=( --ground-truth-end "$GT_END" )
  fi
fi

echo "运行: python -m MARS.code.user_selection.selection_process ${ARGS[*]}"
exec "$PYTHON" -m MARS.code.user_selection.selection_process "${ARGS[@]}"
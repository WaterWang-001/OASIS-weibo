#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   ./run_simulation.sh [mode] [seed_size] [api_key]
# 示例:
#   ./run_simulation.sh all 3000 "sk-xxxxx"
MODE="${1:-all}"
SEED_SIZE="${2:-3000}"
API_KEY="${3:-}"

PROJECT_ROOT="/remote-home/JuelinW/oasis_project"
CONDA_ACTIVATE="/remote-home/JuelinW/anaconda3/bin/activate"
CONDA_ENV="oasis"

echo "模式: $MODE"
echo "seed_size: $SEED_SIZE"
if [ -n "$API_KEY" ]; then
  echo "api_key: *** (已提供)"
else
  echo "api_key: (未提供)"
fi

# 进入项目根
cd "$PROJECT_ROOT"

# 尝试激活 conda 环境（可选）
if [ -f "$CONDA_ACTIVATE" ]; then
  # 如果 activate 脚本支持直接传 env 名称
  # 若你的 conda 安装不同，请手动在终端激活环境后再运行脚本
  source "$CONDA_ACTIVATE" "$CONDA_ENV" || true
fi

PYTHON="$(command -v python || echo python)"

# 构建命令
CMD=( "$PYTHON" -m MARS.simulation.run_simulation "$MODE" --seed-size "$SEED_SIZE" )
if [ -n "$API_KEY" ]; then
  CMD+=( --api-key "$API_KEY" )
fi

echo "运行命令: ${CMD[*]}"
"${CMD[@]}"
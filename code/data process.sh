# 用法: ./data_process.sh [YYYY-MM-DD]
DATE="${1:-2025-06-14}"

PROJECT_ROOT="/remote-home/JuelinW/oasis_project"
RAW_DIR="$PROJECT_ROOT/data/raw/$DATE"
OUT_DIR="$PROJECT_ROOT/MARS/data_process/output/$DATE"

# 激活 conda env 的 activate 脚本与环境名（按需修改）
CONDA_ACTIVATE="/remote-home/JuelinW/anaconda3/bin/activate"
CONDA_ENV="oasis"

echo "日期: $DATE"
echo "输入目录: $RAW_DIR"
echo "输出目录: $OUT_DIR"

# 检查输入目录
if [ ! -d "$RAW_DIR" ]; then
  echo "错误: 输入目录不存在: $RAW_DIR" >&2
  exit 1
fi

# 创建输出目录
mkdir -p "$OUT_DIR"

# 激活 conda 环境
if [ -f "$CONDA_ACTIVATE" ]; then
  # 使用 conda 的 activate 脚本
  # 注意：source /path/to/activate <env>
  source "$CONDA_ACTIVATE" "$CONDA_ENV"
else
  echo "警告: 未找到 conda activate 脚本：$CONDA_ACTIVATE 。请手动激活环境后运行脚本。" >&2
fi

# 进入项目根并运行 pipeline（解压已由 Python 脚本处理或用户预先处理）
cd "$PROJECT_ROOT"
echo "运行 pipeline: python MARS/data_process/data_process.py all --input \"$RAW_DIR\" --out \"$OUT_DIR\""
python MARS/data_process.py all --input "$RAW_DIR" --out "$OUT_DIR"

echo "全部完成。输出位于: $OUT_DIR"
```# filepath: /remote-home/JuelinW/oasis_project/MARS/data_process.sh
#!/usr/bin/env bash
set -euo pipefail

# 用法: ./MARS/data process.sh [YYYY-MM-DD]

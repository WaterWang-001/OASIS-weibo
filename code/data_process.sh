#!/bin/bash
# 用法: ./process_dates_parallel.sh [YYYY-MM-DD1] [YYYY-MM-DD2] [YYYY-MM-DD3] ...
#
# 此脚本会为每一个传入的日期参数启动一个并行的 data_process.py 任务。
# 它会通过 MAX_JOBS 变量控制最大并行数。

# --- 配置 ---

# 设置最大并行任务数（根据你的服务器CPU核心数和内存调整）
MAX_JOBS=4

# 项目和 Conda 路径 (与原脚本一致)
PROJECT_ROOT="/remote-home/JuelinW/oasis_project"
CONDA_ACTIVATE="/remote-home/JuelinW/anaconda3/bin/activate"
CONDA_ENV="oasis"

# --- 脚本主体 ---

# 检查是否至少提供了一个日期
if [ $# -eq 0 ]; then
  echo "错误: 至少需要提供一个日期参数。" >&2
  echo "用法: $0 [YYYY-MM-DD1] [YYYY-MM-DD2] ..." >&2
  exit 1
fi

# 1. 定义一个函数，用于处理 *单个* 日期
#    这个函数将在 xargs 启动的子 shell 中被调用。
process_one_date() {
    local DATE="$1"
    
    # 从环境变量中读取全局配置 (我们稍后会 export 它们)
    local RAW_DIR="$PROJECT_ROOT/data/raw/$DATE"
    local OUT_DIR="$PROJECT_ROOT/MARS/data/output/$DATE"

    # 为日志添加前缀，方便区分不同日期的输出
    local LOG_PREFIX="[$DATE | PID: $$]"

    echo "$LOG_PREFIX 开始处理..."
    echo "$LOG_PREFIX 输入: $RAW_DIR"
    echo "$LOG_PREFIX 输出: $OUT_DIR"

    # 检查输入目录
    if [ ! -d "$RAW_DIR" ]; then
        echo "$LOG_PREFIX 错误: 输入目录不存在: $RAW_DIR" >&2
        return 1 # 函数返回非 0 值，xargs 会认为此任务失败
    fi

    # 创建输出目录
    mkdir -p "$OUT_DIR"

    # 激活 conda 环境
    # (每个并行任务都需要独立激活)
    if [ -f "$CONDA_ACTIVATE" ]; then
        # 注意：source /path/to/activate <env>
        source "$CONDA_ACTIVATE" "$CONDA_ENV"
    else
        echo "$LOG_PREFIX 警告: 未找到 conda activate 脚本。依赖外部激活的环境。" >&2
    fi

    # 进入项目根
    cd "$PROJECT_ROOT"

    # 运行 pipeline
    echo "$LOG_PREFIX 运行: python MARS/code/data_process/data_process.py all --input \"$RAW_DIR\" --out \"$OUT_DIR\""
    
    python MARS/code/data_process/data_process.py all --input "$RAW_DIR" --out "$OUT_DIR"
    
    local exit_code=$? # 获取 Python 脚本的退出码
    
    if [ $exit_code -eq 0 ]; then
        echo "$LOG_PREFIX 成功完成。输出位于: $OUT_DIR"
    else
        echo "$LOG_PREFIX 失败 (退出码: $exit_code)。"
    fi
    
    return $exit_code
}

# 2. 导出函数和变量
#    为了让 xargs 启动的 bash 子进程能访问到它们
export -f process_one_date
export PROJECT_ROOT
export CONDA_ACTIVATE
export CONDA_ENV

# 3. 使用 printf 和 xargs 运行
echo "准备处理 $# 个日期，最大并行数: $MAX_JOBS"
echo "日期列表: $@"
echo "---"

# printf '%s\n' "$@" 会将所有日期参数（"$@"）
#   每个一行 ( %s\n ) 打印到标准输出。
#
# xargs -I {} -P $MAX_JOBS ...
#   -P $MAX_JOBS: 设置最大并行进程数为 $MAX_JOBS。
#   -I {}:       从标准输入（来自printf）
#                一次读取一行，并将该行内容替换 {}。
#   bash -c '...': 启动一个 bash 实例来执行
#                引号内的命令。
#   'process_one_date "{}"': 在这个 bash 实例中，
#                调用我们之前导出的函数，
#                并将日期（来自 {}）作为参数传递。

printf '%s\n' "$@" | xargs -I {} -P $MAX_JOBS bash -c 'process_one_date "{}"'

# 4. 检查 xargs 的最终退出码
#    如果任何一个 process_one_date 任务返回失败
#    (非0退出码)，xargs 最终也会返回一个非0退出码。
if [ $? -eq 0 ]; then
    echo "---"
    echo "所有日期的任务均已成功启动和完成。"
else
    echo "---"
    echo "警告：一个或多个日期的处理失败，请检查上面的日志。"
    exit 1
fi

echo "全部并行处理完成。"
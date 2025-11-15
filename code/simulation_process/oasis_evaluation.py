import sqlite3
import os
import sys
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive saving
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # For custom legend
# --- (移除 Annotator) ---


# --- 1. Configuration (已修改) ---

# --- Database and Time Configuration ---
OASIS_DB_PATH = 'data/oasis/oasis_database_3000_random.db' # 你的数据库路径
# --- [!! 恢复: GT (真实数据) 需要这些 !!] ---
GROUND_TRUTH_START_TIME_STR = "2025-06-02 16:30:00" # T=0 时刻对应的真实时间
TIME_STEP_MINUTES = 3 # 每个时间步的分钟数
# --- [!! 恢复结束 !!] ---

# --- Chart Output Paths (将自动添加 _llm.png 和 _abm.png) ---
CHART_OUTPUT_PATH = 'data/oasis/attitude_timeseries_chart.png'
STATIC_METRICS_CHART_OUTPUT_PATH = 'data/oasis/attitude_static_metrics_chart.png'
SUBPLOTS_TIMESERIES_CHART_OUTPUT_PATH = 'data/oasis/attitude_timeseries_subplots.png'
ALL_METRICS_TIMESERIES_CHART_OUTPUT_PATH = 'data/oasis/attitude_all_metrics_timeseries_chart.png'

# --- Metrics Configuration ---
ATTITUDE_COLUMNS = [
    'attitude_lifestyle_culture',
    'attitude_sport_ent',
    'attitude_sci_health',
    'attitude_politics_econ'
]
ALL_ATTITUDE_DIMS = ATTITUDE_COLUMNS + ['attitude_average']


# --- 3. Evaluation Module (重写) ---

def process_log_group(df_group: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    辅助函数: 计算一个日志数据组 (Sim) 的 Bias 和 Div。
    (此函数不变, 仅用于 Sim)
    """
    if df_group.empty:
        return pd.DataFrame()

    # 1. 计算 Bias (Mean)
    df_bias = df_group.groupby(['time_step', 'dimension'])['attitude_score'].mean()
    # 2. 计算 Div (Std Dev)
    df_div = df_group.groupby(['time_step', 'dimension'])['attitude_score'].std()

    # 3. 转换
    df_bias = df_bias.unstack(level='dimension')
    df_div = df_div.unstack(level='dimension')
    
    # 4. 重命名列 (e.g., 'log_attitude_lifestyle_culture_bias_sim')
    df_bias.columns = [f"log_{col}_bias{suffix}" for col in df_bias.columns]
    df_div.columns = [f"log_{col}_div{suffix}" for col in df_div.columns]
    
    # 5. 合并
    df_metrics = pd.concat([df_bias, df_div], axis=1)
    
    return df_metrics


def get_time_aligned_data(
    db_path: str, 
    start_time_dt: datetime, 
    step_minutes: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    [重写]
    1. 从 5 个日志表中提取 'LLM' 和 'ABM' 的 Sim 数据 (T<0 和 T>=0)。
    2. 从 'ground_truth_post' 表中提取 GT 数据 (T>=0)。
    3. 返回三个独立的 metrics DataFrames: (llm_metrics, abm_metrics, gt_metrics)。
    """
    logging.info("  -> 提取和对齐所有数据...")
    
    llm_metrics_df = pd.DataFrame()
    abm_metrics_df = pd.DataFrame()
    gt_metrics_df = pd.DataFrame()
    
    conn = None
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        
        # --- 1. 构建 Sim (LLM 和 ABM) 数据 ---
        union_queries = []
        for dim_name in ALL_ATTITUDE_DIMS:
            table_name = f"log_{dim_name}"
            union_queries.append(f"""
            SELECT 
                time_step, agent_type, metric_type, attitude_score,
                '{dim_name}' AS dimension
            FROM {table_name}
            """)
        
        full_query = "\nUNION ALL\n".join(union_queries)
        all_logs_df = pd.read_sql_query(full_query, conn)

        # 1a. 处理 LLM (Sim)
        llm_df_raw = all_logs_df[all_logs_df['agent_type'] == 'LLM']
        if llm_df_raw.empty:
            logging.warning("  -> 未找到 LLM (Sim) 日志数据。")
        else:
            llm_metrics_df = process_log_group(llm_df_raw, '_sim')
            # (添加主图表所需的 'avg_attitude_sim' 列)
            if 'log_attitude_average_bias_sim' in llm_metrics_df.columns:
                llm_metrics_df['avg_attitude_sim'] = llm_metrics_df['log_attitude_average_bias_sim']

        # 1b. 处理 ABM (Sim)
        abm_df_raw = all_logs_df[all_logs_df['agent_type'] == 'ABM']
        if abm_df_raw.empty:
            logging.warning("  -> 未找到 ABM (Sim) 日志数据。")
        else:
            abm_metrics_df = process_log_group(abm_df_raw, '_sim')
            # (添加主图表所需的 'avg_attitude_sim' 列)
            if 'log_attitude_average_bias_sim' in abm_metrics_df.columns:
                abm_metrics_df['avg_attitude_sim'] = abm_metrics_df['log_attitude_average_bias_sim']

        # --- 2. 构建 Ground Truth (GT) 数据 (来自 ground_truth_post) ---
        gt_query = f"""
        SELECT created_at, {', '.join(ATTITUDE_COLUMNS)}
        FROM ground_truth_post
        WHERE attitude_annotated = 1 AND created_at >= ?
        """
        gt_df = pd.read_sql_query(
            gt_query, 
            conn, 
            params=(start_time_dt.strftime('%Y-%m-%d %H:%M:%S'),)
        )
        
        if gt_df.empty:
            logging.error("  -> ❌ 错误: 未在 'ground_truth_post' 表中找到 T>=0 的已标注数据。")
        else:
            gt_df['created_at_dt'] = pd.to_datetime(gt_df['created_at'])
            
            # (计算 T>=0 的时间步)
            time_delta_seconds = (gt_df['created_at_dt'] - start_time_dt).dt.total_seconds()
            gt_df['time_step'] = (time_delta_seconds // (step_minutes * 60)).astype(int)
            
            # (计算 Bias 和 Div)
            gt_metrics_list = []
            for step, group in gt_df.groupby('time_step'):
                metrics = {'time_step': step}
                bias_values_for_this_step = []
                for col in ATTITUDE_COLUMNS:
                    attitudes = group[col].values
                    bias = np.mean(attitudes)
                    metrics[f'log_{col}_bias_gt'] = bias # (使用 'log_' 前缀以匹配 Sim)
                    bias_values_for_this_step.append(bias)
                    metrics[f'log_{col}_div_gt'] = np.std(attitudes)
                
                # (计算总平均值)
                avg_bias = np.mean(bias_values_for_this_step)
                metrics[f'log_attitude_average_bias_gt'] = avg_bias
                metrics['avg_attitude_gt'] = avg_bias # (添加主图表所需的列)
                
                gt_metrics_list.append(metrics)
                
            gt_metrics_df = pd.DataFrame(gt_metrics_list).set_index('time_step')
        
        logging.info(f"  -> 数据提取完成。")
        return llm_metrics_df, abm_metrics_df, gt_metrics_df

    finally:
        if conn:
            conn.close()
            logging.info("  -> (评估 DB 连接已关闭)")

# --- (DTW 函数保持不变) ---
def dtw(s, t, keep_internals=False):
    import numpy as _np
    s = _np.asarray(s, dtype=float)
    t = _np.asarray(t, dtype=float)
    n, m = len(s), len(t)

    class _DTWResult:
        def __init__(self, distance, normalizedDistance, path):
            self.distance = distance
            self.normalizedDistance = normalizedDistance
            self.path = path

    if n == 0 or m == 0:
        return _DTWResult(float('inf'), float('inf'), [])

    # DP matrix (n+1 x m+1) initialized with infinities
    dtw_matrix = _np.full((n + 1, m + 1), _np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s[i - 1] - t[j - 1])
            last_min = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
            dtw_matrix[i, j] = cost + last_min

    distance = float(dtw_matrix[n, m])

    # Reconstruct warping path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = [(i - 1, j), (i, j - 1), (i - 1, j - 1)]
        costs = [dtw_matrix[a, b] for a, b in choices]
        idx = int(_np.argmin(costs))
        if idx == 0:
            i -= 1
        elif idx == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()

    normalizedDistance = distance / max(1, len(path))
    return _DTWResult(distance, normalizedDistance, path if keep_internals else [])

# --- (run_evaluation 函数保持不变) ---
def run_evaluation(aligned_df: pd.DataFrame, agent_type: str):
    """
    [修改] 计算指标。 (现在接受 agent_type 用于日志记录)
    """
    if aligned_df.empty:
        logging.warning(f"  -> [Type: {agent_type}] 评估数据为空, 跳过计算。")
        return
        
    print("\n" + "="*50)
    print(f"--- 评估结果 ({agent_type} Agents) ---")
    print("="*50)

    # --- 1. 静态分布 (ΔBias, ΔDiv) ---
    print("\n--- 1. 静态分布 (ΔBias, ΔDiv) ---")
    print(" (越低越好)")
    
    delta_bias_scores = {}
    delta_div_scores = {}

    # [修改] 我们现在循环 5 个维度
    for dim_name in ALL_ATTITUDE_DIMS:
        # [修改] 构造新的列名
        sim_bias_col = f'log_{dim_name}_bias_sim'
        gt_bias_col = f'log_{dim_name}_bias_gt'
        sim_div_col = f'log_{dim_name}_div_sim'
        gt_div_col = f'log_{dim_name}_div_gt'

        # 检查列是否存在, 以防万一 (例如, 如果 'attitude_average' 失败)
        if not all(c in aligned_df.columns for c in [sim_bias_col, gt_bias_col, sim_div_col, gt_div_col]):
            logging.warning(f"  -> [Type: {agent_type}] 缺少 '{dim_name}' 的指标列, 跳过。")
            continue
            
        sim_bias = aligned_df[sim_bias_col]
        gt_bias = aligned_df[gt_bias_col]
        sim_div = aligned_df[sim_div_col]
        gt_div = aligned_df[gt_div_col]
        
        # (计算平均绝对差异)
        delta_bias = (sim_bias - gt_bias).abs().mean()
        delta_div = (sim_div - gt_div).abs().mean()
        
        delta_bias_scores[dim_name] = delta_bias
        delta_div_scores[dim_name] = delta_div
        
        # (使用 .title() 美化显示名称)
        display_name = dim_name.replace('attitude_', '').replace('_', ' ').title()
        print(f"\n  {display_name}:")
        print(f"    ΔBias (差异): {delta_bias:.4f}")
        print(f"    ΔDiv (差异):  {delta_div:.4f}")

    # (移除 'overall_average' 的硬编码计算, 因为它已在循环中)
    
    # [修改] 静态指标图现在使用 5 个维度
    save_static_metrics_chart(
        delta_bias_scores, 
        delta_div_scores, 
        STATIC_METRICS_CHART_OUTPUT_PATH.replace('.png', f'_{agent_type.lower()}.png'),
        agent_type
    )

    # --- 2. 时间序列相似性 (DTW, Pearson) ---
    print("\n--- 2. 时间序列相似性 (所有维度) ---")
    
    # [修改] 循环 5 个维度
    display_names_map = {
        **{col: col.replace('attitude_', '').replace('_', ' ').title() for col in ATTITUDE_COLUMNS},
        'attitude_average': 'Overall Average' # 修正键名
    }

    for dim_name in ALL_ATTITUDE_DIMS:
        display_name = display_names_map[dim_name]
        
        # [修改] 构造新的列名
        sim_col = f'log_{dim_name}_bias_sim'
        gt_col = f'log_{dim_name}_bias_gt'
        
        if not all(c in aligned_df.columns for c in [sim_col, gt_col]):
            logging.warning(f"  -> [Type: {agent_type}] 缺少 '{dim_name}' 的时间序列列, 跳过。")
            continue
        
        print(f"\n  --- 相似性: {display_name} ---")

        sim_ts = aligned_df[sim_col].values
        gt_ts = aligned_df[gt_col].values
        
        if len(sim_ts) < 3:
            logging.warning("  -> 时间序列太短 (少于 3 步), 无法计算 DTW 或 Pearson。")
            continue

        # A. Pearson 相关系数 (越高越好)
        try:
            pearson_corr, p_value = pearsonr(sim_ts, gt_ts)
            print(f"    Pearson 相关性 (越高越好):")
            print(f"      r = {pearson_corr:.4f} (p-value = {p_value:.3f})")
        except ValueError as e:
            print(f"    Pearson 计算失败: {e}")

        # B. 动态时间规整 (DTW) (越低越好)
        try:
            sim_ts_norm = (sim_ts - np.mean(sim_ts)) / (np.std(sim_ts) + 1e-9)
            gt_ts_norm = (gt_ts - np.mean(gt_ts)) / (np.std(gt_ts) + 1e-9)
            
            dtw_result = dtw(sim_ts_norm, gt_ts_norm, keep_internals=True)
            print(f"    DTW 距离 (越低越好):")
            print(f"      Normalized Distance = {dtw_result.normalizedDistance:.4f}")
        except Exception as e:
            print(f"    DTW 计算失败: {e}")

# --- (绘图函数) ---

# --- (save_static_metrics_chart 保持不变) ---
def save_static_metrics_chart(
    delta_bias_scores: Dict[str, float], 
    delta_div_scores: Dict[str, float], 
    output_path: str,
    agent_type: str # <--- 新增
):
    """
    [修改] 绘制 ΔBias 和 ΔDiv 的条形图 (现在包含 5 个维度)
    """
    logging.info(f"  -> [Type: {agent_type}] 生成静态指标对比图...")

    # [修改] 准备 5 个维度的标签和数据
    labels = [dim.replace('attitude_', '').replace('_', ' ').title() for dim in ALL_ATTITUDE_DIMS]
    bias_values = [delta_bias_scores.get(dim, 0.0) for dim in ALL_ATTITUDE_DIMS]
    div_values = [delta_div_scores.get(dim, 0.0) for dim in ALL_ATTITUDE_DIMS]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, bias_values, width, label='ΔBias (Mean 差异)', color='skyblue')
    rects2 = ax.bar(x + width/2, div_values, width, label='ΔDiv (Std Dev 差异)', color='lightcoral')

    ax.set_xlabel('态度维度', fontsize=12)
    ax.set_ylabel('绝对差异 (Sim vs GT)', fontsize=12)
    ax.set_title(f'静态分布指标: Simulation vs. Ground Truth\n(Agent Type: {agent_type})', fontsize=16) # <--- 新增
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.7) # (改为 0.7)

    # (autolabel 函数保持不变)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logging.info(f"  -> [Type: {agent_type}] ✅ 静态指标图已保存: {output_path}")

# --- (save_timeseries_chart 保持不变) ---
def save_timeseries_chart(
    aligned_df: pd.DataFrame, 
    time_step_minutes: int, # <--- [修改] 签名变化
    output_path: str, 
    agent_type: str        # <--- [修改] 签名变化
):
    """
    [修改] 仅绘制 'attitude_average' (总平均) 的时间序列图。
    """
    logging.info(f"  -> [Type: {agent_type}] 生成主时间序列图 (总平均)...")
    
    if aligned_df.empty:
        logging.warning(f"  -> [Type: {agent_type}] 评估数据为空, 跳过主图表。")
        return
    if len(aligned_df) < 2:
        logging.warning(f"  -> [Type: {agent_type}] 时间序列太短, 跳过主图表。")
        return
    # [修改] 检查 'avg_attitude_sim/gt' (由 get_time_aligned_data 创建)
    if 'avg_attitude_sim' not in aligned_df.columns or 'avg_attitude_gt' not in aligned_df.columns:
        logging.error(f"  -> [Type: {agent_type}] 缺少 'avg_attitude_sim/gt' 列, 无法绘制主图表。")
        return

    try:
        time_steps = aligned_df.index
        sim_ts = aligned_df['avg_attitude_sim']
        gt_ts = aligned_df['avg_attitude_gt']
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # [!! 关键 !!] GT (T<0) 和 Sim (T>=0) 可能不连续, 分开绘制
        gt_part = aligned_df[aligned_df.index < 0]
        sim_part = aligned_df[aligned_df.index >= 0]
        
        if not gt_part.empty:
            ax.plot(gt_part.index, gt_part['avg_attitude_gt'], label='Ground Truth (T<0)', marker='x', linestyle='-', color='orange')
        if not sim_part.empty:
            ax.plot(sim_part.index, sim_part['avg_attitude_sim'], label=f'Simulation (T>=0, {agent_type})', marker='o', linestyle='--', color='blue')
            # (可选) 绘制 T>=0 时的 GT (如果有的话)
            if not gt_part.empty: # (只有 T<0 有 GT 时才绘制 T>=0 的 GT)
                ax.plot(sim_part.index, sim_part['avg_attitude_gt'], label='Ground Truth (T>=0, if exists)', marker='x', linestyle=':', color='gray', alpha=0.7)

        ax.set_title(f'总平均态度: Simulation vs. Ground Truth\n(Agent Type: {agent_type})', fontsize=16)
        ax.set_xlabel(f'Time Step (T<0 = 历史, T>=0 = 模拟)', fontsize=12) # [修改] 轴标签
        ax.set_ylabel(f'平均态度 (Overall Average)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # (添加 T=0 的垂直线)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Simulation Start (T=0)')
        # (重新排序图例以包含 vline)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, fontsize=10)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logging.info(f"  -> [Type: {agent_type}] ✅ 主图表已保存: {output_path}")

    except Exception as e:
        logging.error(f"  -> [Type: {agent_type}] ❌ 保存主图表失败: {e}")
        traceback.print_exc()
# --- (save_all_dimensions_timeseries_chart 保持不变) ---
def save_all_dimensions_timeseries_chart(
    aligned_df: pd.DataFrame, 
    time_step_minutes: int, 
    output_path: str,
    agent_type: str # <--- 新增
):
    """
    [修改] 绘制所有 5 个维度的 10 条线 (Sim+GT) 在一个图上。
    """
    logging.info(f"  -> [Type: {agent_type}] 生成 10-线 综合图...")

    if aligned_df.empty or len(aligned_df) < 2:
        logging.warning(f"  -> [Type: {agent_type}] 数据不足, 跳过 10-线 图。")
        return
    
    try:
        fig, ax = plt.subplots(figsize=(16, 8))
        time_steps = aligned_df.index

        colors = ['blue', 'green', 'red', 'purple', 'black']
        # [修改] 使用 5 维
        metrics_to_plot = ALL_ATTITUDE_DIMS
        display_names_map = {
            **{col: col.replace('attitude_', '').replace('_', ' ').title() for col in ATTITUDE_COLUMNS},
            'attitude_average': 'Overall Average' # 修正键名
        }
        
        # [修改] 拆分 T<0 (GT) 和 T>=0 (Sim)
        gt_part = aligned_df[aligned_df.index < 0]
        sim_part = aligned_df[aligned_df.index >= 0]
        
        for i, metric in enumerate(metrics_to_plot):
            # [修改] 新的列名
            sim_col = f'log_{metric}_bias_sim'
            gt_col = f'log_{metric}_bias_gt'
            
            if sim_col not in aligned_df.columns or gt_col not in aligned_df.columns:
                continue

            display_name = display_names_map[metric]
            color = colors[i]
            
            # Plot Sim line (dashed, T>=0)
            if not sim_part.empty:
                ax.plot(sim_part.index, sim_part[sim_col], 
                        label=f'Sim - {display_name}', 
                        marker='o', markersize=4, linestyle='--', alpha=0.7, color=color)
            
            # Plot GT line (solid, T<0)
            if not gt_part.empty:
                ax.plot(gt_part.index, gt_part[gt_col], 
                        label=f'GT - {display_name}', 
                        marker='x', markersize=4, linestyle='-', alpha=0.7, color=color)

        ax.set_title(f'综合态度时间序列: Simulation vs. Ground Truth\n(Agent Type: {agent_type})', fontsize=16) # <--- 新增
        ax.set_xlabel(f'Time Step (T<0 = 历史, T>=0 = 模拟)', fontsize=12) # <--- 修改
        ax.set_ylabel('平均态度 (Bias)', fontsize=12)
        
        # (添加 T=0 的垂直线)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Simulation Start (T=0)')
        
        # Place legend outside the plot area
        handles, labels = ax.get_legend_handles_labels()
        # (去重图例, 因为 Sim/GT 可能共享标签)
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logging.info(f"  -> [Type: {agent_type}] ✅ 10-线 图已保存: {output_path}")

    except Exception as e:
        logging.error(f"  -> [Type: {agent_type}] ❌ 保存 10-线 图失败: {e}")
        traceback.print_exc()

# --- (save_subplots_timeseries_chart 保持不变) ---
def save_subplots_timeseries_chart(
    aligned_df: pd.DataFrame, 
    time_step_minutes: int, 
    output_path: str,
    agent_type: str # <--- 新增
):
    """
    [修改] 绘制 5 个维度的 5 个子图。
    """
    logging.info(f"  -> [Type: {agent_type}] 生成 5-子图...")

    if aligned_df.empty or len(aligned_df) < 2:
        logging.warning(f"  -> [Type: {agent_type}] 数据不足, 跳过 5-子图。")
        return

    try:
        # [修改] 使用 5 维
        metrics_to_plot = ALL_ATTITUDE_DIMS
        display_names_map = {
            **{col: col.replace('attitude_', '').replace('_', ' ').title() for col in ATTITUDE_COLUMNS},
            'attitude_average': 'Overall Average' # 修正键名
        }
        
        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(14, 18), sharex=True)
        
        # [修改] 拆分 T<0 (GT) 和 T>=0 (Sim)
        gt_part = aligned_df[aligned_df.index < 0]
        sim_part = aligned_df[aligned_df.index >= 0]
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # [修改] 新的列名
            sim_col = f'log_{metric}_bias_sim'
            gt_col = f'log_{metric}_bias_gt'
            
            if sim_col not in aligned_df.columns or gt_col not in aligned_df.columns:
                continue

            display_name = display_names_map[metric]
            
            # Plot Sim line (dashed, T>=0)
            if not sim_part.empty:
                ax.plot(sim_part.index, sim_part[sim_col], 
                        label='Simulation (Sim, T>=0)', 
                        marker='o', markersize=4, linestyle='--', alpha=0.8, color='blue')
            
            # Plot GT line (solid, T<0)
            if not gt_part.empty:
                ax.plot(gt_part.index, gt_part[gt_col], 
                        label='Ground Truth (GT, T<0)', 
                        marker='x', markersize=4, linestyle='-', alpha=0.8, color='orange')

            ax.set_title(f'时间序列对比: {display_name}', fontsize=14)
            ax.set_ylabel('平均态度', fontsize=10)
            
            # (添加 T=0 的垂直线)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
            
            ax.legend(fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
        
        axes[-1].set_xlabel(f'Time Step (T<0 = 历史, T>=0 = 模拟)', fontsize=12) # <--- 修改
        fig.suptitle(f'态度时间序列: Simulation vs. Ground Truth\n(Agent Type: {agent_type})', fontsize=18, y=1.02) # <--- 新增
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logging.info(f"  -> [Type: {agent_type}] ✅ 5-子图 已保存: {output_path}")

    except Exception as e:
        logging.error(f"  -> [Type: {agent_type}] ❌ 保存 5-子图 失败: {e}")
        traceback.print_exc()

# --- 4. Main Function (重写) ---

def main():
    """
    [重写] Main execution function.
    - 移除了 Annotator。
    - 恢复了时间配置 (用于 GT)。
    - 对 'LLM' 和 'ABM' agents 分别运行评估。
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(OASIS_DB_PATH):
        logging.error(f"❌ 错误: 数据库未找到 {OASIS_DB_PATH}")
        sys.exit(1)

    # --- [!! 恢复: GT (真实数据) 需要 !!] ---
    try:
        start_time_dt = datetime.strptime(GROUND_TRUTH_START_TIME_STR, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        logging.error(f"❌ 错误: 'GROUND_TRUTH_START_TIME_STR' 格式不正确。")
        logging.error("   请使用 'YYYY-MM-DD HH:MM:SS' 格式。")
        sys.exit(1)
    # --- [!! 恢复结束 !!] ---

    # (移除 Annotator 初始化)

    # 辅助函数, 用于生成带后缀的文件路径
    def get_new_path(original_path: str, suffix: str) -> str:
        return original_path.replace('.png', f'_{suffix}.png')

    try:
        # --- 1. 一次性获取所有数据 ---
        (
            llm_metrics_df, 
            abm_metrics_df, 
            gt_metrics_df
        ) = get_time_aligned_data(
            OASIS_DB_PATH, 
            start_time_dt, 
            TIME_STEP_MINUTES
        )

        # --- 2. 评估阶段 1: LLM Agents vs GT ---
        logging.info("="*60)
        logging.info("--- 正在运行评估: LLM Agents (External Expression) vs GT ---")
        logging.info("="*60)
        
        if gt_metrics_df.empty:
            logging.warning("未找到 GT 数据, 跳过 LLM 评估。")
        elif llm_metrics_df.empty:
            logging.warning("未找到 LLM Sim 数据, 跳过 LLM 评估。")
        else:
            # [!! 关键: 合并 LLM 和 GT !!]
            aligned_df_llm = pd.merge(
                llm_metrics_df, 
                gt_metrics_df, 
                on='time_step', 
                how='outer', 
                suffixes=('_sim', '_gt') # Sim (LLM), GT
            ).fillna(0.0).sort_index()

            run_evaluation(aligned_df_llm, agent_type='LLM')
            
            save_timeseries_chart(
                aligned_df_llm, 0,
                get_new_path(CHART_OUTPUT_PATH, 'llm'), 'LLM'
            )
            save_all_dimensions_timeseries_chart(
                aligned_df_llm, 0,
                get_new_path(ALL_METRICS_TIMESERIES_CHART_OUTPUT_PATH, 'llm'), 'LLM'
            )
            save_subplots_timeseries_chart(
                aligned_df_llm, 0,
                get_new_path(SUBPLOTS_TIMESERIES_CHART_OUTPUT_PATH, 'llm'), 'LLM'
            )
        
        # --- 3. 评估阶段 2: ABM Agents vs GT ---
        logging.info("="*60)
        logging.info("--- 正在运行评估: ABM Agents (Internal State) vs GT ---")
        logging.info("="*60)
        
        if gt_metrics_df.empty:
            logging.warning("未找到 GT 数据, 跳过 ABM 评估。")
        elif abm_metrics_df.empty:
            logging.warning("未找到 ABM Sim 数据, 跳过 ABM 评估。")
        else:
            # [!! 关键: 合并 ABM 和 GT !!]
            aligned_df_abm = pd.merge(
                abm_metrics_df, 
                gt_metrics_df, 
                on='time_step', 
                how='outer', 
                suffixes=('_sim', '_gt') # Sim (ABM), GT
            ).fillna(0.0).sort_index()

            run_evaluation(aligned_df_abm, agent_type='ABM')

            save_timeseries_chart(
                aligned_df_abm, 0, 
                get_new_path(CHART_OUTPUT_PATH, 'abm'), 'ABM'
            )
            save_all_dimensions_timeseries_chart(
                aligned_df_abm, 0,
                get_new_path(ALL_METRICS_TIMESERIES_CHART_OUTPUT_PATH, 'abm'), 'ABM'
            )
            save_subplots_timeseries_chart(
                aligned_df_abm, 0,
                get_new_path(SUBPLOTS_TIMESERIES_CHART_OUTPUT_PATH, 'abm'), 'ABM'
            )
        
    except sqlite3.Error as e:
        logging.error(f"❌ 数据库错误: {e}")
    except Exception as e:
        logging.error(f"❌ 发生意外错误: {e}")
        import traceback
        traceback.print_exc()
    
    logging.info("="*60)
    logging.info("--- 评估完成 ---")


if __name__ == "__main__":
    main()
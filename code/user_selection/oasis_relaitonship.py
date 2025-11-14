"""
合并指定目录下的权重 CSV（edges 表），归一化并基于全局分位数生成关注列表。
输出（默认写入 out_dir）：
 - merged_edges.csv           : 合并后的原始权重 (source,target,weight)
 - normalized_edges.csv       : 归一化后权重 (source,target,weight_norm)
 - follow_list.csv            : 每个 user 的关注列表 (user_id, all_followed_user_ids) 格式为 JSON 列表字符串

用法示例：
 python merge_weights_and_compute_follows.py \
    --input-dir /remote-home/JuelinW/oasis_project/MARS/data/output/2025-06-14 \
    --pattern "attention_matrix*.csv" \
    --out-dir /remote-home/JuelinW/oasis_project/MARS/data/output/2025-06-14 \
    --quantile 0.25 --normalize max
"""
from pathlib import Path
import argparse
import sys
import pandas as pd
import json

def find_weight_files(input_dir: Path, pattern: str, recursive: bool):
    if recursive:
        return sorted(input_dir.rglob(pattern))
    else:
        return sorted(input_dir.glob(pattern))

def read_edge_csv(p: Path):
    # 尝试读取，寻找 source,target,weight 三列
    df = pd.read_csv(p, dtype=str)
    # 若列数 >=3，尝试取前三列并重命名
    cols = list(df.columns)
    if set(['source','target','weight']).issubset(set(cols)):
        df = df[['source','target','weight']].copy()
    elif len(cols) >= 3:
        df = df.iloc[:, :3].copy()
        df.columns = ['source','target','weight']
    else:
        raise ValueError(f"{p} 列不足，无法解析为 edge CSV: columns={cols}")
    # 确保类型
    df['source'] = df['source'].astype(str).str.strip()
    df['target'] = df['target'].astype(str).str.strip()
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(0.0)
    return df

def merge_edges(files):
    dfs = []
    for f in files:
        try:
            df = read_edge_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] 读取 {f} 失败: {e}", file=sys.stderr)
    if not dfs:
        return pd.DataFrame(columns=['source','target','weight'])
    all_df = pd.concat(dfs, ignore_index=True)
    merged = all_df.groupby(['source','target'], as_index=False)['weight'].sum()
    return merged

def normalize_edges(edges_df: pd.DataFrame, method: str = 'max'):
    if edges_df.empty:
        return edges_df.assign(weight_norm=pd.Series(dtype=float))
    if method == 'max':
        # 按 source 的最大值归一化
        max_per_src = edges_df.groupby('source')['weight'].transform('max')
        edges_df['weight_norm'] = edges_df['weight'] / max_per_src.replace({0:1})
    elif method == 'sum':
        sum_per_src = edges_df.groupby('source')['weight'].transform('sum')
        edges_df['weight_norm'] = edges_df['weight'] / sum_per_src.replace({0:1})
    else:
        raise ValueError("normalize method must be 'max' or 'sum'")
    edges_df['weight_norm'] = edges_df['weight_norm'].fillna(0.0)
    return edges_df

def build_follow_list(norm_edges: pd.DataFrame, quantile: float):
    # 全局阈值（基于归一化权重的分位数）
    if norm_edges.empty:
        return pd.DataFrame(columns=['user_id','all_followed_user_ids'])
    thresh = float(norm_edges['weight_norm'].quantile(quantile))
    dff = norm_edges[norm_edges['weight_norm'] >= thresh].copy()
    # 按 source 排序并聚合 targets（按 weight_norm 降序）
    dff = dff.sort_values(['source','weight_norm'], ascending=[True, False])
    agg = dff.groupby('source')['target'].agg(lambda ids: list(ids)).reset_index()
    agg['all_followed_user_ids'] = agg['target'].apply(lambda lst: json.dumps(lst, ensure_ascii=False))
    agg = agg.rename(columns={'source':'user_id'}).loc[:, ['user_id','all_followed_user_ids']]
    return agg

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True, help='包含权重 CSV 的目录')
    p.add_argument('--pattern', default='attention_matrix*.csv', help='匹配权重文件的 glob 模式 (默认: attention_matrix*.csv)')
    p.add_argument('--out-dir', required=True, help='输出目录，保存合并结果与关注列表')
    p.add_argument('--quantile', type=float, default=0.25, help='全局噪声分位数，用于阈值过滤 (默认 0.25)')
    p.add_argument('--normalize', choices=['max','sum'], default='max', help='归一化方法: max 或 sum (默认 max)')
    p.add_argument('--recursive', action='store_true', help='是否递归查找输入目录下的匹配文件')
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_weight_files(input_dir, args.pattern, args.recursive)
    if not files:
        print(f"[ERROR] 在 {input_dir} 未找到匹配 {args.pattern} 的文件。", file=sys.stderr)
        sys.exit(1)
    print(f"找到 {len(files)} 个权重文件，开始合并...")

    merged = merge_edges(files)
    merged_path = out_dir / 'merged_edges.csv'
    merged.to_csv(merged_path, index=False, encoding='utf-8-sig')
    print(f"已保存合并边表: {merged_path}  (rows={len(merged)})")

    norm = normalize_edges(merged.copy(), method=args.normalize)
    norm_path = out_dir / 'normalized_edges.csv'
    norm.to_csv(norm_path, index=False, encoding='utf-8-sig')
    print(f"已保存归一化边表: {norm_path}")

    follow_df = build_follow_list(norm, quantile=args.quantile)
    follow_path = out_dir / 'follow_list.csv'
    follow_df.to_csv(follow_path, index=False, encoding='utf-8-sig')
    print(f"已保存关注列表: {follow_path} (users={len(follow_df)})")
    print("完成。")

if __name__ == '__main__':
    main()
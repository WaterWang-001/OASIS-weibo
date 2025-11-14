"""
终端入口：用户查询 -> 生成 user_id 列表 -> 提取子集数据 -> 三个对接 oasis 的转换器封装
用法示例：
  # 1) 按 profile CSV 条件筛选并输出 user list
  python run_user_selection.py select --input-csv MARS/data/output/2025-06-14/user_profiles.csv --output-file selected.txt gender=female min_followers_count=1000

  # 2) 使用 user list 提取子集数据（遍历所有日期 output）
  python run_user_selection.py collect --user-list selected.txt --input-root MARS/data/output --out-dir MARS/code/user_selection/subset

  # 3) 生成 oasis agent init（profiles + relations -> oasis_agent_init.csv）
  python run_user_selection.py convert_profiles --profile-csv MARS/code/user_selection/subset/profiles_subset.csv --relation-csv MARS/code/user_selection/subset/follow_list.csv --out-csv data/oasis/oasis_agent_init_subset.csv

  # 4) 合并权重并生成 follow_list（从 edges csv）
  python run_user_selection.py convert_relationships --input-dir MARS/data/output/2025-06-14 --out-dir MARS/code/user_selection/subset --pattern "attention_matrix*.csv" --quantile 0.25

  # 5) 把 posts 子集写入 oasis DB（需要源 posts DB）
  python run_user_selection.py convert_posts --source-db path/to/source.db --oasis-db data/oasis/oasis_database_subset.db

注意：脚本会把项目根加入 sys.path，以便 import 项目内模块。
"""
import sys
from pathlib import Path
import argparse
from typing import List, Optional

# 确保项目根在 sys.path（项目结构: /remote-home/JuelinW/oasis_project/MARS/code/user_selection/...）
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 本地模块（user selection / conversion）
from MARS.code.user_selection.user_selection import UserSelector
from MARS.code.user_selection.data_selection import SubsetCollector
from MARS.code.user_selection.oasis_user import OasisUserBuilder
from MARS.code.user_selection.oasis_relaitonship import merge_edges, normalize_edges, build_follow_list
from MARS.code.user_selection.oasis_post import OasisPostProcessor

import pandas as pd
import json

def run_select(input_csv: str, output_file: str, filters: List[str]):
    # 解析 filters 为 key=value dict
    criteria = {}
    for f in filters:
        if '=' not in f:
            print(f"忽略无效 filter: {f}")
            continue
        k, v = f.split('=', 1)
        # 基本类型转换
        if v.lower() in ("true","false"):
            vv = v.lower() == "true"
        else:
            try:
                vv = int(v)
            except Exception:
                try:
                    vv = float(v)
                except Exception:
                    vv = v
        criteria[k] = vv

    selector = UserSelector(profiles_csv_path=input_csv)
    user_ids = selector.select_users(**criteria)
    selector.save_list(user_ids, output_file)
    print(f"select -> saved {len(user_ids)} ids to {output_file}")
    return user_ids

def run_collect(user_list: Optional[str], users: Optional[str], input_root: str, out_dir: str, chunk: int = 500):
    if not user_list and not users:
        raise ValueError("需要 --user-list 或 --users 其中之一")
    if user_list:
        with open(user_list, 'r', encoding='utf-8') as f:
            user_ids = [l.strip() for l in f if l.strip()]
    else:
        user_ids = [u.strip() for u in users.split(',') if u.strip()]

    coll = SubsetCollector(input_root=Path(input_root), out_dir=Path(out_dir), chunk=chunk)
    coll.collect_subset(user_ids)
    print(f"collect -> 输出目录: {out_dir}")

def run_convert_profiles(profile_csv: str, relation_csv: str, out_csv: str):
    # 调用 OasisUserBuilder 将 profile+relation 转为 oasis agent init
    builder = OasisUserBuilder(profile_csv=profile_csv, relation_csv=relation_csv, output_csv=out_csv)
    builder.run()
    print(f"convert_profiles -> 生成: {out_csv}")

def run_convert_relationships(input_dir: str, pattern: str, out_dir: str, quantile: float = 0.25, normalize: str = "max", recursive: bool = False):
    # 使用 oasis_relaitonship 中的函数合并、归一并生成 follow_list
    inp = Path(input_dir)
    files = sorted(inp.rglob(pattern)) if recursive else sorted(inp.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到匹配文件: {input_dir}/{pattern}")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, dtype=str)
        except Exception as e:
            print(f"读取 {f} 失败: {e}")
            continue
        # normalize columns to source,target,weight
        cols = list(df.columns)
        if set(['source','target','weight']).issubset(set(cols)):
            df = df[['source','target','weight']].copy()
        elif len(cols) >= 3:
            df = df.iloc[:, :3].copy()
            df.columns = ['source','target','weight']
        else:
            print(f"跳过无法解析的文件: {f}")
            continue
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(0.0)
        dfs.append(df)
    if not dfs:
        raise RuntimeError("没有可用的 edges 文件")
    merged = pd.concat(dfs, ignore_index=True).groupby(['source','target'], as_index=False)['weight'].sum()
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir_p / "merged_edges.csv"
    merged.to_csv(merged_path, index=False, encoding='utf-8-sig')
    print(f"merged_edges -> {merged_path}")

    norm = normalize_edges(merged.copy(), method=normalize)
    norm_path = out_dir_p / "normalized_edges.csv"
    norm.to_csv(norm_path, index=False, encoding='utf-8-sig')
    print(f"normalized_edges -> {norm_path}")

    follow_df = build_follow_list(norm, quantile=float(quantile))
    follow_path = out_dir_p / "follow_list.csv"
    follow_df.to_csv(follow_path, index=False, encoding='utf-8-sig')
    print(f"follow_list -> {follow_path} (users={len(follow_df)})")

def run_convert_posts(source_db: str, oasis_db: str, calibration_end: Optional[str] = None, ground_truth_end: Optional[str] = None):
    # 调用 OasisPostProcessor 做基本迁移（若需要按时间窗口请传入 ISO 时间字符串）
    from datetime import datetime
    cal_dt = None
    gt_dt = None
    if calibration_end:
        cal_dt = datetime.fromisoformat(calibration_end)
    if ground_truth_end:
        gt_dt = datetime.fromisoformat(ground_truth_end)

    proc = OasisPostProcessor(source_db=source_db, oasis_db=oasis_db,
                              calibration_end=cal_dt or proc.CALIBRATION_END_TIME if 'proc' in locals() else None,
                              ground_truth_end=gt_dt or proc.GROUND_TRUTH_END_TIME if 'proc' in locals() else None,
                              create_calibration=True, create_ground_truth=True)
    # run 会在内部创建/覆盖 oasis_db（注意）
    proc.run()
    print(f"convert_posts -> oasis db written: {oasis_db}")

def cli():
    p = argparse.ArgumentParser(description="用户筛选 / 提取 / 转换 到 oasis 的终端工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    sel = sub.add_parser("select", help="从 profile CSV 筛选用户并保存 user list")
    sel.add_argument("--input-csv", required=True)
    sel.add_argument("--output-file", required=True)
    sel.add_argument("filters", nargs="*", help="过滤条件 key=value")

    coll = sub.add_parser("collect", help="根据 user list 提取子集数据")
    coll.add_argument("--user-list", help="user id 文本文件（每行一个）")
    coll.add_argument("--users", help="逗号分隔的 user id 列表")
    coll.add_argument("--input-root", default=str(REPO_ROOT / "MARS" / "data" / "output"))
    coll.add_argument("--out-dir", default=str(REPO_ROOT / "MARS" / "code" / "user_selection" / "subset_output"))
    coll.add_argument("--chunk", type=int, default=500)

    cp = sub.add_parser("convert_profiles", help="用 profiles+relations 生成 oasis_agent_init.csv")
    cp.add_argument("--profile-csv", required=True)
    cp.add_argument("--relation-csv", required=True)
    cp.add_argument("--out-csv", required=True)

    cr = sub.add_parser("convert_relationships", help="合并权重文件并生成 follow_list.csv")
    cr.add_argument("--input-dir", required=True)
    cr.add_argument("--pattern", default="attention_matrix*.csv")
    cr.add_argument("--out-dir", required=True)
    cr.add_argument("--quantile", type=float, default=0.25)
    cr.add_argument("--normalize", choices=["max","sum"], default="max")
    cr.add_argument("--recursive", action="store_true")

    cpp = sub.add_parser("convert_posts", help="把 posts 子集写入 oasis DB（调用 OasisPostProcessor）")
    cpp.add_argument("--source-db", required=True)
    cpp.add_argument("--oasis-db", required=True)
    cpp.add_argument("--calibration-end", help="ISO datetime string (e.g. 2025-06-02T16:30:00)", default=None)
    cpp.add_argument("--ground-truth-end", help="ISO datetime string", default=None)

    allp = sub.add_parser("all", help="select -> collect -> convert_relationships -> convert_profiles -> convert_posts")
    allp.add_argument("--input-csv", required=True)
    allp.add_argument("--output-file", required=True)
    allp.add_argument("--input-root", default=str(REPO_ROOT / "MARS" / "data" / "output"))
    allp.add_argument("--subset-out", default=str(REPO_ROOT / "MARS" / "code" / "user_selection" / "subset_output"))
    allp.add_argument("--edges-input-dir", default=str(REPO_ROOT / "MARS" / "data" / "output"))
    allp.add_argument("--oasis-out-dir", default=str(REPO_ROOT / "data" / "oasis"))
    allp.add_argument("--seed-oasis-db", default=str(REPO_ROOT / "MARS" / "code" / "user_selection" / "oasis_database_subset.db"))
    allp.add_argument("--quantile", type=float, default=0.25)
    allp.add_argument("--normalize", choices=["max","sum"], default="max")
    allp.add_argument("--chunk", type=int, default=500)
    allp.add_argument("--calibration-end", default=None)
    allp.add_argument("--ground-truth-end", default=None)

    args = p.parse_args()

    if args.cmd == "select":
        run_select(args.input_csv, args.output_file, args.filters)
    elif args.cmd == "collect":
        run_collect(args.user_list, args.users, args.input_root, args.out_dir, chunk=args.chunk)
    elif args.cmd == "convert_profiles":
        run_convert_profiles(args.profile_csv, args.relation_csv, args.out_csv)
    elif args.cmd == "convert_relationships":
        run_convert_relationships(args.input_dir, args.pattern, args.out_dir, quantile=args.quantile, normalize=args.normalize, recursive=args.recursive)
    elif args.cmd == "convert_posts":
        run_convert_posts(args.source_db, args.oasis_db, calibration_end=args.calibration_end, ground_truth_end=args.ground_truth_end)
    elif args.cmd == "all":
        # 流水线：select -> collect -> relationships -> profiles -> posts
        users = run_select(args.input_csv, args.output_file, [])
        run_collect(args.output_file, None, args.input_root, args.subset_out, chunk=args.chunk)
        # relationships on subset output
        subset_edges_in = args.edges_input_dir
        run_convert_relationships(subset_edges_in, "*.csv", args.subset_out, quantile=args.quantile, normalize=args.normalize, recursive=True)
        # convert profiles using generated follow_list
        profile_csv = str(Path(args.subset_out) / "profiles_subset.csv")
        relation_csv = str(Path(args.subset_out) / "follow_list.csv")
        out_csv = str(Path(args.oasis_out_dir) / "oasis_agent_init_subset.csv")
        run_convert_profiles(profile_csv, relation_csv, out_csv)
        # convert posts: try to find posts DB in subset_out (if created)
        src_db = args.seed_oasis_db
        oasis_db = str(Path(args.oasis_out_dir) / "oasis_database_subset.db")
        run_convert_posts(src_db, oasis_db, calibration_end=args.calibration_end, ground_truth_end=args.ground_truth_end)
    else:
        print("未知命令", args.cmd)

if __name__ == "__main__":
    cli()
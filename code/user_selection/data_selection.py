import argparse
import sqlite3
from pathlib import Path
from typing import List, Set, Optional
import pandas as pd
import sys
import json

# REPO 根（/remote-home/JuelinW/oasis_project）
REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_INPUT_ROOT = REPO_ROOT / "MARS" / "code" / "data_process"/ "output"
DEFAULT_OUT_DIR = REPO_ROOT / "MARS" / "code" / "user_selection" / "subset_output"

class SubsetCollector:
    """
    按用户列表从各日期输出目录中收集 profiles, edges, posts 的子集并写入 out_dir。
    用法：
        coll = SubsetCollector(input_root, out_dir)
        coll.collect_subset(user_list)
    """
    def __init__(self, input_root: Optional[Path] = None, out_dir: Optional[Path] = None, chunk: int = 500):
        self.input_root = Path(input_root) if input_root else DEFAULT_INPUT_ROOT
        self.out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
        self.CHUNK = int(chunk)

    @staticmethod
    def load_user_list(path: Optional[Path] = None, users_str: Optional[str] = None) -> List[str]:
        if path:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"user list file not found: {p}")
            return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        if users_str:
            return [u.strip() for u in users_str.split(",") if u.strip()]
        raise ValueError("需要 --user-list 或 --users 参数之一")

    @staticmethod
    def find_candidates(dir_path: Path):
        """返回三类候选文件（profiles csv, edges csv, posts db/csv）"""
        profiles = list(dir_path.glob("*profile*.csv")) + list(dir_path.glob("user_profiles*.csv"))
        edges = [f for f in dir_path.glob("*.csv") if any(k in f.name.lower() for k in ("edge", "attention", "matrix"))]
        posts_db = list(dir_path.glob("*.db")) + list(dir_path.glob("user_post_database*.db"))
        posts_csv = list(dir_path.glob("*post*.csv"))
        return profiles, edges, posts_db, posts_csv

    @staticmethod
    def filter_profiles(df: pd.DataFrame, user_set: Set[str]) -> pd.DataFrame:
        if 'user_id' not in df.columns:
            return pd.DataFrame(columns=df.columns)
        df['user_id'] = df['user_id'].astype(str)
        return df[df['user_id'].isin(user_set)].copy()

    @staticmethod
    def filter_edges(df: pd.DataFrame, user_set: Set[str]) -> pd.DataFrame:
        cols = [c for c in df.columns]
        lower = [c.lower() for c in cols]
        try:
            s_idx = lower.index('source')
            t_idx = lower.index('target')
            w_idx = lower.index('weight')
        except ValueError:
            if len(cols) >= 3:
                s_idx, t_idx, w_idx = 0, 1, 2
            else:
                return pd.DataFrame(columns=['source', 'target', 'weight'])
        df = df.iloc[:, [s_idx, t_idx, w_idx]].copy()
        df.columns = ['source', 'target', 'weight']
        df['source'] = df['source'].astype(str)
        df['target'] = df['target'].astype(str)
        return df[df['source'].isin(user_set) | df['target'].isin(user_set)].copy()

    def extract_posts_from_db(self, dbpath: Path, user_set: Set[str]) -> pd.DataFrame:
        rows = []
        try:
            conn = sqlite3.connect(f'file:{str(dbpath)}?mode=ro', uri=True)
        except Exception:
            conn = sqlite3.connect(str(dbpath))
        try:
            cur = conn.cursor()
            user_list = list(user_set)
            for i in range(0, len(user_list), self.CHUNK):
                chunk = user_list[i:i + self.CHUNK]
                placeholders = ",".join("?" for _ in chunk)
                q = f"SELECT * FROM post WHERE user_id IN ({placeholders})"
                try:
                    cur.execute(q, tuple(chunk))
                    cols = [d[0] for d in cur.description] if cur.description else []
                    for r in cur.fetchall():
                        rows.append(dict(zip(cols, r)))
                except Exception:
                    chunk_int = []
                    for u in chunk:
                        try:
                            chunk_int.append(int(u))
                        except Exception:
                            chunk_int.append(u)
                    placeholders = ",".join("?" for _ in chunk_int)
                    q = f"SELECT * FROM post WHERE user_id IN ({placeholders})"
                    try:
                        cur.execute(q, tuple(chunk_int))
                        cols = [d[0] for d in cur.description] if cur.description else []
                        for r in cur.fetchall():
                            rows.append(dict(zip(cols, r)))
                    except Exception:
                        continue
        finally:
            conn.close()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    @staticmethod
    def extract_posts_from_csv(path: Path, user_set: Set[str]) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception:
            df = pd.read_csv(path, dtype=str, encoding='utf-8', errors='ignore')
        if 'user_id' not in df.columns:
            for alt in ('uid', 'actor_id', 'from_user'):
                if alt in df.columns:
                    df['user_id'] = df[alt].astype(str)
                    break
        if 'user_id' not in df.columns:
            return pd.DataFrame()
        df['user_id'] = df['user_id'].astype(str)
        return df[df['user_id'].isin(user_set)].copy()

    def collect_subset(self, user_list: List[str]):
        user_set = set(user_list)
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        profiles_acc = []
        edges_acc = []
        posts_acc = []

        dates = sorted([p for p in self.input_root.iterdir() if p.is_dir()])
        if not dates:
            print(f"未在 {self.input_root} 找到任何日期目录", file=sys.stderr)
            return

        for d in dates:
            date_label = d.name
            profiles_cand, edges_cand, posts_db_cand, posts_csv_cand = self.find_candidates(d)

            for pf in profiles_cand:
                try:
                    dfp = pd.read_csv(pf, dtype=str)
                    sub = self.filter_profiles(dfp, user_set)
                    if not sub.empty:
                        sub['date'] = date_label
                        profiles_acc.append(sub)
                except Exception as e:
                    print(f"[WARN] 读取 profiles {pf} 失败: {e}", file=sys.stderr)

            for ef in edges_cand:
                try:
                    dfe = pd.read_csv(ef, dtype=str)
                    sube = self.filter_edges(dfe, user_set)
                    if not sube.empty:
                        sube['date'] = date_label
                        edges_acc.append(sube)
                except Exception as e:
                    print(f"[WARN] 读取 edges {ef} 失败: {e}", file=sys.stderr)

            for dbf in posts_db_cand:
                try:
                    dfp = self.extract_posts_from_db(dbf, user_set)
                    if not dfp.empty:
                        dfp['date'] = date_label
                        posts_acc.append(dfp)
                except Exception as e:
                    print(f"[WARN] 读取 posts db {dbf} 失败: {e}", file=sys.stderr)

            for pcf in posts_csv_cand:
                try:
                    dfp = self.extract_posts_from_csv(pcf, user_set)
                    if not dfp.empty:
                        dfp['date'] = date_label
                        posts_acc.append(dfp)
                except Exception as e:
                    print(f"[WARN] 读取 posts csv {pcf} 失败: {e}", file=sys.stderr)

        if profiles_acc:
            profiles_all = pd.concat(profiles_acc, ignore_index=True)
            profiles_all.to_csv(out_dir / "profiles_subset.csv", index=False, encoding='utf-8-sig')
            print(f"profiles_subset saved: {len(profiles_all)} rows")
        else:
            print("no profiles found for given users")

        if edges_acc:
            edges_all = pd.concat(edges_acc, ignore_index=True)
            edges_all = edges_all.drop_duplicates().sort_values(['date', 'source', 'target'])
            edges_all.to_csv(out_dir / "edges_subset.csv", index=False, encoding='utf-8-sig')
            print(f"edges_subset saved: {len(edges_all)} rows")
        else:
            print("no edges found for given users")

        if posts_acc:
            posts_all = pd.concat(posts_acc, ignore_index=True)
            posts_all.to_csv(out_dir / "posts_subset.csv", index=False, encoding='utf-8-sig')
            print(f"posts_subset saved: {len(posts_all)} rows")
        else:
            print("no posts found for given users")

        print("完成，输出目录:", out_dir)

    @classmethod
    def from_args_and_run(cls, argv: Optional[List[str]] = None):
        p = argparse.ArgumentParser(description="从 MARS 数据输出按用户列表提取子集（类接口）")
        p.add_argument("--user-list", help="包含 user_id 的文本文件，每行一个 ID", default=None)
        p.add_argument("--users", help="逗号分隔的 user_id 列表（可替代 --user-list）", default=None)
        p.add_argument("--input-root", help=f"输出日期目录根（默认: {DEFAULT_INPUT_ROOT})", default=str(DEFAULT_INPUT_ROOT))
        p.add_argument("--out-dir", help=f"子集输出目录（默认: {DEFAULT_OUT_DIR})", default=str(DEFAULT_OUT_DIR))
        p.add_argument("--chunk", type=int, default=500, help="SQLite IN 分批大小")
        args = p.parse_args(argv)

        try:
            users = cls.load_user_list(Path(args.user_list) if args.user_list else None, args.users)
        except Exception as e:
            print(f"加载用户列表失败: {e}", file=sys.stderr)
            sys.exit(1)

        coll = cls(input_root=Path(args.input_root), out_dir=Path(args.out_dir), chunk=args.chunk)
        coll.collect_subset(users)


if __name__ == "__main__":
    SubsetCollector.from_args_and_run()
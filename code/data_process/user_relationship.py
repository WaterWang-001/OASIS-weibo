"""
生成用户关系矩阵并在同一类中生成最终 follow list。
合并后类: UserRelationshipPipeline
- process_interactions: 解析 raw/*.txt，统计加权交互
- calculate_scores_and_matrix: 生成 edge DataFrame 与总分 Series
- generate_follow_list: 基于 edges 生成聚合关注列表（按全局分位数过滤）
- run: 一键完成全部并保存 CSV
"""
from pathlib import Path
from collections import defaultdict
import json
import pandas as pd
from typing import Iterable, Tuple, Dict, Set, Any

# --- 可配置项 ---
INPUT_DIRECTORY = 'data/raw/'
GLOBAL_NOISE_QUANTILE = 0.25
OUTPUT_MATRIX_CSV = 'scripts/attention_matrix_edges_COMBINED.csv'
OUTPUT_SCORES_CSV = 'scripts/total_attention_scores_COMBINED.csv'
OUTPUT_FOLLOW_LIST_CSV = 'data/user_relationship/final_follow_list_AGGREGATED.csv'

# 【V3 权重】
WEIGHTS = {
    'strong': 3,    # I_Mention (原创@, 评论@)
    'moderate': 2,  # I_P-Repost (一级转发), I_FOC (一级评论)
    'weak': 1       # I_S-Repost (二级转发), I_S-Comment (二级评论)
}


class UserRelationshipPipeline:
    """
    合并后的流水线类，包含从 raw -> edges -> scores -> follow list 的全流程。
    使用:
        p = UserRelationshipPipeline(input_directory='data/raw/')
        edges_df, scores_series, follow_df = p.run()
    或指定输出文件:
        p.run(output_matrix_csv=..., output_scores_csv=..., output_follow_csv=...)
    """

    def __init__(self, input_directory: str = INPUT_DIRECTORY, weights: Dict[str, int] = WEIGHTS,
                 global_noise_quantile: float = GLOBAL_NOISE_QUANTILE):
        self.input_directory = Path(input_directory)
        self.weights = weights
        self.global_noise_quantile = float(global_noise_quantile)

    def collect_file_list(self) -> Iterable[Path]:
        if not self.input_directory.exists() or not self.input_directory.is_dir():
            raise FileNotFoundError(f"输入目录不存在: {self.input_directory}")
        files = sorted(self.input_directory.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"在目录 {self.input_directory} 中未找到任何 .txt 文件")
        return files

    def _normalize_user_id(self, raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw.strip().lstrip("@")
        if isinstance(raw, (int,)):
            return str(raw)
        if isinstance(raw, dict):
            for k in ("uid", "id", "user_id", "userid", "uid_str"):
                if k in raw:
                    return str(raw[k])
        return str(raw)

    def _extract_mentions(self, record: dict) -> Iterable[Tuple[str, str]]:
        candidates = []
        for key in ("sjcjMentions", "mentions", "at_users", "atUserList", "atUser"):
            if key in record and record[key]:
                candidates = record[key]
                break
        if isinstance(candidates, str):
            try:
                candidates = json.loads(candidates)
            except Exception:
                candidates = [c.strip() for c in candidates.split(",") if c.strip()]
        if not candidates:
            return []

        out = []
        for item in candidates:
            if isinstance(item, dict):
                target = self._normalize_user_id(item.get("uid") or item.get("id") or item.get("user_id") or item.get("uid_str") or item.get("name"))
                typ = None
                for k in ("type", "interaction", "relation"):
                    if k in item:
                        typ = str(item[k])
                        break
                if typ:
                    typ_l = typ.lower()
                    if "mention" in typ_l or "at" in typ_l:
                        strength = "strong"
                    elif "p-repost" in typ_l or "foc" in typ_l or "forward" in typ_l or "repost" in typ_l:
                        strength = "moderate"
                    else:
                        strength = "moderate"
                else:
                    strength = "moderate"
                out.append((target, strength))
            else:
                target = self._normalize_user_id(item)
                out.append((target, "strong"))
        return out

    def process_interactions(self, file_list: Iterable[Path]) -> Tuple[Dict[Tuple[str, str], float], Set[str]]:
        interaction_counts = defaultdict(float)
        users = set()
        for fp in file_list:
            try:
                with fp.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        actor = rec.get("actor_id") or rec.get("user_id") or rec.get("uid") or rec.get("from_user") or rec.get("mid")
                        actor = self._normalize_user_id(actor)
                        if not actor:
                            continue
                        users.add(actor)
                        mentions = list(self._extract_mentions(rec))
                        for tgt, strength in mentions:
                            if not tgt or tgt == actor:
                                continue
                            users.add(tgt)
                            weight = self.weights.get(strength, self.weights.get('moderate', 1))
                            interaction_counts[(actor, tgt)] += float(weight)
            except Exception:
                continue
        return interaction_counts, users

    def calculate_scores_and_matrix(self, interaction_counts: Dict[Tuple[str, str], float]) -> Tuple[pd.DataFrame, pd.Series]:
        rows = []
        for (src, tgt), w in interaction_counts.items():
            rows.append({"source": src, "target": tgt, "weight": w})
        edges_df = pd.DataFrame(rows)
        if edges_df.empty:
            edges_df = pd.DataFrame(columns=["source", "target", "weight"])
            total_scores = pd.Series(dtype=float)
            return edges_df, total_scores
        total_scores = edges_df.groupby("source")["weight"].sum().sort_values(ascending=False)
        edges_df = edges_df.sort_values(["source", "weight"], ascending=[True, False]).reset_index(drop=True)
        return edges_df, total_scores

    def generate_follow_list(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        if edges_df.empty:
            return pd.DataFrame(columns=["user_id", "all_followed_user_ids"])
        thresh = float(edges_df["weight"].quantile(self.global_noise_quantile))
        df_filtered = edges_df[edges_df["weight"] >= thresh].copy()
        df_filtered = df_filtered.sort_values(["source", "weight"], ascending=[True, False])
        agg = df_filtered.groupby("source")["target"].agg(lambda ids: ";".join(map(str, ids))).reset_index()
        agg = agg.rename(columns={"source": "user_id", "target": "all_followed_user_ids"})
        return agg

    def run(self, output_matrix_csv: str = OUTPUT_MATRIX_CSV, output_scores_csv: str = OUTPUT_SCORES_CSV,
            output_follow_csv: str = OUTPUT_FOLLOW_LIST_CSV) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        files = self.collect_file_list()
        interaction_counts, users = self.process_interactions(files)
        edges_df, total_scores = self.calculate_scores_and_matrix(interaction_counts)
        Path(output_matrix_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(output_scores_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(output_follow_csv).parent.mkdir(parents=True, exist_ok=True)
        edges_df.to_csv(output_matrix_csv, index=False)
        total_scores.to_csv(output_scores_csv, header=["total_out_score"])
        follow_df = self.generate_follow_list(edges_df)
        follow_df.to_csv(output_follow_csv, index=False)
        return edges_df, total_scores, follow_df


def main():
    pipeline = UserRelationshipPipeline(input_directory=INPUT_DIRECTORY, weights=WEIGHTS, global_noise_quantile=GLOBAL_NOISE_QUANTILE)
    try:
        edges_df, scores, follow_df = pipeline.run(output_matrix_csv=OUTPUT_MATRIX_CSV,
                                                   output_scores_csv=OUTPUT_SCORES_CSV,
                                                   output_follow_csv=OUTPUT_FOLLOW_LIST_CSV)
        print(f"生成完成: edges({len(edges_df)}) scores({len(scores)}) follow_list({len(follow_df)})")
    except Exception as e:
        print(f"运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

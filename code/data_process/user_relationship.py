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
from typing import Iterable, Tuple, Dict, Set, Any, Optional

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
    合并后的流水线类（改动）：
    - 现在每次运行仅收集并保存“原始互动关系矩阵”（即未归一化/未阈值化的边表）
    - 关注列表（follow list）的生成逻辑已迁移到 simulation_prepare 步骤，不再在此处执行
    使用:
        p = UserRelationshipPipeline(input_directory='data/raw/')
        edges_df, total_scores = p.run(output_matrix_csv=..., save_scores=True/False)
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
        """
        根据 interaction_counts 生成边表（原始权重）和每个 source 的总分（总出度权重和）。
        注意：edges_df 中的 weight 为未经归一化/阈值化的原始累计权重。
        """
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
        """
        保留该方法以供外部（simulation_prepare）复用，但本类 run() 不再调用它。
        建议：在 simulation_prepare 中调用此函数或实现自己的阈值化/聚合逻辑。
        """
        if edges_df.empty:
            return pd.DataFrame(columns=["user_id", "all_followed_user_ids"])
        thresh = float(edges_df["weight"].quantile(self.global_noise_quantile))
        df_filtered = edges_df[edges_df["weight"] >= thresh].copy()
        df_filtered = df_filtered.sort_values(["source", "weight"], ascending=[True, False])
        agg = df_filtered.groupby("source")["target"].agg(lambda ids: ";".join(map(str, ids))).reset_index()
        agg = agg.rename(columns={"source": "user_id", "target": "all_followed_user_ids"})
        return agg

    def run(self, output_matrix_csv: str = OUTPUT_MATRIX_CSV,
            output_scores_csv: Optional[str] = None,
            save_scores: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        一键执行：收集文件 -> 统计互动（原始权重） -> 保存边表（原始互动矩阵）
        - output_matrix_csv: 保存原始边表（source,target,weight）
        - output_scores_csv: 若提供且 save_scores=True，会保存每个 source 的总分
        - 返回 (edges_df, total_scores)
        说明：不再在此处生成/保存 follow list，后续在 simulation_prepare 中处理。
        """
        files = self.collect_file_list()
        interaction_counts, users = self.process_interactions(files)
        edges_df, total_scores = self.calculate_scores_and_matrix(interaction_counts)

        Path(output_matrix_csv).parent.mkdir(parents=True, exist_ok=True)
        edges_df.to_csv(output_matrix_csv, index=False, encoding='utf-8-sig')

        if save_scores and output_scores_csv:
            Path(output_scores_csv).parent.mkdir(parents=True, exist_ok=True)
            total_scores.to_csv(output_scores_csv, header=["total_out_score"], encoding='utf-8-sig')

        return edges_df, total_scores


def main():
    pipeline = UserRelationshipPipeline(input_directory=INPUT_DIRECTORY, weights=WEIGHTS, global_noise_quantile=GLOBAL_NOISE_QUANTILE)
    try:
        edges_df, scores = pipeline.run(output_matrix_csv=OUTPUT_MATRIX_CSV,
                                        output_scores_csv=OUTPUT_SCORES_CSV,
                                        save_scores=True)
        print(f"生成完成: edges({len(edges_df)}) scores({len(scores)})")
    except Exception as e:
        print(f"运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

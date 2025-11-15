import sqlite3
import json
import os
import sys
import asyncio
import logging
import pandas as pd
from pathlib import Path
from attitude_annotator import OpenAIAttitudeAnnotator, VLLMAttitudeAnnotator

class OasisAttitudeProcessor:
    """
    封装后的 Attitude 处理器：
    - 支持标注 post / ground_truth_post
    - 支持从 DB 计算用户初始/最终分数并保存到 CSV
    使用：
        proc = OasisAttitudeProcessor()
        asyncio.run(proc.run())
    或仅运行生成 CSV：
        proc = OasisAttitudeProcessor()
        proc.generate_user_scores_to_csv()
    """
    def __init__(
        self,
        oasis_db_path: str = 'data/oasis/oasis_database_3000_random.db',
        user_csv_path: str = 'data/oasis/oasis_agent_init_3000_random.csv',
        user_csv_output_path: str = 'data/oasis/oasis_agent_init_3000_random.csv',
        attitude_columns=None,
        batch_size: int = 200,
        api_key: str = "",
        base_url: str = "https://api.nuwaapi.com/v1",
        concurrency_limit: int = 50,
        annotate_post_table: bool = True,
        annotate_gt_post_table: bool = True,
        generate_user_scores_csv: bool = True
    ):
        self.OASIS_DB_PATH = oasis_db_path
        self.USER_CSV_PATH = user_csv_path
        self.USER_CSV_OUTPUT_PATH = user_csv_output_path
        self.ATTITUDE_COLUMNS = attitude_columns or [
            'attitude_lifestyle_culture',
            'attitude_sport_ent',
            'attitude_sci_health',
            'attitude_politics_econ'
        ]
        self.BATCH_SIZE = batch_size
        self.API_KEY = api_key
        self.BASE_URL = base_url
        self.API_CONCURRENCY_LIMIT = concurrency_limit

        self.ANNOTATE_POST_TABLE = annotate_post_table
        self.ANNOTATE_GT_POST_TABLE = annotate_gt_post_table
        self.GENERATE_USER_SCORES_CSV = generate_user_scores_csv
        

        self.annotator = None
        self.VLLM_MODEL_PATH = "/remote-home/JuelinW/oasis_project/Qwen2.5-7B-Instruct"
        # 日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def _ensure_db_exists(self):
        if not os.path.exists(self.OASIS_DB_PATH):
            raise FileNotFoundError(f"目标数据库不存在: {self.OASIS_DB_PATH}")

    def _init_annotator(self):
        if self.annotator is None:
            self.annotator = VLLMAttitudeAnnotator(
                model_name=self.VLLM_MODEL_PATH,
                attitude_columns=self.ATTITUDE_COLUMNS,
                concurrency_limit=1,
                log_interval_posts=100
            )
            # self.annotator = OpenAIAttitudeAnnotator(
            #     api_key=self.API_KEY,
            #     base_url=self.BASE_URL,
            #     attitude_columns=self.ATTITUDE_COLUMNS,
            #     batch_size=self.BATCH_SIZE,
            #     concurrency_limit=self.API_CONCURRENCY_LIMIT
            # )

    def generate_user_scores_to_csv(self):
        """
        同步方法：读取 USER_CSV_PATH，连接 OASIS_DB_PATH（只读），计算 initial / final 分数并保存到 USER_CSV_OUTPUT_PATH
        """
        logging.info("开始生成用户分数 CSV...")
        try:
            conn = sqlite3.connect(f'file:{self.OASIS_DB_PATH}?mode=ro', uri=True)
        except sqlite3.Error as e:
            logging.error(f"数据库连接失败: {e}")
            return

        if not os.path.exists(self.USER_CSV_PATH):
            logging.error(f"找不到用户 CSV: {self.USER_CSV_PATH}")
            conn.close()
            return

        try:
            user_df = pd.read_csv(self.USER_CSV_PATH, dtype={'user_id': str})
        except Exception as e:
            logging.error(f"读取用户 CSV 失败: {e}")
            conn.close()
            return

        avg_cols_sql = ", ".join([f"AVG({col}) as {col}" for col in self.ATTITUDE_COLUMNS])
        initial_query = f"SELECT user_id, {avg_cols_sql} FROM post GROUP BY user_id"

        try:
            initial_df = pd.read_sql_query(initial_query, conn, dtype={'user_id': str})
        except sqlite3.Error as e:
            logging.error("查询 'post' 表失败，可能需要先运行标注任务。错误: %s", e)
            conn.close()
            return

        initial_df = initial_df.rename(columns={col: f"initial_{col}" for col in self.ATTITUDE_COLUMNS})

        att_cols_sql = ", ".join(self.ATTITUDE_COLUMNS)
        union_query = f"""
            SELECT user_id, {att_cols_sql} FROM post
            UNION ALL
            SELECT user_id, {att_cols_sql} FROM ground_truth_post
        """
        final_query = f"""
            SELECT user_id, {avg_cols_sql}
            FROM ({union_query}) AS combined
            GROUP BY user_id
        """
        try:
            final_df = pd.read_sql_query(final_query, conn, dtype={'user_id': str})
        except sqlite3.Error as e:
            logging.error("查询 'ground_truth_post' 失败，可能需要先运行标注任务。错误: %s", e)
            conn.close()
            return

        final_df = final_df.rename(columns={col: f"final_{col}" for col in self.ATTITUDE_COLUMNS})
        conn.close()

        # 合并
        cols_to_drop = [f"initial_{col}" for col in self.ATTITUDE_COLUMNS] + [f"final_{col}" for col in self.ATTITUDE_COLUMNS] + ['initial_attitude_avg', 'final_attitude_avg']
        user_df_cleaned = user_df.drop(columns=cols_to_drop, errors='ignore')
        merged_df = pd.merge(user_df_cleaned, initial_df, on='user_id', how='left')
        merged_df = pd.merge(merged_df, final_df, on='user_id', how='left')

        initial_cols = [f"initial_{col}" for col in self.ATTITUDE_COLUMNS]
        final_cols = [f"final_{col}" for col in self.ATTITUDE_COLUMNS]
        new_score_columns = initial_cols + final_cols
        merged_df[new_score_columns] = merged_df[new_score_columns].fillna(0.0)

        merged_df['initial_attitude_avg'] = merged_df[initial_cols].mean(axis=1)
        merged_df['final_attitude_avg'] = merged_df[final_cols].mean(axis=1)

        Path(self.USER_CSV_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
        try:
            merged_df.to_csv(self.USER_CSV_OUTPUT_PATH, index=False, encoding='utf-8-sig')
            logging.info("用户分数 CSV 已保存: %s", self.USER_CSV_OUTPUT_PATH)
        except Exception as e:
            logging.error("保存 CSV 失败: %s", e)

    async def annotate_tables(self):
        """
        异步方法：根据配置标注 post / ground_truth_post 表
        """

        if self.ANNOTATE_POST_TABLE or self.ANNOTATE_GT_POST_TABLE:
            self._init_annotator()

        results = {}
        try:
            if self.ANNOTATE_POST_TABLE:
                logging.info("开始标注 post 表 (校准集)...")
                await self.annotator.annotate_table(self.OASIS_DB_PATH, "post", only_sim_posts=False)
                results['post'] = True
            if self.ANNOTATE_GT_POST_TABLE:
                logging.info("开始标注 ground_truth_post 表...")
                await self.annotator.annotate_table(self.OASIS_DB_PATH, "ground_truth_post", only_sim_posts=False)
                results['ground_truth_post'] = True
        except Exception as e:
            logging.error("标注过程中发生错误: %s", e)
            raise
        return results

    async def run(self):
        """
        一键执行：标注（可选）-> 生成用户分数 CSV（可选）
        """
        logging.info("OasisAttitudeProcessor 启动")
        try:
            self._ensure_db_exists()
        except FileNotFoundError as e:
            logging.error(e)
            return

        try:
            if self.ANNOTATE_POST_TABLE or self.ANNOTATE_GT_POST_TABLE:
                await self.annotate_tables()

            if self.GENERATE_USER_SCORES_CSV:
                # 同步函数直接调用
                self.generate_user_scores_to_csv()

        except Exception as e:
            logging.error("运行失败: %s", e)
            import traceback
            traceback.print_exc()

# 兼容脚本直接运行
if __name__ == "__main__":
    proc = OasisAttitudeProcessor(
        oasis_db_path='oasis_test/oasis/oasis_database_100000_random.db',
        user_csv_path='oasis_test/oasis/oasis_agent_init_100000_random.csv',
        user_csv_output_path='oasis_test/oasis/oasis_agent_init_100000_random.csv',
        api_key="",
        annotate_post_table=True,
        annotate_gt_post_table=True,
        generate_user_scores_csv=True
    )
    asyncio.run(proc.run())
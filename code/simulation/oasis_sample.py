import pandas as pd
import sqlite3
import os
import sys
import ast
from tqdm import tqdm
import time
from pathlib import Path
class OasisSampler:
    """
    抽样器：将 oasis_sample.py 的主流程封装为类。
    用法示例:
        sampler = OasisSampler(
            source_csv="data/oasis/oasis_agent_init.csv",
            source_db="data/oasis/oasis_database.db",
            target_csv="data/oasis/oasis_agent_init_5000_random.csv",
            target_db="data/oasis/oasis_database_5000_random.db",
            seed_size=3000,
            avoid_unannotated=False
        )
        sampler.run()
    """
    def __init__(self,
                 source_csv: str = "data/oasis/oasis_agent_init.csv",
                 source_db: str = "data/oasis/oasis_database.db",
                 target_csv: str = "data/oasis/oasis_agent_init_5000_random.csv",
                 target_db: str = "data/oasis/oasis_database_5000_random.db",
                 seed_size: int = 3000,
                 avoid_unannotated: bool = False):
        self.SOURCE_CSV_PATH = source_csv
        self.SOURCE_DB_PATH = source_db
        self.TARGET_CSV_PATH = target_csv
        self.TARGET_DB_PATH = target_db
        self.SEED_SAMPLE_SIZE = seed_size
        self.AVOID_UNANNOTATED_USERS = avoid_unannotated
        self.TABLES_TO_COPY = ['user', 'post', 'ground_truth_post']

    @staticmethod
    def parse_following_list(list_str: str) -> set:
        if not isinstance(list_str, str) or not list_str.startswith('['):
            return set()
        try:
            parsed_list = ast.literal_eval(list_str)
            return set(parsed_list)
        except (ValueError, SyntaxError):
            return set()

    def run(self):
        start_time = time.time()
        if not os.path.exists(self.SOURCE_CSV_PATH) or not os.path.exists(self.SOURCE_DB_PATH):
            raise FileNotFoundError("源 CSV 或 DB 不存在")
        unannotated_user_set = set()
        if self.AVOID_UNANNOTATED_USERS:
            with sqlite3.connect(f'file:{self.SOURCE_DB_PATH}?mode=ro', uri=True) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT DISTINCT user_id FROM post WHERE attitude_annotated = 0")
                    bad_users_post = {str(row[0]) for row in cursor.fetchall()}
                    unannotated_user_set.update(bad_users_post)
                except sqlite3.Error:
                    pass
                try:
                    cursor.execute("SELECT DISTINCT user_id FROM ground_truth_post WHERE attitude_annotated = 0")
                    bad_users_gt = {str(row[0]) for row in cursor.fetchall()}
                    unannotated_user_set.update(bad_users_gt)
                except sqlite3.Error:
                    pass
        df_full = pd.read_csv(self.SOURCE_CSV_PATH, usecols=['user_id', 'following_agentid_list'], dtype={'user_id': str})
        df_full = df_full.dropna(subset=['user_id'])
        df_full = df_full[df_full['user_id'].str.strip() != '']
        df_eligible = df_full[~df_full['user_id'].isin(unannotated_user_set)]
        if len(df_eligible) < self.SEED_SAMPLE_SIZE:
            df_seed = df_eligible
        else:
            df_seed = df_eligible.sample(n=self.SEED_SAMPLE_SIZE, random_state=42)
        final_user_set = set(df_seed['user_id'])
        for list_str in tqdm(df_seed['following_agentid_list'].fillna('[]'), desc="解析关注列表"):
            followed_users = self.parse_following_list(list_str)
            for user_id in followed_users:
                try:
                    str_user_id = str(user_id)
                    if str_user_id not in final_user_set:
                        final_user_set.add(str_user_id)
                except (ValueError, TypeError):
                    pass
        # 保存子集 CSV
        df_all = pd.read_csv(self.SOURCE_CSV_PATH, dtype={'user_id': str})
        df_all = df_all.dropna(subset=['user_id'])
        df_all = df_all[df_all['user_id'].str.strip() != '']
        df_final = df_all[df_all['user_id'].isin(final_user_set)]
        Path(self.TARGET_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(self.TARGET_CSV_PATH, index=False)
        # 创建子集 DB
        if os.path.exists(self.TARGET_DB_PATH):
            os.remove(self.TARGET_DB_PATH)
        final_user_list_tuples = [(uid,) for uid in final_user_set]
        try:
            with sqlite3.connect(self.TARGET_DB_PATH) as target_conn:
                target_conn.execute(f"ATTACH DATABASE '{self.SOURCE_DB_PATH}' AS source_db")
                target_conn.execute("CREATE TABLE _sampled_users (user_id INTEGER PRIMARY KEY)")
                final_user_int_tuples = []
                for (uid_str,) in final_user_list_tuples:
                    try:
                        final_user_int_tuples.append((int(uid_str),))
                    except ValueError:
                        pass
                batch_size = 50000
                for i in range(0, len(final_user_int_tuples), batch_size):
                    batch = final_user_int_tuples[i:i + batch_size]
                    target_conn.executemany("INSERT INTO _sampled_users (user_id) VALUES (?)", batch)
                target_conn.commit()
                for table_name in self.TABLES_TO_COPY:
                    schema = target_conn.execute(f"SELECT sql FROM source_db.sqlite_master WHERE type='table' AND name='{table_name}'").fetchone()
                    if not schema:
                        continue
                    target_conn.execute(schema[0])
                    copy_query = f"""
                        INSERT INTO {table_name}
                        SELECT * FROM source_db.{table_name}
                        WHERE user_id IN (SELECT user_id FROM _sampled_users)
                    """
                    target_conn.execute(copy_query)
                    target_conn.commit()
                target_conn.execute("DROP TABLE _sampled_users")
                target_conn.execute("DETACH DATABASE source_db")
                target_conn.commit()
        except Exception:
            raise

        end_time = time.time()
        print(f"\n--- 🏁 脚本总运行完毕 ---")
        print(f"  -> 总耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    sampler = OasisSampler(
        source_csv="data/oasis/oasis_agent_init.csv",
        source_db="data/oasis/oasis_database.db",
        target_csv="data/oasis/oasis_agent_init_5000_random.csv",
        target_db="data/oasis/oasis_database_5000_random.db",
        seed_size=3000,
        avoid_unannotated=False
    )
    sampler.run()
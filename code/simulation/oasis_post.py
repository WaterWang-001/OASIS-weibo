import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
import time
import sys

class OasisPostProcessor:
    """
    将原有 oasis_post.py 的逻辑封装为类接口。
    用法示例:
        proc = OasisPostProcessor(
            source_db='data/user_post/user_post_database.db',
            oasis_db='data/oasis/oasis_database.db',
            calibration_end=datetime(2025,6,2,16,30,0),
            ground_truth_end=datetime(2025,6,2,16,45,0),
            batch_size=50000,
            create_calibration=True,
            create_ground_truth=True
        )
        proc.run()
    """
    def __init__(self,
                 source_db: str = 'data/user_post/user_post_database.db',
                 oasis_db: str = 'data/oasis/oasis_database.db',
                 calibration_end: datetime = datetime(2025,6,2,16,30,0),
                 ground_truth_end: datetime = datetime(2025,6,2,16,45,0),
                 batch_size: int = 50000,
                 create_calibration: bool = True,
                 create_ground_truth: bool = True):
        self.SOURCE_DB_PATH = source_db
        self.OASIS_DB_PATH = oasis_db
        self.CALIBRATION_END_TIME = calibration_end
        self.GROUND_TRUTH_END_TIME = ground_truth_end
        self.BATCH_SIZE = batch_size
        self.CREATE_CALIBRATION_SET = create_calibration
        self.CREATE_GROUND_TRUTH_SET = create_ground_truth

    # 以下为原脚本中的函数改为类方法
    def create_target_table(self, target_conn, table_name: str):
        cur = target_conn.cursor()
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            post_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT, 
            original_post_id INTEGER,
            content TEXT DEFAULT '',
            quote_content TEXT,
            created_at DATETIME,
            num_likes INTEGER DEFAULT 0,
            num_dislikes INTEGER DEFAULT 0,
            num_shares INTEGER DEFAULT 0,
            num_reports INTEGER DEFAULT 0,
            temp_sjcjId TEXT UNIQUE, 
            temp_original_sjcjId TEXT,
            FOREIGN KEY(user_id) REFERENCES user(user_id),
            FOREIGN KEY(original_post_id) REFERENCES {table_name}(post_id)
        );
        """)
        target_conn.commit()

    @staticmethod
    def parse_log_line(data_json: str):
        try:
            data = json.loads(data_json)
            comment_pojo = data.get('commentPojo')
            content_pojo = data.get('contentPojo')
            comment_fwd_pojo = data.get('commentForwardPojo')
            content_fwd_pojo = data.get('contentForwardPojo')
            content_root_pojo = data.get('contentRootPojo')

            source_sjcjId = None
            source_original_sjcjId = None
            content = ""
            quote_content = None

            if comment_pojo:
                source_sjcjId = comment_pojo.get('sjcjId')
                content = comment_pojo.get('sjqxTitle', "")
                if comment_fwd_pojo:
                    source_original_sjcjId = comment_fwd_pojo.get('sjcjId')
                    quote_content = comment_fwd_pojo.get('sjcjContent') or comment_fwd_pojo.get('sjqxTitle')
                elif content_pojo:
                    source_original_sjcjId = content_pojo.get('sjcjId')
                    quote_content = content_pojo.get('sjqxContent') or content_pojo.get('sjcjContent')
            elif content_pojo:
                source_sjcjId = content_pojo.get('sjcjId')
                content = content_pojo.get('sjqxContent', "")
                if content_fwd_pojo:
                    if content_root_pojo and content_root_pojo.get('sjcjId'):
                        source_original_sjcjId = content_root_pojo.get('sjcjId')
                    else:
                        source_original_sjcjId = content_fwd_pojo.get('sjcjId')
                    if content_root_pojo:
                        quote_content = content_root_pojo.get('sjqxContent') or content_root_pojo.get('sjcjContent')
                    quote_content = quote_content or content_fwd_pojo.get('sjqxContent') or content_fwd_pojo.get('sjcjContent')
                else:
                    source_original_sjcjId = None
                    quote_content = None
            else:
                return (None, None, None, None)
            return (source_sjcjId, source_original_sjcjId, content, quote_content)
        except (json.JSONDecodeError, TypeError):
            return (None, None, None, None)

    def migrate_data(self, source_conn, target_conn, table_name: str, sql_filter: str, filter_params: tuple):
        source_cur = source_conn.cursor()
        target_cur = target_conn.cursor()
        target_cur.execute("PRAGMA journal_mode = WAL;")
        target_cur.execute("PRAGMA synchronous = NORMAL;")
        source_cur.execute(
            f"SELECT user_id, timestamp, data_json FROM content {sql_filter}",
            filter_params
        )
        insert_batch = []
        total_rows = 0
        error_rows = 0
        start_time = time.time()
        while True:
            rows = source_cur.fetchmany(self.BATCH_SIZE)
            if not rows:
                break
            for row in rows:
                total_rows += 1
                user_id = row[0]
                timestamp_ms = row[1]
                data_json = row[2]
                try:
                    created_at = datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    error_rows += 1
                    continue
                parsed = self.parse_log_line(data_json)
                (source_sjcjId, source_original_sjcjId, content, quote_content) = parsed
                if source_sjcjId is None:
                    error_rows += 1
                    continue
                insert_batch.append((
                    str(user_id),
                    None,
                    content,
                    quote_content,
                    created_at,
                    source_sjcjId,
                    source_original_sjcjId
                ))
            if insert_batch:
                target_cur.executemany(f"""
                    INSERT OR IGNORE INTO {table_name} (
                        user_id, original_post_id, content, quote_content, created_at,
                        temp_sjcjId, temp_original_sjcjId
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, insert_batch)
                target_conn.commit()
                insert_batch = []
        end_time = time.time()
        inserted_rows = target_cur.execute(f"SELECT COUNT(post_id) FROM {table_name}").fetchone()[0]
        return {
            "total_rows": total_rows,
            "inserted_rows": inserted_rows,
            "error_rows": error_rows,
            "elapsed": end_time - start_time
        }

    def link_table_post(self, conn, table_name):
        cur = conn.cursor()
        if table_name == "post":
            cur.execute("CREATE INDEX IF NOT EXISTS idx_post_temp_original ON post (temp_original_sjcjId);")
            conn.commit()
            cur.execute("""
                UPDATE post
                SET original_post_id = (
                    SELECT p2.post_id FROM post AS p2
                    WHERE p2.temp_sjcjId = post.temp_original_sjcjId
                )
                WHERE post.temp_original_sjcjId IS NOT NULL;
            """)
            conn.commit()
        else:
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_temp_original ON {table_name} (temp_original_sjcjId);")
            conn.commit()
            cur.execute(f"""
                UPDATE {table_name}
                SET original_post_id = COALESCE(
                    (SELECT p_cal.post_id FROM post AS p_cal WHERE p_cal.temp_sjcjId = {table_name}.temp_original_sjcjId),
                    (SELECT p_gt.post_id FROM ground_truth_post AS p_gt WHERE p_gt.temp_sjcjId = {table_name}.temp_original_sjcjId)
                )
                WHERE {table_name}.temp_original_sjcjId IS NOT NULL;
            """)
            conn.commit()

    def run(self):
        if not os.path.exists(self.SOURCE_DB_PATH):
            raise FileNotFoundError(f"源数据库不存在: {self.SOURCE_DB_PATH}")
        cal_end_ms = int(self.CALIBRATION_END_TIME.timestamp() * 1000)
        gt_end_ms = int(self.GROUND_TRUTH_END_TIME.timestamp() * 1000)
        try:
            if self.CREATE_CALIBRATION_SET:
                if os.path.exists(self.OASIS_DB_PATH):
                    os.remove(self.OASIS_DB_PATH)
                with sqlite3.connect(self.SOURCE_DB_PATH) as source_conn:
                    with sqlite3.connect(self.OASIS_DB_PATH) as target_conn:
                        self.create_target_table(target_conn, "post")
                        res = self.migrate_data(source_conn, target_conn, "post", "WHERE timestamp <= ?", (cal_end_ms,))
                        # link
                        self.link_table_post(sqlite3.connect(self.OASIS_DB_PATH), "post")
            if self.CREATE_GROUND_TRUTH_SET:
                with sqlite3.connect(self.SOURCE_DB_PATH) as source_conn:
                    with sqlite3.connect(self.OASIS_DB_PATH) as target_conn:
                        self.create_target_table(target_conn, "ground_truth_post")
                        _ = self.migrate_data(source_conn, target_conn, "ground_truth_post", "WHERE timestamp > ? AND timestamp <= ?", (cal_end_ms, gt_end_ms))
                        self.link_table_post(sqlite3.connect(self.OASIS_DB_PATH), "ground_truth_post")
        except Exception as e:
            raise
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import os
import os.path as osp
import sqlite3
from typing import Any, Dict, List

SCHEMA_DIR = "social_platform/schema"
DB_DIR = "db"
DB_NAME = "social_media.db"

USER_SCHEMA_SQL = "user.sql"
POST_SCHEMA_SQL = "post.sql"
INTERVENTION_MESSAGE_SCHEMA_SQL="intervention_message.sql"
FOLLOW_SCHEMA_SQL = "follow.sql"
MUTE_SCHEMA_SQL = "mute.sql"
LIKE_SCHEMA_SQL = "like.sql"
DISLIKE_SCHEMA_SQL = "dislike.sql"
REPORT_SCHEAM_SQL = "report.sql"
TRACE_SCHEMA_SQL = "trace.sql"
REC_SCHEMA_SQL = "rec.sql"
COMMENT_SCHEMA_SQL = "comment.sql"
COMMENT_LIKE_SCHEMA_SQL = "comment_like.sql"
COMMENT_DISLIKE_SCHEMA_SQL = "comment_dislike.sql"
PRODUCT_SCHEMA_SQL = "product.sql"
GROUP_SCHEMA_SQL = "chat_group.sql"
GROUP_MEMBER_SCHEMA_SQL = "group_member.sql"
GROUP_MESSAGE_SCHEMA_SQL = "group_message.sql"
LOG_ATTITUDE_LIFESTYLE_CULTURE_SCHEMA_SQL = "log_attitude_lifestyle_culture.sql"
LOG_ATTITUDE_SPORT_ENT_SCHEMA_SQL = "log_attitude_sport_ent.sql"
LOG_ATTITUDE_SCI_HEALTH_SCHEMA_SQL = "log_attitude_sci_health.sql"
LOG_ATTITUDE_POLITICS_ECON_SCHEMA_SQL = "log_attitude_politics_econ.sql"
LOG_ATTITUDE_AVERAGE_SCHEMA_SQL = "log_attitude_average.sql"

TABLE_NAMES = {
    "user",
    "post",
    "intervention_message",
    "follow",
    "mute",
    "like",
    "dislike",
    "report",
    "trace",
    "rec",
    "comment.sql",
    "comment_like.sql",
    "comment_dislike.sql",
    "product.sql",
    "group",
    "group_member",
    "group_message",
    "log_attitude_lifestyle_culture",
    "log_attitude_sport_ent",
    "log_attitude_sci_health",
    "log_attitude_politics_econ",
    "log_attitude_average",
}


def get_db_path() -> str:
    curr_file_path = osp.abspath(__file__)
    parent_dir = osp.dirname(osp.dirname(curr_file_path))
    db_dir = osp.join(parent_dir, DB_DIR)
    os.makedirs(db_dir, exist_ok=True)
    db_path = osp.join(db_dir, DB_NAME)
    return db_path


def get_schema_dir_path() -> str:
    curr_file_path = osp.abspath(__file__)
    parent_dir = osp.dirname(osp.dirname(curr_file_path))
    schema_dir = osp.join(parent_dir, SCHEMA_DIR)
    return schema_dir


# --- 【!! 新增的辅助函数 !!】 ---
def _execute_schema_if_not_exists(
    cursor: sqlite3.Cursor, schema_dir: str, sql_file_name: str, table_name: str
):
    """
    辅助函数：
    1. 检查 'table_name' 是否存在。
    2. 如果不存在，才执行 'sql_file_name' 脚本。
    """
    try:
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        )
        if not cursor.fetchone():
            print(f"  -> '{table_name}' table not found. Creating from {sql_file_name}...")
            sql_path = osp.join(schema_dir, sql_file_name)
            with open(sql_path, "r") as sql_file:
                sql_script = sql_file.read()
            cursor.executescript(sql_script)
        else:
            print(f"  -> '{table_name}' table already exists. Skipping creation.")
    except sqlite3.Error as e:
        print(f"  -> WARNING: Error checking/creating '{table_name}': {e}")
    except FileNotFoundError:
        print(f"  -> WARNING: SQL file not found, skipping: {sql_file_name}")


# --- 【!! 修改后的 create_db 函数 !!】 ---
def create_db(db_path: str | None = None):
    r"""Create the database tables IF THEY DO NOT EXIST.
    This function is now safe to run on an existing database.
    """
    schema_dir = get_schema_dir_path()
    if db_path is None:
        db_path = get_db_path()

    # Connect to the database:
    print("db_path", db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # --- 重构所有表，使用辅助函数 ---
        _execute_schema_if_not_exists(
            cursor, schema_dir, USER_SCHEMA_SQL, "user"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, POST_SCHEMA_SQL, "post"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, INTERVENTION_MESSAGE_SCHEMA_SQL, "intervention_message"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, FOLLOW_SCHEMA_SQL, "follow"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, MUTE_SCHEMA_SQL, "mute"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, LIKE_SCHEMA_SQL, "like"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, DISLIKE_SCHEMA_SQL, "dislike"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, REPORT_SCHEAM_SQL, "report"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, TRACE_SCHEMA_SQL, "trace"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, REC_SCHEMA_SQL, "rec"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, COMMENT_SCHEMA_SQL, "comment"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, COMMENT_LIKE_SCHEMA_SQL, "comment_like"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, COMMENT_DISLIKE_SCHEMA_SQL, "comment_dislike"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, PRODUCT_SCHEMA_SQL, "product"
        )
    
        _execute_schema_if_not_exists(
            cursor, schema_dir, GROUP_SCHEMA_SQL, "chat_group"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, GROUP_MEMBER_SCHEMA_SQL, "group_member"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, GROUP_MESSAGE_SCHEMA_SQL, "group_message"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, LOG_ATTITUDE_LIFESTYLE_CULTURE_SCHEMA_SQL, "log_attitude_lifestyle_culture"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, LOG_ATTITUDE_SPORT_ENT_SCHEMA_SQL, "log_attitude_sport_ent"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, LOG_ATTITUDE_SCI_HEALTH_SCHEMA_SQL, "log_attitude_sci_health"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, LOG_ATTITUDE_POLITICS_ECON_SCHEMA_SQL, "log_attitude_politics_econ"
        )
        _execute_schema_if_not_exists(
            cursor, schema_dir, LOG_ATTITUDE_AVERAGE_SCHEMA_SQL, "log_attitude_average"
        )

        # Commit the changes:
        conn.commit()
        print("  -> Database check/creation complete.")

    except sqlite3.Error as e:
        print(f"An error occurred while creating tables: {e}")

    return conn, cursor


def print_db_tables_summary():
    # Connect to the SQLite database
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Retrieve a list of all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Print a summary of each table
    for table in tables:
        table_name = table[0]
        # (我们不再需要 TABLE_NAMES 集合了，直接打印所有表)
        # if table_name not in TABLE_NAMES:
        #     continue
        print(f"Table: {table_name}")

        # Retrieve the table schema
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        print("- Columns:", column_names)

        # Retrieve and print foreign key information
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = cursor.fetchall()
        if foreign_keys:
            print("- Foreign Keys:")
            for fk in foreign_keys:
                print(
                    f"    {fk[2]} references {fk[3]}({fk[4]}) on update "
                    f"{fk[5]} on delete {fk[6]}"
                )
        else:
            print("  No foreign keys.")

        # Print the first few rows of the table
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
        rows = cursor.fetchall()
        for row in rows:
            print(row)
        print()  # Adds a newline for better readability between tables

    # Close the database connection
    conn.close()


def fetch_table_from_db(
    cursor: sqlite3.Cursor, table_name: str
) -> List[Dict[str, Any]]:
    cursor.execute(f"SELECT * FROM {table_name}")
    columns = [description[0] for description in cursor.description]
    data_dicts = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return data_dicts


def fetch_rec_table_as_matrix(cursor: sqlite3.Cursor) -> List[List[int]]:
    # First, query all user_ids from the user table, assuming they start from
    # 1 and are consecutive
    cursor.execute("SELECT user_id FROM user ORDER BY user_id")
    user_ids = [row[0] for row in cursor.fetchall()]

    # Then, query all records from the rec table
    cursor.execute(
        "SELECT user_id, post_id FROM rec ORDER BY user_id, post_id"
    )
    rec_rows = cursor.fetchall()
    # Initialize a dictionary, assigning an empty list to each user_id
    user_posts = {user_id: [] for user_id in user_ids}
    # Fill the dictionary with the records queried from the rec table
    for user_id, post_id in rec_rows:
        if user_id in user_posts:
            user_posts[user_id].append(post_id)
    # Convert the dictionary into matrix form
    matrix = [user_posts[user_id] for user_id in user_ids]
    return matrix


def insert_matrix_into_rec_table(
    cursor: sqlite3.Cursor, matrix: List[List[int]]
) -> None:
    # Iterate through the matrix, skipping the placeholder at index 0
    for user_id, post_ids in enumerate(matrix, start=1):
        # Adjusted to start counting from 1
        for post_id in post_ids:
            # Insert each combination of user_id and post_id into the rec table
            cursor.execute(
                "INSERT INTO rec (user_id, post_id) VALUES (?, ?)",
                (user_id, post_id),
            )


if __name__ == "__main__":
    create_db()
    print_db_tables_summary()
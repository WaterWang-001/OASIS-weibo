import asyncio
import sqlite3
import re
import json
import logging
import time
from typing import Dict, List, Optional
from openai import AsyncOpenAI

class AttitudeAnnotator:
    def __init__(
        self, 
        api_key: str, 
        base_url: str, 
        attitude_columns: List[str],
        batch_size: int = 50, 
        concurrency_limit: int = 10
    ):
        """
        初始化标注器。

        参数:
            api_key (str): OpenAI 兼容 API 的密钥。
            base_url (str): API 的基本 URL。
            attitude_columns (List[str]): 要标注的态度列的列表。
            batch_size (int): 每次数据库 UPDATE 的批处理大小。
            concurrency_limit (int): 并发 API 请求的限制数。
        """
        if not api_key or api_key.startswith("sk-...") or len(api_key) < 20:
            raise ValueError("API_KEY 未设置或无效。请在 oasis_evaluation_fix.py 中设置它。")
        
        self.api_key = api_key
        self.base_url = base_url
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        
        self.attitude_columns = attitude_columns
        self.batch_size = batch_size
        self.api_semaphore = asyncio.Semaphore(concurrency_limit)
        
        # 预编译用于文本清理的正则表达式
        self.control_chars_regex = re.compile(r'[\x00-\x1F\x7F-\x9F\u200B-\u200D\uFEFF]')
        
        # 获取一个专用的 logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"AttitudeAnnotator initialized. Columns: {self.attitude_columns}")

    def _clean_text(self, text) -> str:
        """ (V3 Robust) 清理文本, 处理 bytes 和 unicode 错误 """
        if text is None: return ""
        if isinstance(text, bytes):
            try: text = text.decode('utf-8', errors='replace')
            except Exception: return ""
        if not isinstance(text, str):
            try: text = str(text)
            except Exception: return ""
        cleaned = self.control_chars_regex.sub(' ', text)
        return cleaned.strip()

    async def _get_attitude_scores_from_llm(self, content: str) -> Dict[str, float]:
        """ (私有) 调用 LLM (OpenAI) 分析帖子内容 """
        default_scores = {col: 0.0 for col in self.attitude_columns}
        if not content or not isinstance(content, str):
            return default_scores
        
        cleaned_content = self._clean_text(content)
        if not cleaned_content:
            return default_scores
        
        system_prompt = f"""
        You are a content analysis expert. Analyze the user's post.
        
        The post will be provided in one or both of the following forms:
        -   `[User Comment]`: The comment written by the user.
        -   `[Forwarded Original Post]`: The post that the user forwarded or quoted.

        == Your Core Task ==
        Your task is to analyze the sentiment of the **`[User Comment]`**.
        
        1.  **If `[User Comment]` exists:** All your sentiment analysis must be **based on `[User Comment]`**.
            -   `[Forwarded Original Post]` (if present) **is only for context**.
        2.  **If `[User Comment]` does not exist (i.e., the user only forwarded the original post, without commenting):**
            -   In this case, you should analyze the sentiment of the **`[Forwarded Original Post]`**.

        == Scoring and Categorization ==
        1.  **Topic Classification:** Determine the main topic of the analyzed text:
            -   `lifestyle_culture`, `sport_ent`, `sci_health`, `politics_econ`
        2.  **Sentiment Scoring (Continuous Score):** Use a continuous score from -1.0 to 1.0 (float) to precisely assess sentiment intensity.
        3.  **Output Format:**
            -   Assign scores to the matched topics, other topics should be 0.0.
            -   You must return only one JSON object, with keys {", ".join(self.attitude_columns)}.
        """

        try:
            async with self.api_semaphore:
                response = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": cleaned_content}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                json_text = response.choices[0].message.content
                parsed_json = json.loads(json_text)
                validated_scores = {}
                for col in self.attitude_columns:
                    value = parsed_json.get(col)
                    if not isinstance(value, (int, float)):
                        validated_scores[col] = 0.0
                    else:
                        validated_scores[col] = float(value)
                return validated_scores
        except Exception as e:
            self.logger.error(f"  -> LLM call failed: {e}")
            return default_scores

    def _setup_database_columns(self, conn: sqlite3.Connection, table_name: str):
        """ (私有) 检查并为表添加 attitude 列和标注标志 """
        self.logger.info(f"Setting up columns for table '{table_name}'...")
        cur = conn.cursor()
        all_columns_to_add = self.attitude_columns + ['attitude_annotated']
        
        for col in all_columns_to_add:
            col_type = "REAL DEFAULT 0.0" if col.startswith("attitude_") else "INTEGER DEFAULT 0"
            if col == "attitude_annotated":
                 col_type = "INTEGER DEFAULT 0"
            try:
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type}")
                self.logger.info(f"    - Added column: {col}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    self.logger.info(f"    - Column {col} already exists, skipping.")
                else:
                    raise e
        
        self.logger.info(f"Creating annotation index for '{table_name}'...")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_annotated ON {table_name} (attitude_annotated);")
        conn.commit()
        cur.close()
        self.logger.info(f"Column setup for '{table_name}' complete.")

    async def annotate_table(self, db_path: str, table_name: str, only_sim_posts: bool = True):
        """
        (公共方法) 标注单个表中所有 `attitude_annotated = 0` 的帖子。
        
        参数:
            db_path (str): 目标数据库的路径。
            table_name (str): 要标注的表名 (例如 "post")。
            only_sim_posts (bool): 
                True - (默认) 只标注 'created_at' 不是日期戳的帖子 (即模拟帖子)。
                False - 标注此表中所有未标注的帖子 (包括 GT 帖子)。
        """
        self.logger.info(f"--- 🚀 Starting Attitude Annotation for '{table_name}' in {db_path} ---")
        
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            
            # 1. 设置数据库列
            self._setup_database_columns(conn, table_name)
            
            # 2. 获取工作队列
            cur = conn.cursor()
            
            if only_sim_posts:
                self.logger.info("  -> Fetching IDs of (simulator-generated) posts that need annotation...")
                # (只获取 'created_at' 是整数时间步的帖子)
                query_sql = f"""
                    SELECT post_id FROM {table_name} 
                    WHERE attitude_annotated = 0 AND created_at NOT LIKE '%-%'
                    """
            else:
                self.logger.info("  -> Fetching IDs of ALL posts that need annotation...")
                # (获取所有未标注的帖子, 包括 GT 帖子)
                query_sql = f"SELECT post_id FROM {table_name} WHERE attitude_annotated = 0"

            cur.execute(query_sql)
            all_post_ids = [row[0] for row in cur.fetchall()]
            cur.close()

            total_to_process = len(all_post_ids)
            if total_to_process == 0:
                self.logger.info(f"  -> '{table_name}': No new posts found to annotate.")
                return
            
            self.logger.info(f"  -> '{table_name}': Found {total_to_process} posts to annotate.")
            total_processed = 0
            num_total_batches = (total_to_process // self.batch_size) + 1

            # 3. 分批处理
            for i in range(0, total_to_process, self.batch_size):
                batch_ids = all_post_ids[i : i + self.batch_size]
                total_batches = (i // self.batch_size) + 1
                self.logger.info(f"\n  -> Processing batch {total_batches} / {num_total_batches}...")

                placeholders = ','.join('?' for _ in batch_ids)
                select_sql = f"SELECT post_id, content, quote_content FROM {table_name} WHERE post_id IN ({placeholders})"
                
                batch_cur = conn.cursor()
                batch_cur.execute(select_sql, batch_ids)
                rows = batch_cur.fetchall()
                batch_cur.close()

                tasks, post_id_map = [], {}
                for post_id, content, quote_content in rows:
                    user_comment = self._clean_text(content)
                    original_post = self._clean_text(quote_content)
                    
                    text_to_annotate = ""
                    if user_comment:
                        text_to_annotate = f"[User Comment]\n{user_comment}"
                        if original_post:
                            text_to_annotate += f"\n\n[Forwarded Original Post]\n{original_post}"
                    elif original_post:
                        text_to_annotate = f"[Forwarded Original Post]\n{original_post}"
                    else:
                        continue 

                    tasks.append(self._get_attitude_scores_from_llm(text_to_annotate))
                    post_id_map[len(tasks) - 1] = post_id

                if not tasks: continue 

                try:
                    self.logger.info(f"  -> Calling LLM API for {len(tasks)} posts (in parallel)...")
                    api_start_time = time.time()
                    all_scores = await asyncio.gather(*tasks)
                    self.logger.info(f"  -> LLM batch processing completed in: {time.time() - api_start_time:.2f} seconds")
                except Exception as e:
                    self.logger.error(f"  -> ❌ Batch {total_batches} failed: {e}. Skipping this batch.")
                    continue

                # 4. 准备批量 UPDATE
                update_batch_data = []
                set_sql_parts = [f"{col} = ?" for col in self.attitude_columns]
                update_sql = f"UPDATE {table_name} SET {', '.join(set_sql_parts)}, attitude_annotated = 1 WHERE post_id = ?"
                
                for task_index, scores in enumerate(all_scores):
                    post_id = post_id_map[task_index]
                    scores_tuple = tuple(scores[col] for col in self.attitude_columns)
                    update_batch_data.append(scores_tuple + (post_id,))
                
                # 5. 执行批量 UPDATE
                if update_batch_data:
                    try:
                        write_cur = conn.cursor()
                        write_cur.executemany(update_sql, update_batch_data)
                        conn.commit()
                        total_processed += len(update_batch_data)
                        self.logger.info(f"  -> ...Processed and updated {total_processed} / {total_to_process} posts")
                    except sqlite3.Error as e:
                        self.logger.error(f"  -> ❌ Database COMMIT failed: {e}")
                        conn.rollback()
                    finally:
                        write_cur.close()

            self.logger.info(f"--- ✅ '{table_name}' annotation complete ---")

        except Exception as e:
            self.logger.error(f"  -> ❌ Annotation for '{table_name}' failed: {e}")
            import traceback
            traceback.print_exc()
            if conn: conn.rollback()
        finally:
            if conn: conn.close()
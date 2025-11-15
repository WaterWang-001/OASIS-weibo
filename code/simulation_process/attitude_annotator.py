import asyncio
import sqlite3
import re
import json
import logging
import time
from typing import Dict, List, Optional, Tuple
from openai import AsyncOpenAI

# ... logger 定义保持不变 ...
logger = logging.getLogger("AttitudeAnnotation")


# =====================================================================
# 1. 基类 (已重构并行逻辑)
# =====================================================================

# =====================================================================
# 1. 基类 (已添加超时和健壮性)
# =====================================================================

class BaseAttitudeAnnotator:
    """(基类) 态度标注器的共享逻辑。"""
    
    def __init__(
        self, 
        api_key: str, 
        base_url: Optional[str], 
        attitude_columns: List[str],
        concurrency_limit: int = 100, 
        log_interval_posts: int = 100,
        api_timeout_seconds: int = 30  # [新] 添加 API 超时
    ):
        """
        初始化基类。

        参数:
            ... (其他参数) ...
            api_timeout_seconds (int): 单个 API 请求的最大等待时间。
        """
        self.api_key = api_key
        self.base_url = base_url
        self.api_timeout_seconds = api_timeout_seconds # [新] 存储超时
        
        # [修改] 将超时传递给 OpenAI 客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key, 
            base_url=self.base_url,
            timeout=self.api_timeout_seconds
        )
        
        self.attitude_columns = attitude_columns
        self.api_semaphore = asyncio.Semaphore(concurrency_limit)
        self.log_interval_posts = max(1, log_interval_posts) 
        
        self.control_chars_regex = re.compile(r'[\x00-\x1F\x7F-\x9F\u200B-\u200D\uFEFF]')
        self.logger = logger
        self.logger.info(
            f"{self.__class__.__name__} initialized. "
            f"Columns: {self.attitude_columns} | "
            f"Concurrency: {concurrency_limit} | "
            f"Log Interval: {self.log_interval_posts} posts | "
            f"API Timeout: {self.api_timeout_seconds}s" # [新] 记录超时
        )

    # ... _clean_text, _get_system_prompt, _get_attitude_scores_from_llm, 
    #     _setup_database_columns ... 
    #     (这些方法保持不变)
    def _clean_text(self, text) -> str:
        if text is None: return ""
        if isinstance(text, bytes):
            try: text = text.decode('utf-8', errors='replace')
            except Exception: return ""
        if not isinstance(text, str):
            try: text = str(text)
            except Exception: return ""
        cleaned = self.control_chars_regex.sub(' ', text)
        return cleaned.strip()

    def _get_system_prompt(self) -> str:
        return f"""
        You are a content analysis expert. Analyze the user's post.
        
        The post will be provided in one or both of the following forms:
        -   `[User Comment]`: The comment written by the user.
        -   `[Forwarded Original Post]`: The post that the user forwarded or quoted.

        == Your Core Task ==
        Your task is to analyze the sentiment of the **`[User Comment]`**.
        
        1.  **If `[User Comment]` exists:** All your sentiment analysis must be **based on `[User Comment]`**.
            -   `[Forwarded Original Post]` (if present) **is only for context**.
        2.  **If `[User Comment]` does not exist (i.E., the user only forwarded the original post, without commenting):**
            -   In this case, you should analyze the sentiment of the **`[Forwarded Original Post]`**.

        == Scoring and Categorization ==
        1.  **Topic Classification:** Determine the main topic of the analyzed text:
            -   `lifestyle_culture`, `sport_ent`, `sci_health`, `politics_econ`
        2.  **Sentiment Scoring (Continuous Score):** Use a continuous score from -1.0 to 1.0 (float) to precisely assess sentiment intensity.
        3.  **Output Format:**
            -   Assign scores to the matched topics, other topics should be 0.0.
            -   The JSON keys must be: {", ".join(self.attitude_columns)}.
        """

    async def _get_attitude_scores_from_llm(self, content: str) -> Dict[str, float]:
        raise NotImplementedError("Subclass must implement the _get_attitude_scores_from_llm method.")


    def _setup_database_columns(self, conn: sqlite3.Connection, table_name: str):
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

    # [关键修改] _process_post 增加了 try...except
    async def _process_post(self, post_id: str, content: str, quote_content: str) -> Optional[Tuple[str, Dict[str, float]]]:
        """ 
        (私有) [修改] 处理单个帖子的完整工作流，添加健壮的异常处理。
        """
        try:
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
                return None 

            async with self.api_semaphore:
                # [关键] 如果此调用超过 self.api_timeout_seconds,
                # 它将引发异常，并被下面的 except 块捕获。
                scores = await self._get_attitude_scores_from_llm(text_to_annotate)
            
            return (post_id, scores)

        except Exception as e:
            # [关键] 捕获此任务中的任何错误（包括 API 超时）
            # 记录错误，然后返回 None，这样整个批处理不会崩溃
            self.logger.error(f"  -> ❌ Task failed for post_id {post_id}: {e}")
            return None 

    # [重构] annotate_table (此方法保持不变)
    async def annotate_table(self, db_path: str, table_name: str, only_sim_posts: bool = True):
        """
        (公共方法) [重构] 标注表中所有未处理的帖子。
        使用 as_completed 来实时监控进度。
        """
        self.logger.info(f"--- 🚀 Starting Attitude Annotation for '{table_name}' in {db_path} ---")
        
        conn = None
        all_posts_to_process: List[Tuple] = []
        
        try:
            conn = sqlite3.connect(db_path)
            
            # 1. 设置数据库列 (同步)
            self._setup_database_columns(conn, table_name)
            
            # --- 阶段 1: 批量读取 ---
            self.logger.info("  -> Phase 1: Fetching all posts to process...")
            cur = conn.cursor()
            
            base_query = f"SELECT post_id, content, quote_content FROM {table_name} WHERE attitude_annotated = 0"
            if only_sim_posts:
                self.logger.info("    (Fetching simulator-generated posts only)")
                query_sql = f"{base_query} AND created_at NOT LIKE '%-%'"
            else:
                self.logger.info("    (Fetching ALL posts)")
                query_sql = base_query

            cur.execute(query_sql)
            all_posts_to_process = cur.fetchall()
            cur.close()

            total_to_process = len(all_posts_to_process)
            if total_to_process == 0:
                self.logger.info(f"  -> '{table_name}': No new posts found to annotate.")
                return
            
            self.logger.info(f"  -> Phase 1: Found {total_to_process} posts to annotate.")

            # --- 阶段 2: 全速并行 API (使用 as_completed) ---
            self.logger.info(f"  -> Phase 2: Calling LLM API for {total_to_process} posts (Concurrency: {self.api_semaphore._value})...")
            
            tasks = []
            for post_id, content, quote_content in all_posts_to_process:
                tasks.append(
                    self._process_post(post_id, content, quote_content)
                )

            api_start_time = time.time()
            
            update_batch_data = [] # 准备用于数据库写入
            processed_count = 0
            failed_count = 0 # [新] 统计失败次数

            # [关键修改] 使用 as_completed 替换 gather
            for future in asyncio.as_completed(tasks):
                # 等待下一个完成的任务
                result = await future
                processed_count += 1
                
                # 处理结果 (将其添加到待写入列表)
                if result is not None:
                    post_id, scores = result
                    scores_tuple = tuple(scores.get(col, 0.0) for col in self.attitude_columns)
                    update_batch_data.append(scores_tuple + (post_id,))
                else:
                    # [新] 如果任务返回 None (即失败或超时)，则计数
                    failed_count += 1
                
                # [进度监控]
                # 每 N 个帖子打印一次日志，或者在最后一个帖子完成时打印
                if processed_count % self.log_interval_posts == 0 or processed_count == total_to_process:
                    percent_complete = (processed_count / total_to_process) * 100
                    elapsed_time = time.time() - api_start_time
                    posts_per_sec = processed_count / elapsed_time if elapsed_time > 0 else 0
                    
                    self.logger.info(
                        f"  -> Progress: {processed_count}/{total_to_process} "
                        f"({percent_complete:.1f}%) | "
                        f"Failed: {failed_count} | " # [新] 报告失败/超时次数
                        f"Speed: {posts_per_sec:.2f} posts/sec"
                    )

            api_time = time.time() - api_start_time
            self.logger.info(f"  -> Phase 2: LLM processing complete in: {api_time:.2f} seconds. Total Failed/Timeout: {failed_count}")

            # --- 阶段 3: 批量写入 ---
            self.logger.info("  -> Phase 3: Writing results to database...")
            
            if not update_batch_data:
                self.logger.info("  -> Phase 3: No valid results to write.")
                return
            
            total_processed = len(update_batch_data)
            write_cur = conn.cursor()
            try:
                set_sql_parts = [f"{col} = ?" for col in self.attitude_columns]
                update_sql = f"UPDATE {table_name} SET {', '.join(set_sql_parts)}, attitude_annotated = 1 WHERE post_id = ?"
                
                write_cur.executemany(update_sql, update_batch_data)
                conn.commit() # [关键] 只 Commit 一次！
                self.logger.info(f"  -> Phase 3: Successfully processed and updated {total_processed} posts.")
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
# =====================================================================
# 2. vLLM (开源) 子类
# =====================================================================

# =====================================================================
# 2. vLLM (开源) 子类
# =====================================================================

class VLLMAttitudeAnnotator(BaseAttitudeAnnotator):
    """(子类) 使用 vLLM (本地/开源) 兼容 API 进行标注。"""
    
    def __init__(
        self, 
        model_name: str, 
        attitude_columns: List[str],
        base_url: str = "http://localhost:8000/v1",  
        api_key: str = "vllm",  
        concurrency_limit: int = 100,
        log_interval_posts: int = 100,
        api_timeout_seconds: int = 30  # [新]
    ):
        """
        初始化 vLLM 标注器。
        """
        self.model_name = model_name
        # [修改] 调用基类的 __init__，传入所有参数
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            attitude_columns=attitude_columns,
            concurrency_limit=concurrency_limit,
            log_interval_posts=log_interval_posts,
            api_timeout_seconds=api_timeout_seconds # [新]
        )
        self.logger.info(f"VLLM Annotator using model: {self.model_name}")

    # ... _get_attitude_scores_from_llm 方法保持不变 ...
    async def _get_attitude_scores_from_llm(self, content: str) -> Dict[str, float]:
        # (此方法无需修改)
        default_scores = {col: 0.0 for col in self.attitude_columns}
        if not content or not isinstance(content, str):
            return default_scores
        
        cleaned_content = self._clean_text(content)
        if not cleaned_content:
            return default_scores
        
        system_prompt = self._get_system_prompt()
        system_prompt += "\n\nYou must return **only a single JSON object** and nothing else."
        
        json_text = "" 
        try:
            # (此调用现在受 self.api_timeout_seconds 限制)
            async with self.api_semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": cleaned_content}
                    ],
                    temperature=0.0
                )
            json_text = response.choices[0].message.content
            
            try:
                start_index = json_text.index("{")
                end_index = json_text.rindex("}")
                json_text = json_text[start_index : end_index + 1]
            except ValueError:
                self.logger.warning(f"  -> Could not find '{{' or '}}' in response. Trying to parse anyway.")
            
            parsed_json = json.loads(json_text)
            validated_scores = {}
            for col in self.attitude_columns:
                value = parsed_json.get(col)
                if not isinstance(value, (int, float)):
                    self.logger.warning(f"  -> Invalid data type for key '{col}'. Got: {value}. Defaulting to 0.0")
                    validated_scores[col] = 0.0
                else:
                    validated_scores[col] = float(value)
            return validated_scores
        except Exception as e:
            # [注意]：这里的 except 块现在不太可能被触发，
            # 因为超时等错误会在 _process_post 中被捕获。
            # 但保留它以防 JSON 解析等错误。
            self.logger.error(f"  -> LLM JSON parsing failed: {e}. Raw response: '{json_text}'")
            return default_scores

# =====================================================================
# 3. OpenAI (闭源) 子类
# =====================================================================

class OpenAIAttitudeAnnotator(BaseAttitudeAnnotator):
    """(子类) 使用 OpenAI (闭源) 兼容 API (如 gpt-4o-mini) 进行标注。"""

    def __init__(
        self, 
        model_name: str, 
        api_key: str,
        attitude_columns: List[str],
        base_url: Optional[str] = None,  
        concurrency_limit: int = 100,
        log_interval_posts: int = 100,
        api_timeout_seconds: int = 30  # [新]
    ):
        """
        初始化 OpenAI 标注器。
        """
        if not api_key or not api_key.startswith("sk-"):
            raise ValueError("有效的 OpenAI API_KEY (sk-...) 未提供。")
            
        self.model_name = model_name
        # [修改] 调用基类的 __init__，传入所有参数
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            attitude_columns=attitude_columns,
            concurrency_limit=concurrency_limit,
            log_interval_posts=log_interval_posts,
            api_timeout_seconds=api_timeout_seconds # [新]
        )
        self.logger.info(f"OpenAI Annotator using model: {self.model_name}")

    # ... _get_attitude_scores_from_llm 方法保持不变 ...
    async def _get_attitude_scores_from_llm(self, content: str) -> Dict[str, float]:
        # (此方法无需修改)
        default_scores = {col: 0.0 for col in self.attitude_columns}
        if not content or not isinstance(content, str):
            return default_scores
        
        cleaned_content = self._clean_text(content)
        if not cleaned_content:
            return default_scores
        
        system_prompt = self._get_system_prompt()

        try:
            # (此调用现在受 self.api_timeout_seconds 限制)
            async with self.api_semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model_name, 
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
                    self.logger.warning(f"  -> Invalid data type for key '{col}'. Got: {value}. Defaulting to 0.0")
                    validated_scores[col] = 0.0
                else:
                    validated_scores[col] = float(value)
            return validated_scores
        except Exception as e:
            self.logger.error(f"  -> LLM call/parse failed: {e}")
            return default_scores
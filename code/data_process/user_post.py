import json
import collections
import sys
from pathlib import Path
import os
import gc
import sqlite3  # 引入 sqlite
import time

# --- 请修改这里 ---
# 输入目录：包含所有原始 .txt 文件
INPUT_DIRECTORY = 'data/raw/'
# 输出文件：保存处理后的用户-内容数据（帖子+评论）

# 临时数据库文件
PERMANENT_DB_FILE = 'data/user_post/user_post_database.db'
# --------------------



class UserPostProcessor:
    """
    将原始 .txt 文件流式写入 SQLite 的封装类。
    使用示例:
        proc = UserPostProcessor(input_directory='data/raw/', db_path='data/user_post/user_post_database.db')
        proc.run()
    """
    def __init__(self, input_directory=INPUT_DIRECTORY, db_path=PERMANENT_DB_FILE):
        self.input_directory = Path(input_directory)
        self.db_path = db_path

    def collect_file_list(self):
        if not self.input_directory.is_dir():
            raise FileNotFoundError(f"输入目录不存在: {self.input_directory}")
        file_list = list(self.input_directory.glob('*.txt'))
        if not file_list:
            raise FileNotFoundError(f"在目录 '{self.input_directory}' 中找不到任何 .txt 文件。")
        return file_list
    @staticmethod
    def get_user_id(pojo):
        """从POJO中安全地提取用户ID"""
        if not pojo:
            return None
        return pojo.get('sjcjId')

    @staticmethod
    def get_post_timestamp(pojo):
        """从 contentPojo 中安全地提取发布时间戳"""
        if not pojo:
            return None
        return pojo.get('sjcjPublished')

    @staticmethod
    def get_comment_timestamp(pojo):
        """从 commentPojo 中安全地提取发布时间戳"""
        if not pojo:
            return None
        return pojo.get('sjcjPublished')

    # --- 重写 Pass 1 ---

    def process_and_store_to_db(self,file_list, db_path):
        """
        Pass 1 (重写): 遍历所有文件，将数据流式存入 SQLite 数据库
        """
        print(f"--- 🚀 Pass 1 (DB): 正在将内容流式传输到临时数据库... ---")
        
        total_line_count = 0
        total_error_count = 0
        post_count = 0
        comment_count = 0
        
        # BATCH_SIZE：一次性插入 N 条数据，速度更快
        BATCH_SIZE = 50000 
        insert_batch = []

        # 如果临时数据库已存在，先删除
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"  -> 已删除旧的临时数据库: {db_path}")

        # 连接数据库并创建表
        # isolation_level=None (自动提交) 和 journal_mode='WAL' (预写日志) 是为了提高写入性能
        conn = sqlite3.connect(db_path, isolation_level=None)
        cur = conn.cursor()
        
        try:
            # 优化1: 设置高性能 pragma
            cur.execute("PRAGMA journal_mode = WAL;")
            cur.execute("PRAGMA synchronous = NORMAL;")
            
            # 优化2: 创建表
            cur.execute("""
            CREATE TABLE IF NOT EXISTS content (
                user_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                data_json TEXT NOT NULL
            );
            """)
            
            start_time = time.time()

            for filepath in file_list:
                print(f"  -> 正在处理: {filepath.name}")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            total_line_count += 1
                            try:
                                data = json.loads(line.strip())
                                
                                author_content_pojo = data.get('authorContentPojo')
                                author_comment_pojo = data.get('authorCommentPojo')
                                content_pojo = data.get('contentPojo', {})
                                comment_pojo = data.get('commentPojo', {})
                                
                                user_id = None
                                timestamp = None
                                
                                if author_content_pojo and not author_comment_pojo:
                                    user_id = self.get_user_id(author_content_pojo)
                                    timestamp = self.get_post_timestamp(content_pojo)
                                    if user_id and timestamp is not None:
                                        post_count += 1
                                        
                                elif author_comment_pojo:
                                    user_id = self.get_user_id(author_comment_pojo)
                                    timestamp = self.get_comment_timestamp(comment_pojo)
                                    if user_id and timestamp is not None:
                                        comment_count += 1
                                
                                if user_id and timestamp is not None:
                                    # 优化3: 序列化 data 对象，而不是原始 line
                                    # 这样下游就不需要再次 json.loads(row[0]) 了
                                    insert_batch.append(
                                        (user_id, timestamp, json.dumps(data, ensure_ascii=False))
                                    )
                                
                                # 优化4: 批量插入
                                if len(insert_batch) >= BATCH_SIZE:
                                    cur.executemany(
                                        "INSERT INTO content (user_id, timestamp, data_json) VALUES (?, ?, ?)",
                                        insert_batch
                                    )
                                    insert_batch = [] # 清空批次
                                    
                            except (json.JSONDecodeError, Exception) as e:
                                if line_num % 10000 == 0: # 不要打印太多错误
                                    print(f"⚠️ 文件 {filepath.name} Line {line_num}: 处理时发生错误: {e}", file=sys.stderr)
                                total_error_count += 1
                                continue
                except Exception as e:
                    print(f"❌ 错误: 无法读取文件 {filepath.name}. 错误: {e}", file=sys.stderr)
                    total_error_count += 1
            
            # 插入最后一批剩余数据
            if insert_batch:
                cur.executemany(
                    "INSERT INTO content (user_id, timestamp, data_json) VALUES (?, ?, ?)",
                    insert_batch
                )
            
            end_time = time.time()
            print(f"Pass 1 完成: 共处理 {total_line_count} 行, {total_error_count} 行解析/处理失败。")
            print(f"  -> 耗时: {end_time - start_time:.2f} 秒")
            print(f"  -> 共 {post_count + comment_count} 条内容存入数据库 {db_path}")

            # 优化5: 创建索引！
            # 这是最关键的一步，它会让 Pass 2 的 ORDER BY 变得飞快
            print("  -> 正在为数据库创建索引 (user_id, timestamp)... 这可能需要几分钟...")
            index_start_time = time.time()
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_ts ON content (user_id, timestamp);")
            index_end_time = time.time()
            print(f"  -> 索引创建完成! 耗时: {index_end_time - index_start_time:.2f} 秒")

        except Exception as e:
            print(f"❌ Pass 1 发生致命错误: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        finally:
            conn.close() 

    def run(self):
        file_list = self.collect_file_list()
        self.process_and_store_to_db(file_list, self.db_path)
        gc.collect()
        return self.db_path

# 修改 main 以使用类
def main():
    input_dir = INPUT_DIRECTORY
    proc = UserPostProcessor(input_directory=input_dir, db_path=PERMANENT_DB_FILE)
    try:
        proc.run()
        print("\n🎉 全部处理完成。")
        print(f"✅ 最终数据库已成功保存到: {PERMANENT_DB_FILE}")
    except Exception as e:
        print(f"\n❌ 发生未知错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

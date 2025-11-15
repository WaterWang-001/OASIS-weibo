import os
import sqlite3
import logging
from typing import List

def reset_simulation_tables(
    db_path: str, 
    tables_to_keep: List[str], 
    logger: logging.Logger
):
    """
    重置OASIS数据库，删除所有模拟结果表，但保留指定的核心数据表。
    
    如果数据库文件不存在，它只会记录一条信息，因为OASIS的
    'make' 流程稍后会自动创建它。

    参数:
        db_path (str): 数据库文件路径。
        tables_to_keep (List[str]): 不应被删除的表名列表。
        logger (logging.Logger): 用于记录操作的日志记录器实例。
    """
    if os.path.exists(db_path):
        logger.warning(f"数据库 {db_path} 已存在。将重置表，但保留: {', '.join(tables_to_keep)}")
        
        conn = None # 初始化 conn
        try:
            # 1. 连接到数据库
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 2. 获取所有表的列表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            all_tables = [row[0] for row in cursor.fetchall()]
            
            tables_to_drop = []
            
            # 3. 找出所有需要删除的表
            for table_name in all_tables:
                if table_name not in tables_to_keep:
                    tables_to_drop.append(table_name)

            # 4. 逐个删除这些表
            if tables_to_drop:
                logger.warning(f"将删除以下模拟结果表: {', '.join(tables_to_drop)}")
                for table_name in tables_to_drop:
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.commit()
                logger.info("数据库重置完成。")
            else:
                logger.info("没有找到需要重置的模拟结果表。")
                
        except sqlite3.Error as e:
            logger.error(f"重置数据库时出错: {e}")
        finally:
            if conn:
                conn.close()
                
    else:
        logger.info(f"数据库 {db_path} 不存在，将(在env.make中)创建新库。")
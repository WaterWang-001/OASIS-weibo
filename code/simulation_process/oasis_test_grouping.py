import asyncio
import os
import logging
import ast
import random 
from datetime import datetime
from collections import defaultdict
from typing import List, Set, Dict, Any, Iterable, Tuple, Optional
import sqlite3
import pandas as pd
from tqdm import tqdm
import numpy as np 
from import Dict, Optional
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.models import VLLMModel, DeepSeekModel

from attitude_annotator import AttitudeAnnotator

import oasis
from oasis import (ActionType, LLMAction, ManualAction)
# 【!! 关键 !!】 我们现在只导入 "重" 的 graph_generator
from oasis.social_agent.agents_generator import (
    generate_twitter_agent_graph
)
from oasis.social_agent import AgentGraph
from oasis.social_platform.config import UserInfo
from oasis.social_platform import Platform


# Tier 1: "重" LLM Agents (初始化慢, 运行慢)
TIER_1_LLM_GROUPS = {
    "权威媒体/大V",
    "活跃KOL",
    "活跃创作者",
    "普通用户" 
}

# Tier 2: "轻" ABM Agents (初始化快, 运行快)
TIER_2_HEURISTIC_GROUPS = {
    "潜水用户"
}
#时间为：2025-06-02 16:30:00
CALIBRATION_END= "2025-06-02T16:30:00"

async def log_agent_attitudes(
    env: oasis.OasisEnv, 
    db_path: str, 
    current_step: int, 
    attitude_columns: List[str]
):
    """
    [重写] 在每个时间步结束时，记录 *每个* agent 的态度。
    - ABM Agent: 记录其 'internal_state' (来自 .attitude_scores)
    - LLM Agent: 记录其 'external_expression' (来自其 *当前时间步* 帖子的平均分)
    
    假设的表结构 (例如 'log_attitude_lifestyle_culture'):
    CREATE TABLE ... (
        time_step INTEGER,
        user_id TEXT,
        agent_id INTEGER,
        agent_type TEXT,
        metric_type TEXT,
        attitude_score REAL
    );
    """
    logger = logging.getLogger("AttitudeLogger")
    logger.info(f"[Step {current_step}] 正在记录 *每个* Agent 的态度...")
    
    all_agents = list(env.agent_graph.get_agents())
    
    # 准备一个列表来批量插入
    # 格式: (table_name, time_step, user_id, agent_id, agent_type, metric_type, score)
    batch_insert_data = []

    # --- 1. 处理 ABM (Tier 2) - 内部状态 ---
    # (这部分在内存中完成，不需要数据库)
    abm_agent_count = 0
    for agent_id, agent in all_agents:
        if agent.group in TIER_2_HEURISTIC_GROUPS:
            if hasattr(agent, 'attitude_scores') and isinstance(agent.attitude_scores, dict):
                abm_agent_count += 1
                
                # 复制分数并计算总平均分
                scores_dict = agent.attitude_scores.copy()
                valid_scores = [scores_dict.get(col, 0.0) for col in attitude_columns if scores_dict.get(col) is not None]
                scores_dict['attitude_average'] = np.mean(valid_scores) if valid_scores else 0.0
                
                # [!! 修改: 捕获两个 ID !!]
                agent_sim_id = agent.agent_id # (e.g., 1001)
                user_id_str = agent.user_info.profile["other_info"].get("original_user_id") # (e.g., '1618051664')
                
                # 为该 agent 的 5 个维度准备插入数据
                for dim_name, score_value in scores_dict.items():
                    table_name = f"log_{dim_name}"
                    batch_insert_data.append((
                        table_name,
                        current_step,
                        user_id_str,       # <-- user_id
                        agent_sim_id,      # <-- agent_id
                        'ABM',
                        'internal_state',
                        score_value
                    ))

    # --- 2. 处理 LLM (Tier 1) - 外部表现 ---
    llm_agent_ids = {agent.agent_id for agent_id, agent in all_agents if agent.group in TIER_1_LLM_GROUPS}
    llm_agent_count = 0
    
    if llm_agent_ids:
        try:
            # (使用只读模式查询)
            with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
                id_placeholders = ", ".join(["?"] * len(llm_agent_ids))
                avg_cols_sql = ", ".join([f"AVG({col})" for col in attitude_columns])
                
                # 关键查询:
                # [!! 修改: 查询 agent_id (整数) 并选择 user_id 和 agent_id !!]
                query = f"""
                SELECT 
                    user_id, 
                    agent_id,
                    {avg_cols_sql}
                FROM post
                WHERE created_at = ?                     -- 匹配当前时间步
                  AND agent_id IN ({id_placeholders})   -- 匹配 LLM Agent ID (整数)
                  AND attitude_annotated = 1           -- 必须已标注
                GROUP BY user_id, agent_id               -- <-- 按两个 ID 分组
                """
                params = (current_step, *list(llm_agent_ids))
                
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                llm_agent_count = len(rows) # 记录实际发帖的 agent 数量
                
                # 遍历 *发了帖* 的 LLM agents
                for row in rows:
                    # [!! 修改: 提取两个 ID !!]
                    user_id_str = str(row[0])
                    agent_sim_id = int(row[1])
                    llm_avgs_list = list(row[2:])
                    
                    # (A) 4个维度的平均值
                    scores_dict = {col: llm_avgs_list[i] for i, col in enumerate(attitude_columns)}
                    
                    # (B) 总平均值
                    valid_avgs = [x for x in llm_avgs_list if x is not None]
                    scores_dict['attitude_average'] = np.mean(valid_avgs) if valid_avgs else 0.0
                    
                    # 为该 agent 的 5 个维度准备插入数据
                    for dim_name, score_value in scores_dict.items():
                        table_name = f"log_{dim_name}"
                        batch_insert_data.append((
                            table_name,
                            current_step,
                            user_id_str,    # <-- user_id
                            agent_sim_id,   # <-- agent_id
                            'LLM',
                            'external_expression',
                            score_value
                        ))
        except sqlite3.Error as e:
            logger.error(f"[Step {current_step}] 查询 LLM 帖子分数时出错: {e}")
        except Exception as e:
            logger.error(f"[Step {current_step}] 处理 LLM 分数时意外出错: {e}", exc_info=True)

    # --- 3. 批量写入数据库 ---
    if not batch_insert_data:
        logger.info(f"[Step {current_step}] 没有新的态度分数需要记录。")
        return

    inserted_count = 0
    try:
        # (使用写模式连接)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 遍历所有准备好的数据
            # [!! 修改: 解包两个 ID !!]
            for (table_name, ts, user_id, agent_id, agent_type, metric_type, score) in batch_insert_data:
                try:
                    # [!! 修改: 插入两个 ID !!]
                    cursor.execute(
                        f"""
                        INSERT INTO {table_name} (
                            time_step, user_id, agent_id, agent_type, metric_type, attitude_score
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (ts, user_id, agent_id, agent_type, metric_type, score)
                    )
                    inserted_count += 1
                except sqlite3.Error as e:
                    logger.error(f"[Step {current_step}] 写入日志表 '{table_name}' 失败 (Agent {agent_id}): {e}. (请确保该表已创建)")
                    # (继续尝试写入其他条目)
            
            conn.commit() # 提交事务
            logger.info(f"[Step {current_step}] 成功记录 {abm_agent_count} 个 ABM agents 和 {llm_agent_count} 个 LLM agents (共 {inserted_count} 条分数)。")
            
    except sqlite3.Error as e:
        logger.error(f"[Step {current_step}] 批量写入态度日志时数据库出错: {e}")
    except Exception as e:
        logger.error(f"[Step {current_step}] 批量写入态度日志时意外出错: {e}", exc_info=True)
# --- [!! 函数重写结束 !!] ---


async def main():
    # --- (日志配置) ---
    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = f"{log_dir}/oasis_test_{current_time}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"日志将保存到: {log_file_path}")
    logger.info("正在初始化模型...")
    # --- (配置结束) ---
   
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type="gpt-4o-mini",
        url='https://api.nuwaapi.com/v1',
        api_key='sk-tsmw9XQGmKWqE1CvpPCOG2YpLgnYdGisi54GVU0Lf0GFW9rN',
    )
    logger.info("模型初始化完毕。")
    
    # --- 新增：AttitudeAnnotator 配置与初始化 ---
    ATTITUDE_COLUMNS = [
        'attitude_lifestyle_culture',
        'attitude_sport_ent',
        'attitude_sci_health',
        'attitude_politics_econ'
    ]
    ANNOTATOR_API_KEY = 'sk-tsmw9XQGmKWqE1CvpPCOG2YpLgnYdGisi54GVU0Lf0GFW9rN'  # 可改为从 env 读取
    ANNOTATOR_BASE_URL = 'https://api.nuwaapi.com/v1'
    ANNOTATOR_BATCH_SIZE = 200
    ANNOTATOR_CONCURRENCY = 50

    logger.info("正在初始化 AttitudeAnnotator...")
    annotator = AttitudeAnnotator(
        api_key=ANNOTATOR_API_KEY,
        base_url=ANNOTATOR_BASE_URL,
        attitude_columns=ATTITUDE_COLUMNS,
        batch_size=ANNOTATOR_BATCH_SIZE,
        concurrency_limit=ANNOTATOR_CONCURRENCY
    )
    logger.info("AttitudeAnnotator 初始化完毕。")
    # --- (初始化结束) ---
    
    available_actions = [
        ActionType.CREATE_POST,
        ActionType.LIKE_POST,
        ActionType.REPOST,
        ActionType.FOLLOW,
        ActionType.DO_NOTHING,
        ActionType.QUOTE_POST
    ]

    profile_path = "data/oasis/oasis_agent_init_3000_random.csv" 
    db_path = "data/oasis/oasis_database_3000_random.db" 
    
    
    # 1. (慢速) 在内存中构建 Agent Graph
    logger.info(f"正在从 {profile_path} 构建 agent graph...")
    agent_graph = await generate_twitter_agent_graph(
        profile_path=profile_path,
        model=model,
        available_actions=available_actions,
        db_path=db_path
    )
    logger.info(f"Agent graph 构建完毕, 共 {agent_graph.get_num_nodes()} 个 agents (T1+T2)。")


    tables_to_keep = [
        'post', 
        'ground_truth_post', 
        'sqlite_sequence',
        'log_attitude_lifestyle_culture',
        'log_attitude_sport_ent',
        'log_attitude_sci_health',
        'log_attitude_politics_econ',
        'log_attitude_average'
    ]

    if os.path.exists(db_path):
        logger.warning(f"数据库 {db_path} 已存在。将重置表，但保留 'post' 和 'ground_truth_post' 及 'log_attitude_...' 表。")
        
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
                    # (现在会安全地跳过所有 'tables_to_keep' 列表中的表)
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
        logger.info(f"数据库 {db_path} 不存在，将创建新库。")

    # 3. (快速) 创建环境
    logger.info("正在创建 Oasis 环境 (oasis.make)...")
    env = oasis.make(
            agent_graph=agent_graph, 
            platform=oasis.DefaultPlatformType.TWITTER,
            database_path=db_path,
            calibration_end=CALIBRATION_END
    )


    logger.info("正在执行环境重置 (env.reset)...")
    await env.reset()
    logger.info("环境重置完毕。")
    
    # --- [!! 删除: 对 initialize_log_table 的调用 !!] ---
    # (已删除)
    
    # 基础激活率
    TIER_1_ACTIVATION_RATES = {
        "权威媒体/大V": 0.8,
        "活跃KOL": 0.7,
        "活跃创作者": 0.6,
        "普通用户": 0.3, 
    }
    TIER_2_ACTIVATION_RATES = {
        "潜水用户": 0.1, 
    }
    
    # [!! 修改: 运行 5 个 step !!]
    total_steps = 5
    for step in range(total_steps):
        current_step = step + 1 # (从 1 开始计数)
        logger.info(f"--- 🚀 Simulation Step {current_step} / {total_steps} ---")
        
        # --- 1. 动态激活器 (Dynamic Activator) ---
        llm_agents_to_run = [] 
        heuristic_agents_to_run = [] 
        
        total_active_pool = env.agent_graph.get_agents()
        
        for agent_id, agent in total_active_pool:
            group = agent.group # (已在 BaseAgent 中设置)
            
            if group in TIER_1_LLM_GROUPS:
                activation_chance = TIER_1_ACTIVATION_RATES.get(group, 0.0)
                if random.random() < activation_chance:
                    llm_agents_to_run.append(agent) # 添加 LLM Agent
            
            elif group in TIER_2_HEURISTIC_GROUPS:
                activation_chance = TIER_2_ACTIVATION_RATES.get(group, 0.0)
                if random.random() < activation_chance:
                    heuristic_agents_to_run.append(agent) # 添加 Heuristic Agent

        logger.info(f"动态激活器: {len(llm_agents_to_run)} 个 LLM Agents, {len(heuristic_agents_to_run)} 个 Heuristic Agents 被激活。")

        # --- 2. 构建 action 字典 (仅用于 LLM Agents) ---
        llm_action = {
            agent: LLMAction()
            for agent in llm_agents_to_run
        }

        # --- 3. 执行 Step (LLM Agents) ---
        if llm_agents_to_run:
            logger.info(f"即将为 {len(llm_action)} 个 agents (LLM) 执行 actions...")
            await env.step(llm_action)
            
        # --- 4. 【!! 修正: 手动调用 Heuristic Agents !!】 ---
        if heuristic_agents_to_run:
            logger.info(f"即将为 {len(heuristic_agents_to_run)} 个 Heuristic agents 执行 .step()...")
            heuristic_tasks = [agent.step() for agent in heuristic_agents_to_run]
            await asyncio.gather(*heuristic_tasks)
        
        # --- 5. Attitude 标注 (异步) ---
        try:
            logger.info(f"--- 🛠️ Maintenance Phase (after step {current_step}) - Attitude annotation ---")
            logger.info("... 正在标注 'post' 表中的新帖子 (only_sim_posts=True)...")
            await annotator.annotate_table(db_path, "post", only_sim_posts=True)
            logger.info("... 正在标注 'ground_truth_post' 表中的新帖子 (only_sim_posts=False)...")
            await annotator.annotate_table(db_path, "ground_truth_post", only_sim_posts=False)
            logger.info("--- ✅ Attitude annotation completed ---")
        except Exception as e:
            logger.error(f"Attitude 标注失败: {e}", exc_info=True)
        
        # --- 6. [!! 新增: 记录 Agent 态度日志 !!] ---
        # (调用已被重写的新函数)
        await log_agent_attitudes(env, db_path, current_step, ATTITUDE_COLUMNS)
        # --- [!! 新增结束 !!] ---
            
    await env.close()
    logger.info("--- Simulation Finished ---")
        


if __name__ == "__main__":
    asyncio.run(main())
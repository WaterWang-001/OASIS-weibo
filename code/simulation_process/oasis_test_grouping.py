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
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.models import VLLMModel, DeepSeekModel

from attitude_annotator import OpenAIAttitudeAnnotator, VLLMAttitudeAnnotator

import oasis
from oasis import (ActionType, LLMAction, ManualAction, HeuristicAction)

from oasis.social_agent.agents_generator import (
    generate_twitter_agent_graph
)
from oasis.social_agent import AgentGraph
from oasis.social_platform.config import UserInfo
from oasis.social_platform import Platform
from oasis import EnvAction


from attitude_logger import SimulationAttitudeLogger
from db_manager import reset_simulation_tables

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
   
    # model = ModelFactory.create(...)
    model= VLLMModel(
        model_type="/remote-home/JuelinW/oasis_project/Qwen2.5-7B-Instruct",
        model_config_dict={
            "temperature": 0.5
        }
    )
    logger.info("模型初始化完毕。")
    
    # --- AttitudeAnnotator 配置 ---
    ATTITUDE_COLUMNS = [
        'attitude_lifestyle_culture',
        'attitude_sport_ent',
        'attitude_sci_health',
        'attitude_politics_econ'
    ]
    

    logger.info("正在初始化 AttitudeAnnotator...")
    # annotator = OpenAIAttitudeAnnotator(...)
    annotator = VLLMAttitudeAnnotator(
        model_type="/remote-home/JuelinW/oasis_project/Qwen2.5-7B-Instruct",
        attitude_columns=ATTITUDE_COLUMNS,
        concurrency_limit=200
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
    intervention_file_path = "data/oasis/intervention_messages.csv" # <-- 您要加载的干预文件
  
    # 1. 在内存中构建 Agent Graph
    logger.info(f"正在从 {profile_path} 构建 agent graph...")
    agent_graph = await generate_twitter_agent_graph(
        profile_path=profile_path,
        model=model,
        available_actions=available_actions,
        db_path=db_path
    )
    logger.info(f"Agent graph 构建完毕, 共 {agent_graph.get_num_nodes()} 个 agents (T1+T2)。")

    # 2. 重置数据库, 删除模拟结果表, 保留核心数据表
    tables_to_keep = [
        'post', 
        'ground_truth_post', 
        'sqlite_sequence'
    ]


    # 调用外部函数
    reset_simulation_tables(db_path, tables_to_keep, logger)

    # 3. (快速) 创建环境
    logger.info("正在创建 Oasis 环境 (oasis.make)...")
    # --- [!! 修改: 传递干预文件路径 !!] ---
    env = oasis.make(
            agent_graph=agent_graph, 
            platform=oasis.DefaultPlatformType.TWITTER,
            database_path=db_path,
            calibration_end=CALIBRATION_END,
            intervention_file_path=intervention_file_path # <-- 传递路径
    )
    # --- [!! 修改结束 !!] ---

    logger.info("正在执行环境重置 (env.reset)...")
    await env.reset()
    logger.info("环境重置完毕。")
    
    # --- [!! 新增: 初始化 AttitudeLogger !!] ---
    logger.info("正在初始化 SimulationAttitudeLogger...")
    attitude_logger = SimulationAttitudeLogger(
        db_path=db_path,
        attitude_columns=ATTITUDE_COLUMNS,
        tier_1_groups=TIER_1_LLM_GROUPS,
        tier_2_groups=TIER_2_HEURISTIC_GROUPS
    )
    # --- [!! 新增结束 !!] ---
    
    
    
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
    
    total_steps = 5
    for step in range(total_steps):
        current_step = step + 1 # (从 1 开始计数)
        logger.info(f"--- 🚀 Simulation Step {current_step} / {total_steps} ---")
        
        # --- 1. 动态激活器 (Dynamic Activator) ---
        llm_agents_to_run = [] 
        heuristic_agents_to_run = [] 
        total_active_pool = env.agent_graph.get_agents()
        for agent_id, agent in total_active_pool:
            group = agent.group
            if group in TIER_1_LLM_GROUPS:
                activation_chance = TIER_1_ACTIVATION_RATES.get(group, 0.0)
                if random.random() < activation_chance:
                    llm_agents_to_run.append(agent)
            elif group in TIER_2_HEURISTIC_GROUPS:
                activation_chance = TIER_2_ACTIVATION_RATES.get(group, 0.0)
                if random.random() < activation_chance:
                    heuristic_agents_to_run.append(agent)
        logger.info(f"动态激活器: {len(llm_agents_to_run)} 个 LLM Agents, {len(heuristic_agents_to_run)} 个 Heuristic Agents 被激活。")

        # --- 2. 构建 action 字典 ---
        all_actions = {}
        all_actions.update({
            agent: LLMAction()
            for agent in llm_agents_to_run
        })
        all_actions.update({
            agent: HeuristicAction()
            for agent in heuristic_agents_to_run
        })

        # --- 3. 执行 Step ---
        if all_actions:
            logger.info(f"即将为 {len(all_actions)} 个 agents统一执行 actions...")
            await env.step(all_actions)
        
        # --- 4. Attitude 标注 (异步) ---
        try:
            logger.info(f"--- 🛠️ Maintenance Phase (after step {current_step}) - Attitude annotation ---")
            logger.info("... 正在标注 'post' 表中的新帖子 (only_sim_posts=True)...")
            await annotator.annotate_table(db_path, "post", only_sim_posts=True)
            logger.info("--- ✅ 'post' 表标注完成 ---")
        except Exception as e:
            logger.error(f"Attitude 标注失败: {e}", exc_info=True)
        
        await attitude_logger.log_step_attitudes(env, current_step)

            
    await env.close()
    logger.info("--- Simulation Finished ---")
        

if __name__ == "__main__":
    asyncio.run(main())
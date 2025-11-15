import logging
import sqlite3
from typing import List, Set, Dict
import numpy as np

import oasis 
from oasis.environment.env import OasisEnv 

# --- 定义Tiers, 确保与 test.py 一致 ---
# (或者, 您可以将这些 Tiers 定义在一个共享的 config.py 文件中)
TIER_1_LLM_GROUPS = {
    "权威媒体/大V",
    "活跃KOL",
    "活跃创作者",
    "普通用户" 
}
TIER_2_HEURISTIC_GROUPS = {
    "潜水用户"
}


class SimulationAttitudeLogger:
    """
    一个专门用于在OASIS模拟的每个步骤中记录
    ABM (内部) 和 LLM (外部) 智能体态度的类。
    """
    
    def __init__(
        self, 
        db_path: str, 
        attitude_columns: List[str],
        tier_1_groups: Set[str] = TIER_1_LLM_GROUPS,
        tier_2_groups: Set[str] = TIER_2_HEURISTIC_GROUPS
    ):
        """
        初始化日志记录器。

        参数:
            db_path (str): 要写入日志的OASIS数据库路径。
            attitude_columns (List[str]): 4个态度维度的列名。
            tier_1_groups (Set[str]): 识别为 LLM Agents 的组名。
            tier_2_groups (Set[str]): 识别为 ABM/Heuristic Agents 的组名。
        """
        self.db_path = db_path
        self.attitude_columns = attitude_columns
        self.tier_1_groups = tier_1_groups
        self.tier_2_groups = tier_2_groups
        self.logger = logging.getLogger("AttitudeLogger")
        self.logger.info(f"AttitudeLogger 初始化, 将写入 {self.db_path}")

    async def log_step_attitudes(
        self, 
        env: OasisEnv, 
        current_step: int
    ):
        """
        在每个时间步结束时，记录 *每个* agent 的态度。
        - ABM Agent: 记录其 'internal_state' (来自 .attitude_scores)
        - LLM Agent: 记录其 'external_expression' (来自其 *当前时间步* 帖子的平均分)
        
        这完全基于您在 test.py 中提供的原始 log_agent_attitudes 函数逻辑。
        """
        self.logger.info(f"[Step {current_step}] 正在记录 *每个* Agent 的态度...")
        
        all_agents = list(env.agent_graph.get_agents())
        
        # 准备一个列表来批量插入
        # 格式: (table_name, time_step, user_id, agent_id, agent_type, metric_type, score)
        batch_insert_data = []

        # --- 1. 处理 ABM (Tier 2) - 内部状态 ---
        abm_agent_count = 0
        for agent_id, agent in all_agents:
            if agent.group in self.tier_2_groups:
                if hasattr(agent, 'attitude_scores') and isinstance(agent.attitude_scores, dict):
                    abm_agent_count += 1
                    
                    scores_dict = agent.attitude_scores.copy()
                    valid_scores = [scores_dict.get(col, 0.0) for col in self.attitude_columns if scores_dict.get(col) is not None]
                    scores_dict['attitude_average'] = np.mean(valid_scores) if valid_scores else 0.0
                    
                    agent_sim_id = agent.agent_id
                    user_id_str = agent.user_info.profile["other_info"].get("original_user_id")
                    
                    for dim_name, score_value in scores_dict.items():
                        table_name = f"log_{dim_name}"
                        batch_insert_data.append((
                            table_name,
                            current_step,
                            user_id_str,
                            agent_sim_id,
                            'ABM',
                            'internal_state',
                            score_value
                        ))

        # --- 2. 处理 LLM (Tier 1) - 外部表现 ---
        llm_agent_ids = {agent.agent_id for agent_id, agent in all_agents if agent.group in self.tier_1_groups}
        llm_agent_count = 0
        
        if llm_agent_ids:
            try:
                # (使用只读模式查询)
                with sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True) as conn:
                    id_placeholders = ", ".join(["?"] * len(llm_agent_ids))
                    avg_cols_sql = ", ".join([f"AVG({col})" for col in self.attitude_columns])
                    
                    query = f"""
                    SELECT 
                        user_id, 
                        agent_id,
                        {avg_cols_sql}
                    FROM post
                    WHERE created_at = ?
                      AND agent_id IN ({id_placeholders})
                      AND attitude_annotated = 1
                    GROUP BY user_id, agent_id
                    """
                    params = (current_step, *list(llm_agent_ids))
                    
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    llm_agent_count = len(rows)
                    
                    for row in rows:
                        user_id_str = str(row[0])
                        agent_sim_id = int(row[1])
                        llm_avgs_list = list(row[2:])
                        
                        scores_dict = {col: llm_avgs_list[i] for i, col in enumerate(self.attitude_columns)}
                        valid_avgs = [x for x in llm_avgs_list if x is not None]
                        scores_dict['attitude_average'] = np.mean(valid_avgs) if valid_avgs else 0.0
                        
                        for dim_name, score_value in scores_dict.items():
                            table_name = f"log_{dim_name}"
                            batch_insert_data.append((
                                table_name,
                                current_step,
                                user_id_str,
                                agent_sim_id,
                                'LLM',
                                'external_expression',
                                score_value
                            ))
            except sqlite3.Error as e:
                self.logger.error(f"[Step {current_step}] 查询 LLM 帖子分数时出错: {e}")
            except Exception as e:
                self.logger.error(f"[Step {current_step}] 处理 LLM 分数时意外出错: {e}", exc_info=True)

        # --- 3. 批量写入数据库 ---
        if not batch_insert_data:
            self.logger.info(f"[Step {current_step}] 没有新的态度分数需要记录。")
            return

        inserted_count = 0
        try:
            # (使用写模式连接)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for (table_name, ts, user_id, agent_id, agent_type, metric_type, score) in batch_insert_data:
                    try:
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
                        self.logger.error(f"[Step {current_step}] 写入日志表 '{table_name}' 失败 (Agent {agent_id}): {e}. (请确保该表已创建)")
                
                conn.commit()
                self.logger.info(f"[Step {current_step}] 成功记录 {abm_agent_count} 个 ABM agents 和 {llm_agent_count} 个 LLM agents (共 {inserted_count} 条分数)。")
                
        except sqlite3.Error as e:
            self.logger.error(f"[Step {current_step}] 批量写入态度日志时数据库出错: {e}")
        except Exception as e:
            self.logger.error(f"[Step {current_step}] 批量写入态度日志时意外出错: {e}", exc_info=True)
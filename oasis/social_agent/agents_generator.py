# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
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

import ast
import asyncio
import json
import time 
import logging
from typing import List, Optional, Union
from collections import defaultdict 
from datetime import datetime 

import pandas as pd
import numpy as np
import tqdm
import sqlite3
from camel.memories import MemoryRecord
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelManager, ModelFactory
from camel.types import OpenAIBackendRole


# 【!! 关键 !!】 导入你的 4+1 Agent
from oasis.social_agent.agent_custom import (
    BaseAgent, SocialAgent, AuthorityAgent, KOLAgent, 
    ActiveCreatorAgent, NormalUserAgent, # (Tier 1 - LLM)
    HeuristicAgent, LurkerAgent # (Tier 2 - ABM)
)
from oasis.social_platform import Channel, Platform
from oasis.social_platform.config import Neo4jConfig, UserInfo
from oasis.social_platform.typing import ActionType, RecsysType
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_agent.agent_action import SocialAction

# Tier 1: "重" LLM Agents (初始化慢, 运行慢)
TIER_1_LLM_GROUPS = {
    "权威媒体/大V",
    "活跃KOL",
    "活跃创作者",
    "普通用户" 
}
TIER_1_CLASS_MAP = {
    "权威媒体/大V": AuthorityAgent,
    "活跃KOL": KOLAgent,
    "活跃创作者": ActiveCreatorAgent,
    "普通用户": NormalUserAgent, 
    "default": SocialAgent
}

# Tier 2: "轻" ABM Agents (初始化快, 运行快)
TIER_2_HEURISTIC_GROUPS = {
    "潜水用户"
}
TIER_2_CLASS_MAP = {
    "潜水用户": LurkerAgent,
    "default": HeuristicAgent
}

def _parse_follow_list(follow_str: str) -> List[int]:
    """
    (【!! 修正: 按逗号解析 !!】)
    解析格式为 "[id1, id2, ...]" 或 "[]" 的字符串.
    """
    if not follow_str or follow_str == "[]" or pd.isna(follow_str):
        return []
    try:
        stripped_str = follow_str.strip("[]")
        if not stripped_str:
            return []
        ids_str_list = stripped_str.split(',')
        return [
            int(id_str.strip()) for id_str in ids_str_list if id_str.strip()
        ]
    except Exception as e:
        # 使用 logging (如果 agent_log 尚未定义)
        logging.warning(f"⚠️ 警告: _parse_follow_list 失败，输入: '{follow_str}', 错误: {e}")
        return []


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """(辅助函数) 清洗从 CSV 加载的数据"""
    logger = logging.getLogger("agents_generator")
    
    # --- 1. 【!! 修正 (User 64): user_id 是 TEXT !!】 ---
    df['user_id'] = df['user_id'].astype(str)
    
    # --- 2. 清洗 TEXT 列 (float nan -> str "") ---
    df['user_char'] = df['user_char'].fillna('')
    df['description'] = df['description'].fillna('')
    df['following_agentid_list'] = df['following_agentid_list'].fillna('[]')
        
    return df
def _load_initial_posts_from_db(db_path: str) -> dict[str, List[tuple[Optional[str], Optional[str]]]]:
    logger = logging.getLogger("agents_generator")
    logger.info(f"(Graph Build) 正在从 {db_path} 的 'post' 表预加载所有初始帖子...")
    
    # { "user_id_str": [ (content1, quote1), (content2, quote2) ], ... }
    initial_posts_map = defaultdict(list)
    
    try:
        # 使用只读模式 (mode=ro) 连接, 更安全
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # 【!! 关键修改: SELECT content 和 quote_content !!】
        cur.execute(
            "SELECT user_id, content, quote_content FROM post ORDER BY created_at"
        )
        
        count = 0
        for row in cur:
            # 确保 user_id 是 str, 以匹配 agent_info 中的 str
            user_id_str = str(row['user_id']) 
            
            # 【!! 关键修改: 存储元组 (content, quote_content) !!】
            initial_posts_map[user_id_str].append(
                (row['content'], row['quote_content'])
            )
            count += 1
        
        cur.close()
        conn.close()
        
        logger.info(f"(Graph Build) 成功从数据库加载 {count} 条初始帖子, "
                    f"分布在 {len(initial_posts_map)} 个用户中。")
        return initial_posts_map
        
    except sqlite3.Error as e:
        logger.error(f"❌ (Graph Build) 无法从 {db_path} 读取 'post' 表: {e}")
        logger.error("   将继续执行, 但所有 agent 的 memory 都会是空的。")
        return initial_posts_map # 返回空字典
    except Exception as e:
         logger.error(f"❌ (Graph Build) _load_initial_posts_from_db 发生意外错误: {e}")
         return initial_posts_map
    
def _preload_agent_memory(
    agent: BaseAgent, 
    initial_posts: List[tuple[Optional[str], Optional[str]]] 
):
    """
    (【!! 已修复 !!】)
    将初始帖子列表 (来自数据库) 作为 "post" 动作写入 agent 的 memory。
    现在会正确处理 'content' 和 'quote_content'。
    """
    logger = logging.getLogger("agents_generator")
    
    if not initial_posts: # 检查列表是否为空
        return

    try:
        post_count = 0
        
        # 【!! 关键修改: 迭代元组 !!】
        for post_tuple in initial_posts:
            user_comment_raw, original_post_raw = post_tuple
            
            # (进行基本清理, 处理 None 和 bytes)
            user_comment = ""
            if isinstance(user_comment_raw, bytes):
                user_comment = user_comment_raw.decode('utf-8', 'replace').strip()
            elif isinstance(user_comment_raw, str):
                user_comment = user_comment_raw.strip()
                
            original_post = ""
            if isinstance(original_post_raw, bytes):
                original_post = original_post_raw.decode('utf-8', 'replace').strip()
            elif isinstance(original_post_raw, str):
                original_post = original_post_raw.strip()

        
            text_to_load_in_memory = ""
            if user_comment:
                text_to_load_in_memory = f"[用户评论]\n{user_comment}"
                if original_post:
                    text_to_load_in_memory += f"\n\n[转发的原帖]\n{original_post}"
            elif original_post:
                text_to_load_in_memory = f"[转发的原帖]\n{original_post}"
            else:
                continue # 如果 content 和 quote_content 都为空, 则跳过

            # 格式化为 LLM 动作输出 (模仿 "post" 动作)
            action_content = json.dumps({
                "reason": "This is an initial post from my history.",
                "functions": [
                    {
                        "name": "post",
                        "arguments": {
                            "content": text_to_load_in_memory
                        }
                    }
                ]
            })
            
            
            # 2. 创建一个 'user' 消息, 但 role_name 是 'system'
            #    (这模仿了 OASIS 记录动作的方式)
            agent_msg = BaseMessage.make_user_message(
                role_name=OpenAIBackendRole.ASSISTANT.value, # "system"
                content=action_content
            )
            
            # 3. 将其作为 SYSTEM 角色写入 Memory
            #    (这对应于 to_openai_system_message())
            agent.memory.write_record(
                MemoryRecord(message=agent_msg, 
                             role_at_backend=OpenAIBackendRole.ASSISTANT)
            )
            post_count += 1
        
        if post_count > 0:
            logger.debug(f"(Graph Build) 成功为 Agent {agent.agent_id} "
                         f"预加载了 {post_count} 条帖子到 Memory。")
    
    except Exception as e:
        logger.error(f"❌ (Graph Build) 预加载 Memory 失败 for agent "
                     f"{agent.agent_id}: {e}")

async def generate_agents(
    agent_info_path: str,
    channel: Channel,
    model: Union[BaseModelBackend, List[BaseModelBackend]],
    start_time,
    recsys_type: str = "twitter",
    twitter: Platform = None,
    available_actions: list[ActionType] = None,
    neo4j_config: Neo4jConfig | None = None,
) -> AgentGraph:
    
    
    agent_info = pd.read_csv(agent_info_path, index_col=0)
    agent_info['user_id'] = agent_info['user_id'].astype(str)

    # --- 【!! 预计算 !!】 ---
    print("... (Pre-processing) 正在构建关注图以计算关注数/粉丝数 ...")
    all_agent_ids = set(agent_info.index)
    followings_map = defaultdict(set) 
    followers_map = defaultdict(set)  

    for agent_id, row in agent_info.iterrows():
        followee_list = _parse_follow_list(row["following_agentid_list"])
        for followee_id in followee_list:
            if followee_id in all_agent_ids:
                followings_map[agent_id].add(followee_id)
                followers_map[followee_id].add(agent_id)
    print("... (Pre-processing) 关注图构建完成。")
    # --- 【!! 修正结束 !!】 ---

    agent_graph = (AgentGraph() if neo4j_config is None else AgentGraph(
        backend="neo4j",
        neo4j_config=neo4j_config,
    ))

    sign_up_list = []
    follow_list = []
    post_list = []

    for agent_id, row in tqdm.tqdm(agent_info.iterrows(), 
                                  total=len(agent_info), 
                                  desc="Generating agents from CSV"):
        profile = {
            "nodes": [],
            "edges": [],
            "other_info": {},
        }
        profile["other_info"]["user_profile"] = row["user_char"]
        profile["other_info"]["original_user_id"] = row["user_id"] 

        user_info = UserInfo(
            name=row["username"], 
            user_name=row["name"], 
            description=row["description"],
            profile=profile,
            recsys_type=recsys_type,
        )

        agent = SocialAgent(
            agent_id=agent_id,
            user_info=user_info,
            channel=channel,
            model=model,
            agent_graph=agent_graph,
            available_actions=available_actions,
        )

        agent_graph.add_agent(agent)
        
        num_followings = len(followings_map.get(agent_id, set()))
        num_followers = len(followers_map.get(agent_id, set()))

        sign_up_list.append((
            row["user_id"],
            agent_id,
            row["name"],
            row["username"],
            row["description"],
            start_time, 
            num_followings,
            num_followers,
        ))

        following_id_list = followings_map.get(agent_id, set())
        
        if len(following_id_list) != 0:
            for follow_id in following_id_list:
                follow_list.append((agent_id, follow_id, start_time)) 
                agent_graph.add_edge(agent_id, follow_id)
        
        if 'previous_tweets' in row and pd.notna(row['previous_tweets']):
            try:
                previous_posts = ast.literal_eval(row["previous_tweets"])
                if len(previous_posts) != 0:
                    for post in previous_posts:
                        post_list.append((row["user_id"], post, start_time, 0, 0))
            except (ValueError, SyntaxError):
                pass 


    user_insert_query = (
        "INSERT INTO user (user_id, agent_id, user_name, name, bio, "
        "created_at, num_followings, num_followers) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?)")
    twitter.pl_utils._execute_many_db_command(user_insert_query,
                                              sign_up_list,
                                              commit=True)

    follow_insert_query = (
        "INSERT INTO follow (follower_id, followee_id, created_at) "
        "VALUES (?, ?, ?)")
    twitter.pl_utils._execute_many_db_command(follow_insert_query,
                                              follow_list,
                                              commit=True)
    
    post_insert_query = (
        "INSERT INTO post (user_id, content, created_at, num_likes, "
        "num_dislikes) VALUES (?, ?, ?, ?, ?)")
    twitter.pl_utils._execute_many_db_command(post_insert_query,
                                              post_list,
                                              commit=True)

    return agent_graph


async def generate_agents_100w(
    agent_info_path: str,
    channel: Channel,
    start_time,
    model: Union[BaseModelBackend, List[BaseModelBackend]],
    recsys_type: str = "twitter",
    twitter: Platform = None,
    available_actions: list[ActionType] = None,
) -> List:
    """... (此函数已包含 TODO 修复, 保持不变) ..."""
    
    agent_info = pd.read_csv(agent_info_path, index_col=0)
    agent_info['user_id'] = agent_info['user_id'].astype(str)

    # --- 【!! 预计算 !!】 ---
    print("... (Pre-processing 100w) 正在构建关注图以计算关注数/粉丝数 ...")
    all_agent_ids = set(agent_info.index)
    followings_map = defaultdict(set) 
    followers_map = defaultdict(set)  

    for agent_id, row in agent_info.iterrows():
        followee_list = _parse_follow_list(row["following_agentid_list"])
        for followee_id in followee_list:
            if followee_id in all_agent_ids:
                followings_map[agent_id].add(followee_id)
                followers_map[followee_id].add(agent_id)
    print("... (Pre-processing 100w) 关注图构建完成。")
    # --- 【!! 修正结束 !!】 ---

    agent_graph = []
    sign_up_list = []
    follow_list = []
    post_list = []

    previous_tweets_lists = agent_info["previous_tweets"].apply(
        ast.literal_eval)

    for agent_id, row in tqdm.tqdm(agent_info.iterrows(), 
                                  total=len(agent_info), 
                                  desc="Generating 1M agents"):
        profile = {
            "nodes": [],
            "edges": [],
            "other_info": {},
        }
        profile["other_info"]["user_profile"] = row["user_char"]
        profile["other_info"]["original_user_id"] = row["user_id"]

        user_info = UserInfo(
            name=row["username"], 
            user_name=row["name"], 
            description=row["description"],
            profile=profile,
            recsys_type=recsys_type,
        )

        agent = SocialAgent(
            agent_id=agent_id,
            user_info=user_info,
            channel=channel,
            model=model,
            agent_graph=agent_graph,
            available_actions=available_actions,
        )

        agent_graph.append(agent)
        
        num_followings = len(followings_map.get(agent_id, set()))
        num_followers = len(followers_map.get(agent_id, set()))

        sign_up_list.append((
            row["user_id"],
            agent_id,
            row["name"],
            row["username"],
            row["description"],
            start_time, # <-- 此函数依赖传入的 start_time (应为 datetime)
            num_followings,
            num_followers,
        ))

        following_id_list = followings_map.get(agent_id, set())

        if len(following_id_list) != 0:
            for follow_id in following_id_list:
                follow_list.append((agent_id, follow_id, start_time))
        
        previous_posts = previous_tweets_lists[agent_id]
        if len(previous_posts) != 0:
            for post in previous_posts:
                post_list.append((row["user_id"], post, start_time, 0, 0))

    user_insert_query = (
        "INSERT INTO user (user_id, agent_id, user_name, name, bio, "
        "created_at, num_followings, num_followers) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?)")
    twitter.pl_utils._execute_many_db_command(user_insert_query,
                                              sign_up_list,
                                              commit=True)

    follow_insert_query = (
        "INSERT INTO follow (follower_id, followee_id, created_at) "
        "VALUES (?, ?, ?)")
    twitter.pl_utils._execute_many_db_command(follow_insert_query,
                                              follow_list,
                                              commit=True)

    post_insert_query = (
        "INSERT INTO post (user_id, content, created_at, num_likes, "
        "num_dislikes) VALUES (?, ?, ?, ?, ?)")
    twitter.pl_utils._execute_many_db_command(post_insert_query,
                                              post_list,
                                              commit=True)

    return agent_graph


async def generate_controllable_agents(
    channel: Channel,
    control_user_num: int,
) -> tuple[AgentGraph, dict]:
    agent_graph = AgentGraph()
    agent_user_id_mapping = {}
    for i in range(control_user_num):
        user_info = UserInfo(
            is_controllable=True,
            profile={"other_info": {
                "user_profile": "None"
            }},
            recsys_type="reddit",
        )
        agent = SocialAgent(agent_id=i,
                            user_info=user_info,
                            channel=channel,
                            agent_graph=agent_graph)
        agent_graph.add_agent(agent)
        username = input(f"Please input username for agent {i}: ")
        name = input(f"Please input name for agent {i}: ")
        bio = input(f"Please input bio for agent {i}: ")
        response = await agent.env.action.sign_up(username, name, bio)
        user_id = response["user_id"]
        agent_user_id_mapping[i] = user_id
    for i in range(control_user_num):
        for j in range(control_user_num):
            agent = agent_graph.get_agent(i)
            if i != j:
                user_id = agent_user_id_mapping[j]
                await agent.env.action.follow(user_id)
                agent_graph.add_edge(i, j)
    return agent_graph, agent_user_id_mapping


async def gen_control_agents_with_data(
    channel: Channel,
    control_user_num: int,
    models: list[BaseModelBackend] | None = None,
    available_actions: list[ActionType] | None = None
) -> tuple[AgentGraph, dict]:
    
    agent_graph = AgentGraph()
    agent_user_id_mapping = {}
    for i in range(control_user_num):
        user_info = UserInfo(
            is_controllable=True,
            profile={
                "other_info": {
                    "user_profile": "None",
                    "gender": "None",
                    "mbti": "None",
                    "country": "None",
                    "age": "None",
                }
            },
            recsys_type="reddit",
        )
        agent = SocialAgent(
            agent_id=i,
            user_info=user_info,
            channel=channel,
            agent_graph=agent_graph,
            model=models,
            available_actions=available_actions,
        )
        agent_graph.add_agent(agent)
        user_name = "momo"
        name = "momo"
        bio = "None."
        response = await agent.env.action.sign_up(user_name, name, bio)
        user_id = response["user_id"]
        agent_user_id_mapping[i] = user_id
    return agent_graph, agent_user_id_mapping


async def generate_reddit_agents(
    agent_info_path: str,
    channel: Channel,
    agent_graph: AgentGraph | None = None,
    agent_user_id_mapping: dict[int, int] | None = None,
    follow_post_agent: bool = False,
    mute_post_agent: bool = False,
    model: Optional[Union[BaseModelBackend, List[BaseModelBackend],
                          ModelManager]] = None,
    available_actions: list[ActionType] = None
) -> AgentGraph:
    if agent_user_id_mapping is None:
        agent_user_id_mapping = {}
    if agent_graph is None:
        agent_graph = AgentGraph()
    control_user_num = agent_graph.get_num_nodes()
    with open(agent_info_path, "r") as file:
        agent_info = json.load(file)
    async def process_agent(i):
        profile = {
            "nodes": [],
            "edges": [],
            "other_info": {},
        }
        profile["other_info"]["user_profile"] = agent_info[i]["persona"]
        profile["other_info"]["mbti"] = agent_info[i]["mbti"]
        profile["other_info"]["gender"] = agent_info[i]["gender"]
        profile["other_info"]["age"] = agent_info[i]["age"]
        profile["other_info"]["country"] = agent_info[i]["country"]
        user_info = UserInfo(
            name=agent_info[i]["username"],
            description=agent_info[i]["bio"],
            profile=profile,
            recsys_type="reddit",
        )
        agent = SocialAgent(
            agent_id=i + control_user_num,
            user_info=user_info,
            channel=channel,
            agent_graph=agent_graph,
            model=model,
            available_actions=available_actions,
        )
        agent_graph.add_agent(agent)
        response = await agent.env.action.sign_up(agent_info[i]["username"],
                                                  agent_info[i]["realname"],
                                                  agent_info[i]["bio"])
        user_id = response["user_id"]
        agent_user_id_mapping[i + control_user_num] = user_id
        if follow_post_agent:
            await agent.env.action.follow(1)
            content = """
{
    "reason": "He is my friend, and I would like to follow him "
              "on social media.",
    "functions": [
        {
            "name": "follow",
            "arguments": {
                "user_id": 1
            }
        }
    ]
}
"""
            agent_msg = BaseMessage.make_assistant_message(
                role_name="Assistant", content=content)
            agent.memory.write_record(
                MemoryRecord(agent_msg, OpenAIBackendRole.ASSISTANT))
        elif mute_post_agent:
            await agent.env.action.mute(1)
            content = """
{
    "reason": "He is my enemy, and I would like to mute him on social media.",
    "functions": [{
        "name": "mute",
        "arguments": {
            "user_id": 1
        }
}
"""
            agent_msg = BaseMessage.make_assistant_message(
                role_name="Assistant", content=content)
            agent.memory.write_record(
                MemoryRecord(agent_msg, OpenAIBackendRole.ASSISTANT))
    tasks = [process_agent(i) for i in range(len(agent_info))]
    await asyncio.gather(*tasks)
    return agent_graph


def connect_platform_channel(
    channel: Channel,
    agent_graph: AgentGraph | None = None,
) -> AgentGraph:
    # ... (此函数保持不变) ...
    for _, agent in agent_graph.get_agents():
        agent.channel = channel
        agent.env.action.channel = channel
    return agent_graph


async def generate_custom_agents(
    platform: Platform, 
    agent_graph: AgentGraph | None = None,
    CALIBRATION_END: datetime = None,
    TIME_STEP_MINUTES: int = 3
) -> AgentGraph:
    """
    这个函数现在是 env.reset() 的一部分。
    它 *假定* agent_graph 已经由 generate_twitter_agent_graph() 填充了。
    它 *只* 负责将这个图注册到数据库中。
    
    - 不再将 attitude 写入 'user' 表。
    - 在注册后, 它将 *查询* `ground_truth_post` 表。
    - 它会计算 T<0 (e.g., -1, -2, ...) 的 *历史* 态度分数 (基于 CALIBRATION_END)
    - 并将这些历史分数 (逐 agent, 逐 time_step) 写入日志表。
    """
    logger = logging.getLogger("agents_generator")
    
    # --- ( ATTITUDE_COLUMNS 和 Agent 分组定义保持不变 ) ---
    ATTITUDE_COLUMNS = [
        'attitude_lifestyle_culture',
        'attitude_sport_ent',
        'attitude_sci_health',
        'attitude_politics_econ'
    ]
    TIER_1_LLM_GROUPS = {
        "权威媒体/大V", "活跃KOL", "活跃创作者", "普通用户" 
    }
    TIER_2_HEURISTIC_GROUPS = {
        "潜水用户"
    }
    
    
    if agent_graph is None:
        agent_graph = AgentGraph()
    
    channel = platform.channel
    agent_graph = connect_platform_channel(channel=channel,
                                           agent_graph=agent_graph)
    
    logger.info("... (generate_custom_agents) 正在准备批量注册用户到数据库 ...")
    
    # --- ( 预计算关注/粉丝 保持不变 ) ---
    logger.info("... (generate_custom_agents) 正在预计算关注数/粉丝数 ...")
    followings_map = defaultdict(set) 
    followers_map = defaultdict(set)
    all_agent_ids = set(str(aid) for aid, _ in agent_graph.get_agents())
    
    for agent_id, agent in agent_graph.get_agents():
        agent_id_str = str(agent_id)
        follow_str = agent.user_info.profile["other_info"].get(
            "following_agentid_list_str", "[]"
        )
        followee_list = _parse_follow_list(follow_str)
        for followee_id in followee_list:
            if str(followee_id) in all_agent_ids:
                followings_map[agent_id_str].add(str(followee_id))
                followers_map[str(followee_id)].add(agent_id_str)
    logger.info(f"... (generate_custom_agents) 关注图构建完成。{len(followings_map)} 个用户有关注列表。")

    
    sign_up_list = []
    follow_list = []
    
    # --- [!! 新增: T<0 日志所需 Agent 映射 !!] ---
    agent_id_to_type_map = {} # ( e.g., 1001 -> ('ABM', 'internal_state') )
    
    for agent_id_int, agent in agent_graph.get_agents():
        
        user_name = agent.user_info.user_name
        name = agent.user_info.name
        bio = agent.user_info.description
        
        profile_info = agent.user_info.profile["other_info"]
        user_id_str = profile_info.get("original_user_id") # (e.g., '1618051664')
            
        current_time = datetime.now()
        
        agent_sim_id_str = str(agent_id_int) # (e.g., '1001')
        num_followings = len(followings_map.get(agent_sim_id_str, set()))
        num_followers = len(followers_map.get(agent_sim_id_str, set()))
        
        # --- [!! 修改: 不再收集 T-1 日志, 仅收集 Agent 类型 !!] ---
        agent_sim_id = agent.agent_id # (e.g., 1001)
        
        if agent.group in TIER_1_LLM_GROUPS:
            agent_id_to_type_map[agent_sim_id] = ('LLM', 'external_expression')
        elif agent.group in TIER_2_HEURISTIC_GROUPS:
            agent_id_to_type_map[agent_sim_id] = ('ABM', 'internal_state')
        
        sign_up_list.append((
            user_id_str,    # (str)
            agent_sim_id,   # (int)
            user_name,      # (str)
            name,           # (str)
            bio,            # (str)
            current_time,   # (datetime)
            num_followings, # (int)
            num_followers,  # (int)
        ))
        
        # ( follow_list.append 保持不变 )
        following_id_list = followings_map.get(agent_sim_id_str, set())
        for follow_id in following_id_list:
            follow_list.append((agent_sim_id, int(follow_id), current_time))
    
    # --- ( user_insert_query 保持不变 ) ---
    user_insert_query = (
        f"INSERT OR IGNORE INTO user (user_id, agent_id, user_name, name, bio, "
        f"created_at, num_followings, num_followers) VALUES "
        f"(?, ?, ?, ?, ?, ?, ?, ?)"
    )
    
    platform.pl_utils._execute_many_db_command(user_insert_query,
                                               sign_up_list,
                                               commit=True)
    
    logger.info(f"... (generate_custom_agents) 成功注册 {len(sign_up_list)} 个用户 (无 Attitude)。")
    
    # --- ( follow_insert_query 保持不变 ) ---
    follow_insert_query = (
        "INSERT OR IGNORE INTO follow (follower_id, followee_id, created_at) "
        "VALUES (?, ?, ?)")
    platform.pl_utils._execute_many_db_command(follow_insert_query,
                                              follow_list,
                                              commit=True)
    logger.info(f"... (generate_custom_agents) 成功插入 {len(follow_list)} 条关注关系。")
    
    
    # --- [!! 重写: 记录 T<0 历史态度日志 (基于post) !!] ---
    if CALIBRATION_END is None:
        logger.warning("... (generate_custom_agents) 未提供 CALIBRATION_END, 跳过 T<0 历史态度日志记录。")
    else:
        logger.info(f"... (generate_custom_agents) 正在计算 T<0 历史态度日志 (基于 {CALIBRATION_END})...")
        
        try:
            conn = platform.pl_utils.db_conn
            
            # 1. 查询: `JOIN` `user` (我们刚创建的) 和 `post`
            att_cols_sql = ", ".join([f"T1.{col}" for col in ATTITUDE_COLUMNS])
            query = f"""
            SELECT 
                T1.created_at, 
                T1.user_id, 
                T2.agent_id,
                {att_cols_sql}
            FROM post AS T1
            INNER JOIN user AS T2 ON T1.user_id = T2.user_id
            WHERE T1.created_at < ? AND T1.attitude_annotated = 1
            """
            
            df_history = pd.read_sql_query(
                query, 
                conn, 
                params=(CALIBRATION_END.strftime("%Y-%m-%d %H:%M:%S"),)
            )
            
            if df_history.empty:
                logger.info("... (generate_custom_agents) 未在 T<0 范围内找到已标注的历史帖子。")
                return agent_graph # (仍然返回 graph)

            logger.info(f"... (generate_custom_agents) 找到 {len(df_history)} 条 T<0 历史帖子。")

            # 2. 计算历史 time_step
            df_history['created_at_dt'] = pd.to_datetime(df_history['created_at'])
            
            # (计算距离 CALIBRATION_END 的秒数)
            delta_seconds = (CALIBRATION_END - df_history['created_at_dt']).dt.total_seconds()
            
            # (计算 time_step: 1-180s -> -1; 181-360s -> -2)
            time_step_col = -((delta_seconds // (TIME_STEP_MINUTES * 60)) + 1).astype(int)
            df_history['time_step'] = time_step_col

            # 3. 按 (agent, time_step) 聚合, 计算平均态度
            df_grouped = df_history.groupby(
                ['time_step', 'user_id', 'agent_id']
            )[ATTITUDE_COLUMNS].mean().reset_index()

            # 4. 映射 Agent 类型
            df_grouped['agent_type_metric'] = df_grouped['agent_id'].map(agent_id_to_type_map)
            df_grouped = df_grouped.dropna(subset=['agent_type_metric']) # 移除不在模拟中的 agents
            df_grouped['agent_type'] = df_grouped['agent_type_metric'].apply(lambda x: x[0])
            df_grouped['metric_type'] = df_grouped['agent_type_metric'].apply(lambda x: x[1])

            # 5. 准备批量插入
            all_dims_to_log = ATTITUDE_COLUMNS + ['attitude_average']
            batch_insert_data = []
            
            for row in df_grouped.itertuples(index=False):
                scores_dict = {col: getattr(row, col) for col in ATTITUDE_COLUMNS}
                
                # 计算总平均分
                valid_scores = [s for s in scores_dict.values() if s is not None]
                overall_avg = np.mean(valid_scores) if valid_scores else 0.0
                scores_dict['attitude_average'] = overall_avg
                
                # 为 5 个维度准备插入
                for dim_name in all_dims_to_log:
                    table_name = f"log_{dim_name}"
                    score_to_log = scores_dict.get(dim_name)
                    if score_to_log is not None:
                        batch_insert_data.append((
                            table_name,
                            row.time_step,
                            row.user_id,
                            row.agent_id,
                            row.agent_type,
                            row.metric_type,
                            score_to_log
                        ))

            # 6. 执行批量插入
            if not batch_insert_data:
                logger.info("... (generate_custom_agents) 聚合后没有 T<0 日志需要插入。")
            else:
                cursor = conn.cursor()
                inserted_count = 0
                for (tbl, ts, uid, aid, atype, mtype, score) in batch_insert_data:
                    try:
                        cursor.execute(
                            f"""
                            INSERT INTO {tbl} (
                                time_step, user_id, agent_id, agent_type, metric_type, attitude_score
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (ts, uid, aid, atype, mtype, score)
                        )
                        inserted_count += 1
                    except sqlite3.Error as e:
                        logger.error(f"... (generate_custom_agents) 写入 T<0 日志表 '{tbl}' 失败 (Agent {aid}, Step {ts}): {e}")
                
                conn.commit()
                logger.info(f"... (generate_custom_agents) 成功将 {inserted_count} 条 T<0 历史分数 (5 维 * N agents * M steps) 写入日志表。")

        except Exception as e:
            logger.error(f"... (generate_custom_agents) 记录 T<0 历史态度日志时意外出错: {e}", exc_info=True)
    # --- [!! 重写结束 !!] ---
    
    
    logger.info("... (generate_custom_agents) 已跳过 post 插入 (假设数据库已由 init_post_tables.py 填充)。")
    
    return agent_graph

async def generate_reddit_agent_graph(
    profile_path: str,
    model: Optional[Union[BaseModelBackend, List[BaseModelBackend],
                          ModelManager]] = None,
    available_actions: list[ActionType] = None,
) -> AgentGraph:
    # ... (此函数保持不变) ...
    agent_graph = AgentGraph()
    with open(profile_path, "r") as file:
        agent_info = json.load(file)
    async def process_agent(i):
        profile = {
            "nodes": [],
            "edges": [],
            "other_info": {},
        }
        profile["other_info"]["user_profile"] = agent_info[i]["persona"]
        profile["other_info"]["mbti"] = agent_info[i]["mbti"]
        profile["other_info"]["gender"] = agent_info[i]["gender"]
        profile["other_info"]["age"] = agent_info[i]["age"]
        profile["other_info"]["country"] = agent_info[i]["country"]
        user_info = UserInfo(
            name=agent_info[i]["username"],
            description=agent_info[i]["bio"],
            profile=profile,
            recsys_type="reddit",
        )
        agent = SocialAgent(
            agent_id=i,
            user_info=user_info,
            agent_graph=agent_graph,
            model=model,
            available_actions=available_actions,
        )
        agent_graph.add_agent(agent)
    tasks = [process_agent(i) for i in range(len(agent_info))]
    await asyncio.gather(*tasks)
    return agent_graph


async def generate_twitter_agent_graph(
    profile_path: str,
    db_path: str,  
    model: Optional[Union[BaseModelBackend, List[BaseModelBackend],
                          ModelManager]] = None,
    available_actions: list[ActionType] = None,
) -> AgentGraph:
    
    logger = logging.getLogger("agents_generator")
    
    
    initial_posts_map = _load_initial_posts_from_db(db_path)
    
    agent_graph = AgentGraph()
    
    # --- 1. 加载并拆分所有用户 (逻辑不变) ---
    logger.info(f"(Graph Build) 正在从 {profile_path} 加载并清洗所有用户数据...")
    try:
        all_user_info = pd.read_csv(profile_path, index_col=0, dtype={'user_id': str})
        all_user_info = _clean_dataframe(all_user_info)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        logger.error(f"❌ (Graph Build) 找不到或用户文件 {profile_path} 为空。模拟无法启动。")
        return agent_graph
    except KeyError as e:
        logger.error(f"❌ (Graph Build) CSV 文件缺少必需的列: {e}")
        logger.error("   请确保 'create_agent_init.py' 脚本已正确运行并生成了 'Unnamed: 0' (索引)。")
        return agent_graph

    tier1_info = all_user_info[all_user_info['group'].isin(TIER_1_LLM_GROUPS)]
    tier2_info = all_user_info[all_user_info['group'].isin(TIER_2_HEURISTIC_GROUPS)]
    
    logger.info(f"(Graph Build) 数据加载完毕: {len(tier1_info)} 个 [Tier 1 LLM Agents], {len(tier2_info)} 个 [Tier 2 ABM Agents]")
    
    # --- 2. 遍历并创建数据 (分两步) ---
    
    # 步骤 A: 遍历 Heuristic Agents (Tier 2)
    logger.info(f"(Graph Build) 正在为 {len(tier2_info)} 个 Heuristic Agents (Tier 2) 创建对象...")
    for agent_id, row in tqdm.tqdm(tier2_info.iterrows(), total=len(tier2_info), desc="Building Heuristic Agents"):
        
        
        user_name = row["username"]
        name = row["name"]
        bio = row["description"]
        user_id = row["user_id"] # str
        group_name = row["group"]

        AgentClass = TIER_2_CLASS_MAP.get(group_name, HeuristicAgent)

        profile = {
            "nodes": [], "edges": [], "other_info": {
                "user_profile": row["user_char"],
                "original_user_id": user_id,
                "following_agentid_list_str": row["following_agentid_list"],
                "group": group_name,
                "attitude_lifestyle_culture": row.get("initial_attitude_lifestyle_culture", 0.0),
                "attitude_sport_ent": row.get("initial_attitude_sport_ent", 0.0),
                "attitude_sci_health": row.get("initial_attitude_sci_health", 0.0),
                "attitude_politics_econ": row.get("initial_attitude_politics_econ", 0.0),
                "initial_attitude_avg": row.get("initial_attitude_avg", 0.0)
            }
        }
        user_info = UserInfo(
            name=user_name, user_name=name, description=bio,
            profile=profile, recsys_type='twitter',
        )
        
        agent_env = SocialAction(agent_id=agent_id, channel=None)

        
       
        agent = AgentClass(
                agent_id=agent_id,
                env=agent_env,
                user_info=user_info  # (LurkerAgent 会从这里解析 attitude_scores)
            )
        
        agent_graph.add_agent(agent)
     
            

            
    # 步骤 B: 遍历 LLM Agents (Tier 1) (逻辑不变)
    logger.info(f"(Graph Build) 正在为 {len(tier1_info)} 个 LLM Agents (Tier 1) 创建对象...")
    for agent_id, row in tqdm.tqdm(tier1_info.iterrows(), total=len(tier1_info), desc="Building LLM Agents (This will be slow)"):
        # --- (所有变量定义保持不变) ---
        user_name = row["username"]
        name = row["name"]
        bio = row["description"]
        user_id = row["user_id"] # str
        group_name = row["group"]

        AgentClass = TIER_1_CLASS_MAP.get(group_name, SocialAgent)

        profile = {
            "nodes": [], "edges": [], "other_info": {
                "user_profile": row["user_char"],
                "original_user_id": user_id,
                "following_agentid_list_str": row["following_agentid_list"],
                "group": group_name,
                "attitude_lifestyle_culture": row.get("initial_attitude_lifestyle_culture", 0.0),
                "attitude_sport_ent": row.get("initial_attitude_sport_ent", 0.0),
                "attitude_sci_health": row.get("initial_attitude_sci_health", 0.0),
                "attitude_politics_econ": row.get("initial_attitude_politics_econ", 0.0),
                "initial_attitude_avg": row.get("initial_attitude_avg", 0.0)
            }
        }
        user_info = UserInfo(
            name=user_name, user_name=name, description=bio,
            profile=profile, recsys_type='twitter',
        )
        
        agent = AgentClass(
            agent_id=agent_id, 
            user_info=user_info, 
            model=model,
            agent_graph=agent_graph, 
            available_actions=available_actions,
            channel=None 
        )
        agent_graph.add_agent(agent)

        posts_for_this_agent = initial_posts_map.get(user_id, [])
        _preload_agent_memory(agent, posts_for_this_agent)

    logger.info(f"(Graph Build) 成功创建 {agent_graph.get_num_nodes()} 个 agent (T1+T2)。")
    
    return agent_graph
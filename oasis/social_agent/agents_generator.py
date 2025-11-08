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
# --- 【!! 关键: 定义你的 4+1 架构 !!】 ---

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
) -> AgentGraph:
    """
    (【!! 关键重构 !!】)
    这个函数现在是 env.reset() 的一部分。
    它 *假定* agent_graph 已经由 generate_twitter_agent_graph() 填充了。
    它 *只* 负责将这个图注册到数据库中。
    """
    logger = logging.getLogger("agents_generator")
    
    if agent_graph is None:
        agent_graph = AgentGraph()
    

    channel = platform.channel

    agent_graph = connect_platform_channel(channel=channel,
                                           agent_graph=agent_graph)
    
    logger.info("... (generate_custom_agents) 正在准备批量注册用户到数据库 ...")
    
    logger.info("... (generate_custom_agents) 正在预计算关注数/粉丝数 ...")
    followings_map = defaultdict(set) 
    followers_map = defaultdict(set)
    
    all_agent_ids = set(aid for aid, _ in agent_graph.get_agents())

    for agent_id, agent in agent_graph.get_agents():
        follow_str = agent.user_info.profile["other_info"].get(
            "following_agentid_list_str", "[]"
        )
        followee_list = _parse_follow_list(follow_str)
        for followee_id in followee_list:
            if followee_id in all_agent_ids:
                followings_map[agent_id].add(followee_id)
                followers_map[followee_id].add(agent_id)
    
    logger.info(f"... (generate_custom_agents) 关注图构建完成。{len(followings_map)} 个用户有关注列表。")
    
    sign_up_list = []
    follow_list = []
    
    for agent_id, agent in agent_graph.get_agents():
        
        user_name = agent.user_info.user_name
        name = agent.user_info.name
        bio = agent.user_info.description # (已清洗)
        user_id = agent.user_info.profile["other_info"].get("original_user_id") # (已清洗)
            
        current_time = datetime.now()
            
        num_followings = len(followings_map.get(agent_id, set()))
        num_followers = len(followers_map.get(agent_id, set()))

        sign_up_list.append((
            user_id,        # (str)
            agent_id,       # (int)
            user_name,      # (str)
            name,           # (str)
            bio,            # (str)
            current_time,   # (datetime)
            num_followings, # (int)
            num_followers,  # (int)
        ))
        
        following_id_list = followings_map.get(agent_id, set())
        for follow_id in following_id_list:
            follow_list.append((agent_id, follow_id, current_time))
            # (在 graph_build 步骤中已添加)
            # agent_graph.add_edge(agent_id, follow_id) 
    
    user_insert_query = (
        "INSERT OR IGNORE INTO user (user_id, agent_id, user_name, name, bio, "
        "created_at, num_followings, num_followers) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?)")
    
    platform.pl_utils._execute_many_db_command(user_insert_query,
                                               sign_up_list,
                                               commit=True)
    
    logger.info(f"... (generate_custom_agents) 成功注册 {len(sign_up_list)} 个用户。")
    
    follow_insert_query = (
        "INSERT OR IGNORE INTO follow (follower_id, followee_id, created_at) "
        "VALUES (?, ?, ?)")
    platform.pl_utils._execute_many_db_command(follow_insert_query,
                                              follow_list,
                                              commit=True)
    logger.info(f"... (generate_custom_agents) 成功插入 {len(follow_list)} 条关注关系。")
    
    # 【!! 修正 (User 63): 移除 post 插入 !!】
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
    model: Optional[Union[BaseModelBackend, List[BaseModelBackend],
                          ModelManager]] = None,
    available_actions: list[ActionType] = None,
) -> AgentGraph:
    
    logger = logging.getLogger("agents_generator")
    
    # 【!! 关键修正: 这个函数现在是*主要*的 Agent 创建函数 !!】
    
    agent_graph = AgentGraph()
    
    # --- 1. 加载并拆分所有用户 ---
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
    # (Tier 3 '潜水用户' 将被忽略, 不会创建 agent 对象)
    
    logger.info(f"(Graph Build) 数据加载完毕: {len(tier1_info)} 个 [Tier 1 LLM Agents], {len(tier2_info)} 个 [Tier 2 ABM Agents]")
    
    # --- 2. 遍历并创建数据 (分两步) ---
    
    # 步骤 A: 遍历 Heuristic Agents (Tier 2) (轻量级 Init, 极快)
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
                "group": group_name 
            }
        }
        user_info = UserInfo(
            name=user_name, user_name=name, description=bio,
            profile=profile, recsys_type='twitter',
        )
        
        agent = AgentClass(
            agent_id=agent_id, 
            user_info=user_info, 
            agent_graph=agent_graph, 
            channel=None # Channel 将在 env.reset() 中被连接
        )
        agent_graph.add_agent(agent)
            
    # 步骤 B: 遍历 LLM Agents (Tier 1) (重量级 Init, 慢速)
    logger.info(f"(Graph Build) 正在为 {len(tier1_info)} 个 LLM Agents (Tier 1) 创建对象...")
    for agent_id, row in tqdm.tqdm(tier1_info.iterrows(), total=len(tier1_info), desc="Building LLM Agents (This will be slow)"):
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
                "group": group_name
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

    logger.info(f"(Graph Build) 成功创建 {agent_graph.get_num_nodes()} 个 agent (T1+T2)。")
    
    return agent_graph
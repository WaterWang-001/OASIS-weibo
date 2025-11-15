# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# ... (Apache License) ...
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========

import random
import asyncio
import numpy as np
from typing import TYPE_CHECKING, Optional, Union, List, Callable, Dict
# 核心 CAMEL 导入
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelManager
from camel.toolkits import FunctionTool
from camel.types import OpenAIBackendRole

# 核心 OASIS 导入
# 【!! 修正 !!】 我们需要从 oasis.social_agent.agent 导入原始的 SocialAgent
# 以便我们的 'SocialAgent' 子类可以正确地从它继承
from oasis.social_agent.agent import SocialAgent as OriginalOasisAgent
from oasis.social_agent.agent_action import SocialAction
from oasis.social_agent.agent_environment import SocialEnvironment
from oasis.social_platform import Channel
from oasis.social_platform.config import UserInfo
# from oasis.social_platform.typing import ActionType

if TYPE_CHECKING:
    from oasis.social_agent import AgentGraph

# --- 架构定义 ---
# BaseAgent 是所有 Agent (LLM 或 ABM) 的轻量级基类
# 它 *不* 继承 ChatAgent
class BaseAgent:
    def __init__(self,
                 agent_id: int,
                 user_info: UserInfo,
                 channel: Channel | None = None,
                 agent_graph: "AgentGraph" = None,
                 **kwargs): # 接受额外的参数 (例如 model) 但不使用
        self.social_agent_id = agent_id
        self.user_info = user_info
        self.channel = channel or Channel()
        self.env = SocialEnvironment(SocialAction(agent_id, self.channel))
        self.agent_graph = agent_graph
        self.group = user_info.profile["other_info"].get("group", "default")
    
    async def step(self):
        # 默认行为: ABM 和 LLM Agent 都会覆盖它
        await self.env.action.do_nothing()



class SocialAgent(OriginalOasisAgent):
    """
    Tier 1 (重) LLM Agent 的基类。
    我们重写 __init__ 只是为了方便地注入 persona。
    """
    def __init__(self,
                 agent_id: int,
                 user_info: UserInfo,
                 user_info_template: str | None = None,
                 *args, **kwargs):
        
        # 注入 Persona
        if user_info_template:
            base_persona = user_info.profile["other_info"].get("user_profile", "")
            user_info.profile["other_info"]["user_profile"] = \
                user_info_template.format(base_persona=base_persona)
        
        # 调用原始的 "重" __init__
        super().__init__(agent_id=agent_id, user_info=user_info, *args, **kwargs)
        self.group = user_info.profile["other_info"].get("group", "default")

    async def step(self):
        """
        覆盖 BaseAgent.step()。
        对于 Tier 1 Agent, "step" 意味着调用 LLM。
        (这是 OASIS SocialAgent.perform_action_by_llm 的逻辑)
        """
        env_prompt = await self.env.to_text_prompt()
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=(
                f"Please perform social media actions after observing the "
                f"platform environments. Notice that don't limit your "
                f"actions for example to just like the posts. "
                f"Here is your social media environment: {env_prompt}"))
        try:
            response = await self.astep(user_msg)
            for tool_call in response.info.get('tool_calls', []):
                # (TODO: Add logging if needed)
                pass
            return response
        except Exception as e:
            return e
class HeuristicAgent(BaseAgent):
    """
    Tier 2 (轻) Agent 的基类。
    它不继承 ChatAgent, 它的 __init__ 必须极快。
    """
    def __init__(self, *args, **kwargs):
        # 弹出 Tier 1 特有的参数
        kwargs.pop("model", None)
        kwargs.pop("available_actions", None)
        kwargs.pop("user_info_template", None)
        kwargs.pop("tools", None)
        kwargs.pop("single_iteration", None)
        kwargs.pop("interview_record", None)
        
        super().__init__(*args, **kwargs) #

# --- Tier 1 (LLM) 子类 ---

class AuthorityAgent(SocialAgent):
    """Tier 1: 权威媒体/大V (LLM-based)"""
    def __init__(self, *args, **kwargs):
        template = (
            "You are an authoritative media outlet or a top-tier influencer (Big V). "
            "Your purpose is to broadcast high-impact information, news, or official statements. "
            "You have a very large following and your posts are formal and objective.\n"
            "Your specific persona: {base_persona}"
        )
        kwargs["user_info_template"] = template
        super().__init__(*args, **kwargs)

class KOLAgent(SocialAgent):
    """Tier 1: 活跃KOL (LLM-based)"""
    def __init__(self, *args, **kwargs):
        template = (
            "You are an active and popular Key Opinion Leader (KOL). "
            "You have a significant following and post frequently to engage your audience. "
            "Your goal is to be interactive, set trends, and maintain high engagement.\n"
            "Your specific persona: {base_persona}"
        )
        kwargs["user_info_template"] = template
        super().__init__(*args, **kwargs)
        
class ActiveCreatorAgent(SocialAgent):
    """Tier 1: 活跃创作者 (LLM-based)"""
    def __init__(self, *args, **kwargs):
        template = (
            "You are a highly active content creator. You don't have many followers, "
            "but you post very frequently. "
            "You are casual, conversational, and share your daily life or opinions often.\n"
            "Your specific persona: {base_persona}"
        )
        kwargs["user_info_template"] = template
        super().__init__(*args, **kwargs)


class NormalUserAgent(SocialAgent):
    """Tier 1: 普通用户 (LLM-based)"""
    def __init__(self, *args, **kwargs):
        template = (
            "You are a typical social media consumer and participant.\n"
            "Your specific persona: {base_persona}"
        )
        kwargs["user_info_template"] = template
        super().__init__(*args, **kwargs)

class LurkerAgent(HeuristicAgent):
    """
    Tier 2: 潜水用户 (ABM - BCM, 基于帖子的演化)
    
    【!! 已更新: 按维度演化 !!】
    这个 Agent 拥有一个内部的、包含4个维度的态度分数 (字典),
    该分数会根据其在 'refresh' 动作中接收到的帖子的 *相应维度* 态度而随时间演化。
    """
    
    def __init__(
        self, 
        agent_id: str, 
        env: SocialAction, 
        # (移除了 initial_attitude: float, 因为我们将从 kwargs['user_info'] 中获取所有4个)
        # (kwargs 中会包含 user_info 等)
        **kwargs 
    ):
        """
        初始化一个有状态的 ABM Agent。

        参数:
            agent_id (str): Agent 的唯一 ID (应为 user_id 字符串)
            env (Action): Agent 的环境动作接口 (用于 refresh 和 DB 访问)
            kwargs (dict): 必须包含 'user_info'
        """
        # (kwargs 会包含 user_info, 传递给父类)
        super().__init__(agent_id=agent_id, env=env, **kwargs) 
        
        # --- [!! 关键修改: 初始化4个维度的态度 !!] ---
        try:
            profile_info = kwargs['user_info'].profile["other_info"]
            self.attitude_scores: Dict[str, float] = {
                'attitude_lifestyle_culture': profile_info.get('attitude_lifestyle_culture', 0.0),
                'attitude_sport_ent': profile_info.get('attitude_sport_ent', 0.0),
                'attitude_sci_health': profile_info.get('attitude_sci_health', 0.0),
                'attitude_politics_econ': profile_info.get('attitude_politics_econ', 0.0),
            }
        except KeyError as e:
            print(f"ERROR: Agent {self.agent_id} 初始化失败, 缺少 'user_info' in kwargs: {e}")
            # Fallback
            self.attitude_scores: Dict[str, float] = {
                'attitude_lifestyle_culture': 0.0,
                'attitude_sport_ent': 0.0,
                'attitude_sci_health': 0.0,
                'attitude_politics_econ': 0.0,
            }
        
        # ABM 模型超参数 (你可以调整这些)
        self.confidence_threshold: float = 0.5  # (ε) 信任阈值 (0 到 2.0)
        self.convergence_mu: float = 0.2        # (μ) 收敛速度 (0 到 0.5)
        self.base_action_prob: float = 0.05     # 基础动作概率 (5%)
        
        # (确保 env 是 SocialAction 的一个实例, 以便访问 platform)
        if not hasattr(self.env, 'platform') or not hasattr(self.env.platform, 'pl_utils'):
            raise ValueError("LurkerAgent 传入的 'env' 必须是 SocialAction 且已连接到 Platform")

        print(f"[ABM Lurker {self.agent_id}] created. Attitudes: {self.attitude_scores}")

    
    def get_attitudes_for_posts(self, post_ids: list[int]) -> list[Dict[str, float]]:
        """
        (内部函数) 从数据库查询给定 post_id 列表的 *完整* 态度向量 (4个维度)。
        
        
        返回:
            list[Dict[str, float]]: 
            例如: [{'lifestyle': 0.5, 'sport': 0.0, ...}, 
                   {'lifestyle': -0.1, 'sport': 0.8, ...}]
        """
        if not post_ids:
            return []

        # 1. 动态构建 SQL 查询
        cols = ['attitude_lifestyle_culture', 'attitude_sport_ent', 
                'attitude_sci_health', 'attitude_politics_econ']
        cols_sql = ", ".join(cols)
        placeholders = ','.join('?' for _ in post_ids)
        
        # --- [!! 关键修改: 移除了 UNION 和 ground_truth_post !!] ---
        # 查询 'post' 表, 因为 refresh 动作只返回模拟中的帖子
        sql = f"""
        SELECT {cols_sql} FROM post 
        WHERE post_id IN ({placeholders}) AND attitude_annotated = 1
        """
        # --- [!! 修改结束 !!] ---
        
        # 2. 访问数据库连接
        conn = self.env.platform.pl_utils.db_conn
        if not conn:
            print(f"ERROR: Agent {self.agent_id} DB connection is None.")
            return []
        
        cursor = None
        try:
            # 3. 执行查询
            cursor = conn.cursor()
            cursor.execute(sql, post_ids)
            # --- [!! 修改结束 !!] ---
            rows = cursor.fetchall()
            
            # 4. 构建态度向量列表
            attitude_vectors = []
            for row in rows:
                # row 是一个元组 (lifestyle, sport, sci, politics)
                post_vector = {cols[i]: float(row[i]) for i in range(len(cols))}
                attitude_vectors.append(post_vector)
            return attitude_vectors
        
        except Exception as e:
            print(f"ERROR: Agent {self.agent_id} DB query failed: {e}")
            return []
        finally:
            if cursor:
                cursor.close()


    async def step(self):
        """
        在每个模拟时间步 t 执行。
        1. Refresh 获取帖子 (消息 M_i,t)
        2. 查询帖子的 *4维度* 态度分数
        3. *分维度* 执行 BCM 演化 (f_update)
        4. 根据 *新的平均* 态度执行启发式动作 (like / do_nothing)
        """
        
        # --- 1. 感知: 从环境中获取帖子 (消息 M_i,t) ---
        try:
            refresh_response = await self.env.action.refresh()
            posts_list = refresh_response.get("posts", [])
            
            if not refresh_response.get("success") or not posts_list:
                await self.env.action.do_nothing()
                return
        except Exception as e:
            print(f"ERROR: Agent {self.agent_id} refresh failed: {e}")
            await self.env.action.do_nothing()
            return

        # --- 2. 态度演化 (ABM 核心: f_update) ---
        try:
            post_ids = [post['post_id'] for post in posts_list]
            
            # [MODIFIED] post_attitude_vectors 是一个 list[dict]
            post_attitude_vectors = self.get_attitudes_for_posts(post_ids)
            
            if post_attitude_vectors:
                
                # 复制当前状态, 以免在迭代时修改
                new_scores = self.attitude_scores.copy()
                
                # --- [!! 关键修改: 按维度独立演化 !!] ---
                for dim in self.attitude_scores.keys():
                    
                    a_i = self.attitude_scores[dim] # Agent 在此维度的当前分数
                    candidate_attitudes_for_dim = []
                    
                    # 遍历所有 "看到" 的帖子, 收集此维度的分数
                    for post_vec in post_attitude_vectors:
                        # 获取帖子 *在同一维度* 的分数
                        a_j = post_vec.get(dim, 0.0) 
                        
                        # 应用 BCM 信任检查
                        if abs(a_i - a_j) < self.confidence_threshold:
                            candidate_attitudes_for_dim.append(a_j)

                    # 如果此维度有可信的帖子
                    if candidate_attitudes_for_dim:
                        # 从 *可信* 的帖子中随机选择一个态度
                        target_attitude = random.choice(candidate_attitudes_for_dim)
                        
                        # a_i(t+1) = a_i(t) + μ * (a_j(t) - a_i(t))
                        delta_a = self.convergence_mu * (target_attitude - a_i)
                        new_score = a_i + delta_a
                        
                        # 将分数限制在 [-1.0, 1.0] 范围内
                        new_scores[dim] = max(-1.0, min(1.0, new_score))
                
                # (可选) 打印日志
                # if self.attitude_scores != new_scores:
                #    print(f"[ABM Evolve] Agent {self.agent_id} -> {new_scores}")
                
                # 原子性更新: 在所有计算完成后, 应用新状态
                self.attitude_scores = new_scores

        except Exception as e:
            # 确保演化失败不会导致模拟崩溃
            print(f"ERROR: Agent {self.agent_id} attitude evolution failed: {e}")

        # --- 3. 动作决策 (Heuristic 核心) ---
        # 潜水用户的动作概率。
        
        # [MODIFIED] 概率现在基于 *所有* 态度的平均极端程度
        overall_extremity = np.mean([abs(s) for s in self.attitude_scores.values()])
        action_probability = self.base_action_prob + (overall_extremity * 0.10)
        
        if random.random() < (1.0 - action_probability):
            # (1 - P) 的几率: 什么都不做
            await self.env.action.do_nothing()
        else:
            # (P) 的几率: 行动 (例如, 随机点赞)
            
            # 50% 几率在 refresh 成功后点赞
            if random.random() < 0.5:
                # 我们已经有了 posts_list, 不需要再次 refresh
                post_to_like = random.choice(posts_list)
                
                await self.env.action.like_post(post_to_like["post_id"])
            else:
                await self.env.action.do_nothing()
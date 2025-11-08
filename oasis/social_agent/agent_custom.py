# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# ... (Apache License) ...
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========

import random
import asyncio
from typing import TYPE_CHECKING, Optional, Union, List, Callable

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
from oasis.social_platform.typing import ActionType

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


# --- Tier 1: "重" LLM Agents ---
# (继承自 OASIS 原始的 "重" SocialAgent)

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
        
        super().__init__(*args, **kwargs) # 调用 BaseAgent.__init__

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


# --- Tier 2: "轻" ABM Agents (Rule-based) ---


class LurkerAgent(HeuristicAgent):
    """Tier 2: 潜水用户 (ABM)"""
    
    async def step(self):
        """
        一个简单的、基于规则的“潜水用户”行为。
        (你的要求是: 潜水用户使用基于规则的ABM)
        """
        # 95% 的几率什么都不做
        if random.random() < 0.95:
            await self.env.action.do_nothing()
        else:
            # 5% 的几率，他们会看帖子并可能点赞
            refresh_response = await self.env.action.refresh()
            if refresh_response.get("success") and refresh_response.get("posts") and random.random() < 0.5:
                posts_list = refresh_response["posts"]
                post_to_like = random.choice(posts_list)
                await self.env.action.like_post(post_to_like["post_id"])
            else:
                await self.env.action.do_nothing()

# --- Tier 3: NPCs ---
# (在此方案中, 没有 Tier 3 NPCs)
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
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

import json
from abc import ABC, abstractmethod
from string import Template

from oasis.social_agent.agent_action import SocialAction


class Environment(ABC):

    @abstractmethod
    def to_text_prompt(self) -> str:
        r"""Convert the environment to text prompt."""
        raise NotImplementedError


class SocialEnvironment(Environment):
    followers_env_template = Template("I have $num_followers followers.")
    follows_env_template = Template("I have $num_follows follows.")
    broadcast_env_template = Template(
        "You see the following global broadcast messages: $broadcasts"
    )

    posts_env_template = Template(
        "After refreshing, you see some posts $posts")

    groups_env_template = Template(
        "And there are many group chat channels $all_groups\n"
        "And You are already in some groups $joined_groups\n"
        "You receive some messages from them $messages\n"
        "You can join the groups you are interested, "
        "leave the groups you already in, send messages to the group "
        "you already in.\n"
        "You must make sure you can only send messages to the group you "
        "are already in")
    env_template = Template(
        "$groups_env\n"
        "$broadcast_env\n"
        "$posts_env\npick one you want to perform action that best "
        "reflects your current inclination based on your profile and "
        "posts content. Do not limit your action in just `like` to like posts")
 

    def __init__(self, action: SocialAction):
        self.action = action

    def get_posts_env(self, refresh_data: dict) -> str:
        r""" (已修改) 从已获取的 refresh_data 中格式化帖子部分 """
        # (移除了 await self.action.refresh())
        if refresh_data.get("success") and refresh_data.get("posts"):
            # (使用传入的 "posts" 键)
            posts_env = json.dumps(refresh_data["posts"], indent=4)
            posts_env = self.posts_env_template.substitute(posts=posts_env)
        else:
            posts_env = "After refreshing, there are no existing posts."
        return posts_env
    

    def get_broadcast_env(self, refresh_data: dict) -> str:
        r""" 从已获取的 refresh_data 中格式化广播消息部分 """
        broadcasts = refresh_data.get("broadcast_messages")
        
        if broadcasts:
            broadcast_env = json.dumps(broadcasts, indent=4)
            broadcast_env = self.broadcast_env_template.substitute(
                broadcasts=broadcast_env
            )
        else:
            broadcast_env = "" 
        return broadcast_env
  

    async def get_followers_env(self) -> str:
        # TODO: Implement followers env
        return self.followers_env_template.substitute(num_followers=0)

    async def get_follows_env(self) -> str:
        # TODO: Implement follows env
        return self.follows_env_template.substitute(num_follows=0)

    async def get_group_env(self) -> str:
        groups = await self.action.listen_from_group()
        if groups["success"]:
            all_groups = json.dumps(groups["all_groups"])
            joined_groups = json.dumps(groups["joined_groups"])
            messages = json.dumps(groups["messages"])
            groups_env = self.groups_env_template.substitute(
                all_groups=all_groups,
                joined_groups=joined_groups,
                messages=messages,
            )
        else:
            groups_env = "No groups."
        return groups_env

    # --- [!! 修改: 重构 to_text_prompt 以提高效率 !!] ---
    async def to_text_prompt(
        self,
        include_posts: bool = True,
        include_followers: bool = False,
        include_follows: bool = False,
    ) -> str:
        
        # 1. (效率优化) 只调用一次 refresh()
        refresh_data = {}
        if include_posts:
            try:
                refresh_data = await self.action.refresh()
            except Exception as e:
                print(f"Error during action.refresh(): {e}")
                # (确保 refresh_data 至少是一个空字典)
                refresh_data = {"success": False, "error": str(e), "posts": [], "broadcast_messages": []}

        # 2. 获取其他异步部分
        followers_env = (await self.get_followers_env()
                         if include_follows else "No followers.")
        follows_env = (await self.get_follows_env()
                       if include_followers else "No follows.")
        groups_env = await self.get_group_env()
        
        # 3. (修改) 调用非异步辅助函数
        posts_env = (self.get_posts_env(refresh_data) 
                     if include_posts else "")
        broadcast_env = (self.get_broadcast_env(refresh_data)
                         if include_posts else "")

        # 4. (修改) 注入所有 substitution
        return self.env_template.substitute(
            followers_env=followers_env,
            follows_env=follows_env,
            posts_env=posts_env,
            broadcast_env=broadcast_env, 
            groups_env=groups_env,
        )
import pandas as pd
import os
from collections import defaultdict
from pathlib import Path

class OasisUserBuilder:
    """
    将 oasis_user.py 的用户构建逻辑封装为类。
    用法:
        builder = OasisUserBuilder(
            profile_csv='data/user_profiles/oasis_twitter_users.csv',
            relation_csv='data/user_relationship/final_follow_list_AGGREGATED.csv',
            output_csv='data/oasis/oasis_agent_init.csv'
        )
        builder.run()
    """
    def __init__(self,
                 profile_csv: str = 'data/user_profiles/oasis_twitter_users.csv',
                 relation_csv: str = 'data/user_relationship/final_follow_list_AGGREGATED.csv',
                 output_csv: str = 'data/oasis/oasis_agent_init.csv'):
        self.PROFILE_CSV_PATH = profile_csv
        self.RELATION_CSV_PATH = relation_csv
        self.OUTPUT_CSV_PATH = output_csv

    def run(self):
        if not os.path.exists(self.PROFILE_CSV_PATH):
            raise FileNotFoundError(self.PROFILE_CSV_PATH)
        if not os.path.exists(self.RELATION_CSV_PATH):
            raise FileNotFoundError(self.RELATION_CSV_PATH)
        df_profiles = pd.read_csv(self.PROFILE_CSV_PATH, dtype={'user_id': str})
        df_profiles['bio'] = df_profiles['bio'].fillna('')
        df_profiles['user_char'] = df_profiles['bio']
        df_profiles['description'] = df_profiles['bio']
        df_profiles['followers_count'] = pd.to_numeric(df_profiles['followers_count'], errors='coerce').fillna(0)
        df_profiles['following_count'] = pd.to_numeric(df_profiles['following_count'], errors='coerce').fillna(0)
        df_profiles['posts_count'] = pd.to_numeric(df_profiles['posts_count'], errors='coerce').fillna(0)
        df_profiles['favorites_count'] = pd.to_numeric(df_profiles['favorites_count'], errors='coerce').fillna(0)
        followers_q99 = df_profiles['followers_count'].quantile(0.99)
        followers_q90 = df_profiles['followers_count'].quantile(0.90)
        posts_q90 = df_profiles['posts_count'].quantile(0.90)
        posts_q80 = df_profiles['posts_count'].quantile(0.80)
        followers_q50 = df_profiles['followers_count'].quantile(0.50)
        posts_q50 = df_profiles['posts_count'].quantile(0.50)
        def assign_user_group(row):
            if row['followers_count'] > followers_q99:
                return '权威媒体/大V'
            if (row['followers_count'] > followers_q90) and (row['posts_count'] > posts_q90):
                return '活跃KOL'
            if (row['followers_count'] <= followers_q50) and (row['posts_count'] <= posts_q50):
                return '潜水用户'
            if (row['followers_count'] <= followers_q90) and (row['posts_count'] > posts_q80):
                return '活跃创作者'
            return '普通用户'
        df_profiles['group'] = df_profiles.apply(assign_user_group, axis=1)
        original_id_to_agent_id = { original_id: agent_id for agent_id, original_id in enumerate(df_profiles['user_id']) }
        df_relations = pd.read_csv(self.RELATION_CSV_PATH, dtype={'user_id': str, 'all_followed_user_ids': str})
        following_map_original = defaultdict(list)
        for _, row in df_relations.iterrows():
            follower_id = row['user_id']
            followed_ids_str = row['all_followed_user_ids']
            if not follower_id or pd.isna(followed_ids_str) or followed_ids_str.strip() == '':
                continue
            parsed_ids = [uid.strip() for uid in followed_ids_str.split(' ') if uid.strip()]
            if parsed_ids:
                following_map_original[follower_id].extend(parsed_ids)
        def map_relations_to_agent_id(original_user_id):
            orig_follows_list = following_map_original.get(str(original_user_id), [])
            new_agent_id_follows_list = []
            for orig_id in orig_follows_list:
                followed_agent_id = original_id_to_agent_id.get(str(orig_id).strip())
                if followed_agent_id is not None:
                    new_agent_id_follows_list.append(followed_agent_id)
            if not new_agent_id_follows_list:
                return "[]"
            unique_list = list(set(new_agent_id_follows_list))
            return f"[{', '.join(map(str, unique_list))}]"
        df_final = df_profiles.copy()
        df_final['name'] = [f"user_{i}" for i in range(len(df_final))]
        df_final['following_agentid_list'] = df_final['user_id'].apply(map_relations_to_agent_id)
        output_columns = [
            'user_id', 'name', 'username', 'following_agentid_list',
            'user_char', 'description', 'group'
        ]
        missing_cols = [col for col in output_columns if col not in df_final.columns]
        if missing_cols:
            raise ValueError(f"缺少列: {missing_cols}")
        df_final_pruned = df_final[output_columns]
        Path(self.OUTPUT_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
        df_final_pruned.to_csv(self.OUTPUT_CSV_PATH, index_label='Unnamed: 0')


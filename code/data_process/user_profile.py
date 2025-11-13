import json
import pandas as pd
import numpy as np
from pathlib import Path
import random
from datetime import datetime
import logging
import sys

# 配置路径（相对于项目根目录）
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FOLDER = PROJECT_ROOT / "data" / "raw"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "user_profiles"
LOG_FOLDER = PROJECT_ROOT / "logs"

# 确保文件夹存在
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


class WeiboToOASISConverter:
    """将微博数据转换为OASIS平台格式的转换器"""
    
    def __init__(self):
        self.users_data = []
        self.processed_users = set()
        self.file_stats = {}
        
        # (已移除 personality_traits 和 interest_categories)
    
    def extract_users_from_json(self, json_data):
        """从JSON数据中提取用户信息"""
        users = []
        
        if 'authorContentPojo' in json_data:
            users.append(self.parse_user(json_data['authorContentPojo'], 'content_author'))
        
        if 'authorCommentPojo' in json_data:
            users.append(self.parse_user(json_data['authorCommentPojo'], 'comment_author'))
            
        if 'authorCommentForwardPojo' in json_data:
            users.append(self.parse_user(json_data['authorCommentForwardPojo'], 'forward_author'))
            
        return users
    
    def parse_user(self, user_data, user_type):
        """解析单个用户数据"""
        user_id = user_data.get('sjcjId', '')
        
        if user_id in self.processed_users:
            return None
        self.processed_users.add(user_id)
        
        user = {
            'user_id': user_id,
            'username': user_data.get('sjcjNickName', f'user_{user_id}'),
            'display_name': user_data.get('sjcjNickName', ''),
            'gender': self.map_gender(user_data.get('sjcjGender', 'unknown')),
            
            # 认证信息 (根据用户需求保留)
            'verified': user_data.get('sjcjVerified', False),
            'verified_type': user_data.get('sjcjVerifiedType', -1),
            
            'bio': user_data.get('sjcjDescription', ''),
            
            'followers_count': user_data.get('sjcjFollowersCount', 0),
            'following_count': user_data.get('sjcjFriendsCount', 0),
            'posts_count': user_data.get('sjcjStatusesCount', 0),
            'favorites_count': user_data.get('sjcjFavouritesCount', 0),
            
            'province': user_data.get('sjqxProvince', ''),
            'city': user_data.get('sjqxCity', ''),
            'location': user_data.get('sjcjLocation', ''),
            'ip_location': user_data.get('sjcjIpLocation', ''),
            
            'registration_time': self.format_timestamp(user_data.get('sjcjRegistrationTime')),
            'last_published': self.format_timestamp(user_data.get('sjqxLastPublished')),
            
            'source': user_data.get('sjqxSource', ''),
            'source_mobile': user_data.get('sjqxSourceMobileV2', ''),
            
            'profile_image_url': user_data.get('sjcjProfileImageUrl', ''),
            'user_type': user_type,
            
       
            
            # 保留的生成字段
            'influence_score': self.calculate_influence_score(user_data),
            'core_user': self.is_core_user(user_data)
        }
        
    
        
        return user
    
    def map_gender(self, gender):
        gender_map = {'m': 'male', 'f': 'female'}
        return gender_map.get(gender, 'unknown')
    
    def format_timestamp(self, timestamp):
        if not timestamp:
            return None
        try:
            return datetime.fromtimestamp(timestamp/1000).isoformat()
        except:
            return None
    
 
    
    def calculate_influence_score(self, user_data):
        """计算影响力分数 (保留)"""
        followers = user_data.get('sjcjFollowersCount', 0)
        posts = user_data.get('sjcjStatusesCount', 0)
        verified = user_data.get('sjcjVerified', False)
        
        score = np.log1p(followers) * 0.5 + np.log1p(posts) * 0.3
        
        if verified:
            score *= 1.5
            
        score = min(100, score * 5)
        
        return round(score, 2)
    
    def is_core_user(self, user_data):
        """判断是否为核心用户 (保留)"""
        followers = user_data.get('sjcjFollowersCount', 0)
        verified = user_data.get('sjcjVerified', False)
        
        if verified or followers > 10000:
            return True
        elif followers > 1000:
            return random.random() < 0.3
        else:
            return False
    
    def process_single_file(self, filepath):
        filename = filepath.name
        logging.info(f"Processing file: {filename}")
        
        line_count = 0
        error_count = 0
        users_in_file = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                try:
                    json_data = json.loads(line.strip())
                    users = self.extract_users_from_json(json_data)
                    
                    for user in users:
                        if user:
                            self.users_data.append(user)
                            users_in_file += 1
                            
                except json.JSONDecodeError as e:
                    error_count += 1
                    if error_count <= 5:  # 只记录前5个错误
                        logging.warning(f"{filename} Line {line_num}: JSON decode error")
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        logging.error(f"{filename} Line {line_num}: Error - {e}")
        
        self.file_stats[filename] = {
            'total_lines': line_count,
            'errors': error_count,
            'users_extracted': users_in_file
        }
        
        logging.info(f"✓ {filename}: {users_in_file} users from {line_count} lines")
    
    def process_all_files(self):
        txt_files = list(INPUT_FOLDER.glob('*.txt'))
        
        if not txt_files:
            logging.warning(f"No .txt files found in {INPUT_FOLDER}")
            return
        
        logging.info(f"Found {len(txt_files)} txt files")
        
        for i, filepath in enumerate(txt_files, 1):
            logging.info(f"\n[{i}/{len(txt_files)}] Processing...")
            self.process_single_file(filepath)
        
        logging.info(f"\n✓ Total unique users: {len(self.processed_users)}")
    
    def save_to_csv(self, filename='oasis_twitter_users.csv'):
        if not self.users_data:
            logging.warning("No data to save")
            return None
            
        output_path = OUTPUT_FOLDER / filename
        
        df = pd.DataFrame(self.users_data)
        df = df.sort_values('influence_score', ascending=False)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        logging.info(f"✓ Saved {len(df)} users to {output_path}")
        
        return df
    
    def generate_report(self):
        if not self.users_data:
            return "No data to summarize"
            
        df = pd.DataFrame(self.users_data)
        
        report = f"""
{'='*60}
OASIS Data Conversion Report
{'='*60}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Input: {INPUT_FOLDER}
Output: {OUTPUT_FOLDER}

FILE STATISTICS:
{'-'*60}
"""
        
        for filename, stats in self.file_stats.items():
            report += f"\n{filename}:\n"
            report += f"  Lines: {stats['total_lines']}, Errors: {stats['errors']}, Users: {stats['users_extracted']}\n"
        
        report += f"""
{'='*60}
USER STATISTICS:
{'-'*60}
Total Users: {len(df)}
Verified: {df['verified'].sum()} ({df['verified'].sum()/len(df)*100:.1f}%)
Core Users: {df['core_user'].sum()} ({df['core_user'].sum()/len(df)*100:.1f}%)

Gender: {dict(df['gender'].value_counts())}

Averages:
  Followers: {df['followers_count'].mean():.0f}
  Following: {df['following_count'].mean():.0f}
  Posts: {df['posts_count'].mean():.0f}
  Influence: {df['influence_score'].mean():.2f}
  
  (已移除 Age 统计)

Top Provinces:
{df['province'].value_counts().head(5).to_string()}
{'='*60}
"""
        # 注意: 上面 report 字符串中的 '  (已移除 Age 统计)' 只是一个注释，
        # 实际代码中已将 `  Age: {df['age'].mean():.1f}` 这一行删除。
        # 为避免混淆，我将把它从最终报告中也删除。
        
        # 修正 generate_report (删除 Age 行)
        df = pd.DataFrame(self.users_data)
        
        report = f"""
{'='*60}
OASIS Data Conversion Report
{'='*60}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Input: {INPUT_FOLDER}
Output: {OUTPUT_FOLDER}

FILE STATISTICS:
{'-'*60}
"""
        
        for filename, stats in self.file_stats.items():
            report += f"\n{filename}:\n"
            report += f"  Lines: {stats['total_lines']}, Errors: {stats['errors']}, Users: {stats['users_extracted']}\n"
        
        report += f"""
{'='*60}
USER STATISTICS:
{'-'*60}
Total Users: {len(df)}
Verified: {df['verified'].sum()} ({df['verified'].sum()/len(df)*100:.1f}%)
Core Users: {df['core_user'].sum()} ({df['core_user'].sum()/len(df)*100:.1f}%)

Gender: {dict(df['gender'].value_counts())}

Averages:
  Followers: {df['followers_count'].mean():.0f}
  Following: {df['following_count'].mean():.0f}
  Posts: {df['posts_count'].mean():.0f}
  Influence: {df['influence_score'].mean():.2f}

Top Provinces:
{df['province'].value_counts().head(5).to_string()}
{'='*60}
"""
        
        return report
    
    def save_report(self):
        report = self.generate_report()
        report_path = OUTPUT_FOLDER / 'conversion_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logging.info(f"✓ Report saved to {report_path}")
        print(report)


def main():
    print(f"输入: {INPUT_FOLDER}，输出: {OUTPUT_FOLDER}")
    
    converter = WeiboToOASISConverter()
    
    logging.info("Starting conversion...")
    converter.process_all_files()
    
    converter.save_to_csv()
    
    converter.save_report()
    
    print("\n✅ 转换完成！")
    print(f"📁 输出文件: {OUTPUT_FOLDER}")



if __name__ == "__main__":
    main()
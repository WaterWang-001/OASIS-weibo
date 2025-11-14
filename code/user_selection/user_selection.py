import pandas as pd
from pathlib import Path
import argparse
import sys
from typing import List, Any

# --- 配置路径 ---
# 假设此脚本位于 MARS/code/data_process/
PROJECT_ROOT = Path(__file__).resolve().parents[2] # MARS/code/
DEFAULT_PROFILES_CSV = PROJECT_ROOT / "data" / "output" / "user_profiles.csv" # 假设这是 data_process_pipeline.py 的默认输出

class UserSelector:
    """
    根据标准从用户配置文件CSV中加载和筛选用户。
    """
    
    def __init__(self, profiles_csv_path: str):
        self.csv_path = Path(profiles_csv_path)
        self.df = None
        
        if not self.csv_path.exists():
            print(f"错误: 找不到用户配置文件: {self.csv_path}", file=sys.stderr)
            raise FileNotFoundError(f"找不到文件: {self.csv_path}")
            
        try:
            self.df = pd.read_csv(self.csv_path)
            # 确保 user_id 列被正确读取
            if 'user_id' not in self.df.columns:
                print(f"错误: CSV文件中缺少 'user_id' 列。", file=sys.stderr)
                self.df = pd.DataFrame() # 置空
            else:
                # 确保 user_id 是字符串，以防止科学计数法问题
                self.df['user_id'] = self.df['user_id'].astype(str)
                
            print(f"成功加载 {len(self.df)} 条用户配置。")
            
        except Exception as e:
            print(f"加载 {self.csv_path} 时出错: {e}", file=sys.stderr)
            self.df = pd.DataFrame()

    def select_users(self, **criteria) -> List[str]:
        """
        根据动态标准筛选用户。
        
        使用方法:
        - 精确匹配: gender='female', verified=True
        - 最小阈值: min_followers_count=10000
        - 最大阈值: max_posts_count=500
        - 包含列表: province_in='北京,上海' (值是逗号分隔的字符串)
        
        返回: user_id 列表
        """
        if self.df.empty:
            print("警告: DataFrame 为空，无法筛选。", file=sys.stderr)
            return []
            
        query_parts = []
        
        for key, value in criteria.items():
            if key.startswith('min_'):
                col = key[4:]
                if col in self.df.columns:
                    # 确保比较列是数字
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    query_parts.append(f"`{col}` >= {value}")
                else:
                    print(f"警告: 列 '{col}' (用于 min_) 在CSV中未找到。", file=sys.stderr)
            
            elif key.startswith('max_'):
                col = key[4:]
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    query_parts.append(f"`{col}` <= {value}")
                else:
                    print(f"警告: 列 '{col}' (用于 max_) 在CSV中未找到。", file=sys.stderr)
            
            elif key.endswith('_in'):
                col = key[:-3]
                if col in self.df.columns:
                    # 将 '北京,上海' 转换为 ['北京', '上海']
                    val_list = [v.strip() for v in str(value).split(',')]
                    # !r 会在列表中每个元素周围加上引号，使其成为有效的查询
                    query_parts.append(f"`{col}` in {val_list!r}") 
                else:
                    print(f"警告: 列 '{col}' (用于 _in) 在CSV中未找到。", file=sys.stderr)
            
            else: # 精确匹配
                col = key
                if col in self.df.columns:
                    # !r 会自动处理字符串和布尔值 (e.g., 'female' vs True)
                    query_parts.append(f"`{col}` == {value!r}")
                else:
                    print(f"警告: 列 '{col}' (用于精确匹配) 在CSV中未找到。", file=sys.stderr)

        if not query_parts:
            print("未提供有效筛选条件。")
            return []
            
        query_string = " & ".join(query_parts)
        print(f"执行查询: {query_string}")
        
        try:
            filtered_df = self.df.query(query_string)
            user_ids = list(filtered_df['user_id'])
            return user_ids
        except Exception as e:
            print(f"查询失败: {e}", file=sys.stderr)
            print("请检查您的条件（例如，是否在字符串列上使用了 'min_'？）")
            return []

    def save_list(self, user_ids: List[str], output_path: str):
        """将 user_id 列表保存到纯文本文件，每行一个ID。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for user_id in user_ids:
                    f.write(f"{user_id}\n")
            print(f"已保存 {len(user_ids)} 个 user_id 到 {output_path}")
        except Exception as e:
            print(f"保存文件到 {output_path} 时出错: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="从用户配置文件CSV中筛选用户ID。")
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_PROFILES_CSV),
        help=f"用户配置CSV文件路径 (默认: {DEFAULT_PROFILES_CSV})"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="输出的 user_id 列表文件名 (例如: filtered_users.txt)"
    )
    parser.add_argument(
        'filters',
        nargs='*',
        help="筛选条件，格式为 'key=value'。例如: gender=female min_followers_count=10000"
    )
    
    args = parser.parse_args()
    
    # 解析筛选条件
    criteria = {}
    for f in args.filters:
        if '=' not in f:
            print(f"错误: 筛选器格式无效 '{f}'。必须是 'key=value'。", file=sys.stderr)
            continue
            
        key, value = f.split('=', 1)
        
        # 尝试自动转换类型
        try:
            # 尝试转为整数
            value = int(value)
        except ValueError:
            try:
                # 尝试转为浮点数
                value = float(value)
            except ValueError:
                # 检查布尔值
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                # 否则保持为字符串
        
        criteria[key] = value

    if not criteria:
        print("警告: 未提供任何筛选条件。将生成空文件。", file=sys.stderr)
        
    try:
        selector = UserSelector(profiles_csv_path=args.input_csv)
        user_ids = selector.select_users(**criteria)
        
        if user_ids:
            selector.save_list(user_ids, args.output_file)
        else:
            print("未找到匹配的用户。")
            # 也许仍然创建一个空文件
            selector.save_list([], args.output_file)
            
    except FileNotFoundError:
        sys.exit(1) # UserSelector 已经打印了错误
    except Exception as e:
        print(f"发生意外错误: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
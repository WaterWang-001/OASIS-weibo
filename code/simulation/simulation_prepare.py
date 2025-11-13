import argparse
import sys
from pathlib import Path

# 确保项目根在 sys.path，方便直接 import 包内模块
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.simulation.oasis_post import OasisPostProcessor
from code.simulation.oasis_sample import OasisSampler
from code.simulation.oasis_user import OasisUserBuilder
from code.simulation.oasis_attitude import OasisAttitudeProcessor

def main():
    parser = argparse.ArgumentParser(description="运行 MARS simulation 模块")
    parser.add_argument("mode", choices=["post","sample","user","attitude","all"], help="要运行的模块")
    parser.add_argument("--input-db", help="源 user_post DB 路径", default="data/user_post/user_post_database.db")
    parser.add_argument("--oasis-db", help="目标 oasis DB 路径", default="data/oasis/oasis_database.db")
    parser.add_argument("--out-dir", help="统一输出目录（部分模块使用）", default="data/oasis")
    parser.add_argument("--seed-size", type=int, default=3000)
    parser.add_argument("--avoid-unannotated", action="store_true")
    parser.add_argument("--api-key", help="AttitudeAnnotator API key", default="")
    args = parser.parse_args()

    if args.mode in ("post","all"):
        proc = OasisPostProcessor(
            source_db=args.input_db,
            oasis_db=args.oasis_db
        )
        proc.run()
    if args.mode in ("sample","all"):
        sampler = OasisSampler(
            source_csv=str(Path(args.out_dir)/"oasis_agent_init.csv"),
            source_db=args.oasis_db,
            target_csv=str(Path(args.out_dir)/f"oasis_agent_init_{args.seed_size}_random.csv"),
            target_db=str(Path(args.out_dir)/f"oasis_database_{args.seed_size}_random.db"),
            seed_size=args.seed_size,
            avoid_unannotated=args.avoid_unannotated
        )
        sampler.run()
    if args.mode in ("user","all"):
        builder = OasisUserBuilder(
            profile_csv=str(Path("data/user_profiles")/"oasis_twitter_users.csv"),
            relation_csv=str(Path("data/user_relationship")/"final_follow_list_AGGREGATED.csv"),
            output_csv=str(Path(args.out_dir)/"oasis_agent_init.csv")
        )
        builder.run()
    if args.mode in ("attitude","all"):
        att = OasisAttitudeProcessor(
            oasis_db_path=str(Path(args.out_dir)/"oasis_database.db"),
            user_csv_path=str(Path(args.out_dir)/"oasis_agent_init_3000_random.csv"),
            user_csv_output_path=str(Path(args.out_dir)/"oasis_agent_init_3000_random.csv"),
            api_key=args.api_key
        )
        import asyncio
        asyncio.run(att.run())

if __name__ == "__main__":
    main()
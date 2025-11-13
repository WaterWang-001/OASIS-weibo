import argparse
import sys
from pathlib import Path
from typing import Optional
from typing import Dict, List
import zipfile
import tempfile
import shutil
import os
import errno
from code.data_process.user_post import UserPostProcessor
from code.data_process.user_profile import WeiboToOASISConverter
from code.data_process.user_relationship import UserRelationshipPipeline 
from code.data_process import user_profile 

# 项目根
PROJECT_ROOT = Path(__file__).parent.parent


def make_default_paths(out_dir: Path):
    out_dir = Path(out_dir)
    return {
        "db_path": str(out_dir / "user_post_database.db"),
        "profiles_out": str(out_dir / "user_profiles.csv"),
        "matrix_out": str(out_dir / "attention_matrix_edges.csv"),
        "scores_out": str(out_dir / "total_attention_scores.csv"),
        "follow_out": str(out_dir / "user_follow_list.csv"),
    }


def unzip_and_flatten(input_dir: str, pattern: str = "*.zip", keep_original_zip: bool = True):
    """
    解压 input_dir 下所有 zip 文件，并把所有 .txt 文件扁平移动到 input_dir 根目录。
    - pattern: zip 文件匹配模式（默认 *.zip）
    - keep_original_zip: 是否保留原始 zip 文件（默认 True）
    行为：
      * 使用临时目录安全解压（防止 zip 内绝对路径问题）
      * 将解出的所有 .txt 文件移动到 input_dir，
        若遇到同名文件则自动加后缀避免覆盖，例如 name_1.txt
      * <--- 新增：自动跳过已处理过的 zip 文件（通过 .processed_zips.log 记录）
    """
    src = Path(input_dir)
    if not src.exists() or not src.is_dir():
        print(f"[unzip] Source directory not found: {src}")
        return


    log_file = src / ".processed_zips.log"
    processed_set = set()

    # <--- 新增：1. 读取已处理列表
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                processed_set = {line.strip() for line in f if line.strip()}
        except Exception as e:
            print(f"[unzip] Warning: Could not read log file {log_file}: {e}")

    zip_files = list(src.glob(pattern))
    if not zip_files:
        print("[unzip] No zip files found.")
        return

    src.mkdir(parents=True, exist_ok=True)
    
    # <--- 新增：2. 准备日志文件写入 (使用 'a' a+ 模式)
    try:
        log_f = open(log_file, "a", encoding="utf-8")
    except Exception as e:
        print(f"[unzip] Fatal: Could not open log file for writing {log_file}: {e}")
        return 

    print(f"[unzip] Found {len(zip_files)} zips. {len(processed_set)} already processed.")

    with log_f:
        for z in zip_files:
            # <--- 新增：3. 检查是否跳过
            if z.name in processed_set:
                print(f"[unzip] Skipping (already processed): {z.name}")
                continue

            try:
                print(f"[unzip] Processing: {z.name} ...") # <--- 新增：处理提示
                with tempfile.TemporaryDirectory() as td:
                    try:
                        with zipfile.ZipFile(z, "r") as zf:
                            # 安全解压到临时目录
                            zf.extractall(td)
                    except zipfile.BadZipFile:
                        print(f"[unzip] Skipping (bad zip): {z.name}")
                        continue # 跳过损坏的 zip

                    # 找到所有解压出的 .txt 文件并移动到 src
                    files_moved = 0 # <--- 新增：计数
                    for root, _, files in os.walk(td):
                        for fn in files:
                            if fn.lower().endswith(".txt"):
                                src_path = Path(root) / fn
                                dest = src / fn
                                if dest.exists():
                                    # 自动改名以避免覆盖
                                    base = dest.stem
                                    ext = dest.suffix
                                    i = 1
                                    while True:
                                        new_name = f"{base}_{i}{ext}"
                                        new_dest = src / new_name
                                        if not new_dest.exists():
                                            dest = new_dest
                                            break
                                        i += 1
                                try:
                                    shutil.move(str(src_path), str(dest))
                                    files_moved += 1
                                except Exception:
                                    # 若移动失败，尝试复制后删除
                                    try:
                                        shutil.copy2(str(src_path), str(dest))
                                        os.remove(str(src_path))
                                        files_moved += 1
                                    except Exception as move_e:
                                        print(f"[unzip] Error moving file {src_path}: {move_e}")
                                        pass
                    
                    print(f"[unzip] -> Moved {files_moved} .txt files from {z.name}") # <--- 新增：移动计数

        
                
                # <--- 新增：4. 如果成功（未抛出异常），记录到日志
                log_f.write(f"{z.name}\n")
                log_f.flush() # 确保立即写入
                
                # <--- 新增：5. （可选）删除
                if not keep_original_zip:
                    try:
                        z.unlink()
                    except Exception as del_e:
                        print(f"[unzip] Warning: Could not delete original zip {z.name}: {del_e}")
                        pass
                        
            except Exception as e:
                # 单个 zip 失败不要中断整个流程
                print(f"[unzip] ERROR processing {z.name} (will retry next time): {e}")
                # <--- 新增：打印更详细的错误
                import traceback
                traceback.print_exc()
                continue
    
    print("[unzip] Flattening complete.")


def run_posts(input_dir: str, db_path: str):
    # 在处理之前先解压并扁平化所有 zip（避免外部 sh）
    print("[posts] 1/2 Checking/Unzipping files...") 
    unzip_and_flatten(input_dir)
    print("[posts] 2/2 Running post processor...") 
    

    proc = UserPostProcessor(input_directory=input_dir, db_path=db_path)
    
    print(f"[posts] input={input_dir} -> db={db_path}")
    return proc.run()


def run_profiles(input_dir: str, profiles_out: Optional[str]):
    # 覆盖模块级常量（若传入）
    # <--- 修改：setattr 现在会作用于我们新导入的 user_profile 模块
    if input_dir:
        setattr(user_profile, "INPUT_FOLDER", Path(input_dir))
    if profiles_out:
        setattr(user_profile, "OUTPUT_FOLDER", Path(profiles_out))
        Path(profiles_out).mkdir(parents=True, exist_ok=True)

  
    conv = WeiboToOASISConverter()
    
    # <--- 修改：getattr 现在会作用于我们新导入的 user_profile 模块
    print(f"[profiles] input={input_dir} -> out={getattr(user_profile, 'OUTPUT_FOLDER', 'unspecified')}")
    conv.process_all_files()
    df = conv.save_to_csv()
    conv.save_report()
    return df


def run_relationships(input_dir: str, matrix_out: str, scores_out: str, follow_out: str, quantile: float):
    

    pipeline = UserRelationshipPipeline(input_directory=input_dir, global_noise_quantile=quantile)
    
    print(f"[relationships] input={input_dir} -> matrix={matrix_out}, scores={scores_out}, follow={follow_out}")
    edges_df, scores_series, follow_df = pipeline.run(output_matrix_csv=matrix_out,
                                                       output_scores_csv=scores_out,
                                                       output_follow_csv=follow_out)
    return edges_df, scores_series, follow_df


def run_all(input_dir: str, out_dir: str, quantile: float):
    print("0/4 - Checking/Unzipping files...") 
    unzip_and_flatten(input_dir)

    paths = make_default_paths(Path(out_dir))
    # Ensure base output dirs
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # <--- 修改：修复 profiles_out 是 csv 文件名而不是目录的问题
    Path(paths["profiles_out"]).parent.mkdir(parents=True, exist_ok=True)
    

    print("1/4 - posts -> db")
    run_posts(input_dir=input_dir, db_path=paths["db_path"])

    print("2/4 - profiles")
    # <--- 修改：run_profiles 期待的 profiles_out 是 *文件*路径，其父目录才是输出目录
    run_profiles(input_dir=input_dir, profiles_out=paths["profiles_out"])

    print("3/4 - relationships")
    run_relationships(input_dir=input_dir,
                      matrix_out=paths["matrix_out"],
                      scores_out=paths["scores_out"],
                      follow_out=paths["follow_out"],
                      quantile=quantile)

    print("✅ Pipeline completed. All outputs under:", out_dir)


def main():
    parser = argparse.ArgumentParser(description="简化版数据处理 pipeline（统一输出到一个 out 目录）")
    parser.add_argument("mode", choices=["all", "posts", "profiles", "relationships"], help="运行模式")
    parser.add_argument("--input", required=False, help="raw input folder")
    parser.add_argument("--out", required=False, help="统一输出目录 (默认: project/output)")
    parser.add_argument("--quantile", type=float, default=0.25, help="全局噪声分位数（用于关系/关注列表）")
    args = parser.parse_args()

    input_dir = args.input or str(PROJECT_ROOT / "data" / "raw")
    out_dir = args.out or str(PROJECT_ROOT / "output")
    paths = make_default_paths(Path(out_dir))

    try:
        if args.mode == "all":
            run_all(input_dir=input_dir, out_dir=out_dir, quantile=args.quantile)
        elif args.mode == "posts":
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            run_posts(input_dir=input_dir, db_path=paths["db_path"])
        elif args.mode == "profiles":
            # <--- 修改：确保 profile 输出*文件*的*父*目录存在
            Path(paths["profiles_out"]).parent.mkdir(parents=True, exist_ok=True)
            run_profiles(input_dir=input_dir, profiles_out=paths["profiles_out"])
        elif args.mode == "relationships":
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            run_relationships(input_dir=input_dir,
                              matrix_out=paths["matrix_out"],
                              scores_out=paths["scores_out"],
                              follow_out=paths["follow_out"],
                              quantile=args.quantile)
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
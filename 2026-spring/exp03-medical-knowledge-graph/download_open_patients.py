# download_open_patients.py
import os
import json
from datasets import load_dataset

def main():
    # 1. 下载 HuggingFace 上的 ncbi/Open-Patients 数据集
    print("正在从 HuggingFace 下载 ncbi/Open-Patients 数据集...")
    dataset = load_dataset("ncbi/Open-Patients", split="train")
    print(f"数据集条目数: {len(dataset)}")

    # 2. 确保 data/raw 目录存在
    output_path = os.path.join("data", "raw", "Open-Patients.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 3. 保存为 JSONL 文件（每行一个 JSON）
    print(f"正在保存为 JSONL 文件: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            # item 里应该只有 "_id" 和 "description"
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + "\n")

    print("保存完成！")
    print(f"最终文件位置: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()

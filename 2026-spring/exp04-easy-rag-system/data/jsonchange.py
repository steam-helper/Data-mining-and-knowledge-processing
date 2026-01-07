import json

# 用绝对路径尝试加载文件
file_path = "C:/Users/lh008/Desktop/exp04-easy-rag-system/data/processed_data.json"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"Loaded {len(data)} articles from {file_path}")
except FileNotFoundError:
    print(f"文件未找到: {file_path}")
except json.JSONDecodeError:
    print(f"JSON 解码错误: {file_path}")
except Exception as e:
    print(f"发生错误: {e}")

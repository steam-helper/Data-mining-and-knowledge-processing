# config.py
class Config:
    """
    模型配置类，包含所有可配置参数
    """
    # ========= 模型参数 =========
    # 建议用 1.5B 先跑通，3B 对显存和内存都比较吃
    model_name = "Qwen/Qwen2.5-1.5B"   # 或 "Qwen/Qwen2.5-3B"
    max_seq_length = 128               # 最大序列长度
    num_classes = 2                    # 二分类

    # 可选：精度与设备自动映射（如果你代码里会用到可以加上）
    torch_dtype = "bfloat16"           # 或 "float16" / "float32"
    device_map = "auto"                # 如果用 accelerate / device_map 的话

    # ========= 训练参数 =========
    batch_size = 4          # Qwen 比 BERT 大，batch_size 建议先减小
    learning_rate = 2e-5
    num_epochs = 3          # 先少跑几轮试通流程

    # ========= 路径配置 =========
    train_path = "C:/Users/lh008/Desktop/exp02-sentiment-classificationn/dataset/train.csv"
    dev_path = "C:/Users/lh008/Desktop/exp02-sentiment-classificationn/dataset/dev.csv"
    test_path = "C:/Users/lh008/Desktop/exp02-sentiment-classificationn/dataset/test.csv"
    model_save_path = "saved_models/qwen_sentiment_model.pth"

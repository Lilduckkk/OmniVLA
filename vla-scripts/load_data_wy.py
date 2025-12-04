import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # 关闭 TF 的 INFO / WARNING
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="   # 避免重复注册警告
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # 不用 oneDNN（可避免一些警告）
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)
import numpy as np
from transformers import AutoProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer

# 尝试导入你刚才保存的简化版 Dataset
try:
    from OmniVLA.prismatic.vla.datasets.navitrace_dataset import Navitrace_Dataset
    print("✅ 成功导入 NaviTrace_Dataset")
except ImportError:
    print("❌ 错误: 无法导入 NaviTrace_Dataset。请确保 navitrace_new_dataset.py 在当前目录下。")
    sys.exit(1)

def print_structure(data, indent=0):
    """递归打印数据结构的辅助函数"""
    spacing = " " * indent
    if isinstance(data, dict):
        print(f"{spacing}Type: dict, Keys: {list(data.keys())}")
        for k, v in data.items():
            print(f"{spacing}- Key: '{k}'")
            print_structure(v, indent + 4)
    elif isinstance(data, list):
        print(f"{spacing}Type: list, Length: {len(data)}")
        if len(data) > 0:
            print(f"{spacing}  First Element Preview:")
            print_structure(data[0], indent + 4)
    elif hasattr(data, 'size') and hasattr(data, 'mode'): # PIL Image
        print(f"{spacing}Type: PIL Image, Size: {data.size}, Mode: {data.mode}")
    elif isinstance(data, (int, float, str, bool)):
        print(f"{spacing}Type: {type(data).__name__}, Value: {data}")
    else:
        print(f"{spacing}Type: {type(data)}")

def main():
    # 配置路径
    DATA_ROOT = "data/data_splits/navitrace_dataset"
    MODEL_PATH = "./omnivla-original" 

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    base_tokenizer = processor.tokenizer
    action_tokenizer = ActionTokenizer(base_tokenizer)
    image_transform = processor.image_processor.apply_transform
    # 配置路径
    DATA_ROOT = "data/data_splits/navitrace_dataset"
    
    print("\n=== 开始最简数据调试 ===")
    
    # 1. 实例化 Dataset
    # 因为我们现在的 Dataset 只是为了看数据，不需要真的 Tokenizer，全传 None 即可
    try:
        dataset = Navitrace_Dataset(
            action_tokenizer=action_tokenizer,
            base_tokenizer=base_tokenizer, 
            image_transform=image_transform,
            prompt_builder_fn=PurePromptBuilder,
            dataset_name="navitrace",
            data_root_dir=DATA_ROOT,
            data_split_type="train",
            predict_stop_token=True,
        )
        print(f"✅ Dataset 实例化成功，样本总数: {len(dataset)}")
    except Exception as e:
        print(f"❌ Dataset 初始化失败: {e}")
        return

    # 2. 读取第 0 个样本
    if len(dataset) == 0:
        print("⚠️ 数据集为空！")
        return
        
    idx = 0
    print(f"\n[读取样本 {idx} 的原始内容]")
    sample = dataset[idx]
    # 先查看所有的字段
    # print(f"样本字段: {list(sample.keys())}")
    # # 3. 逐个字段打印详情,不要输出'image', 'segmentation_mask'等大数据，输出的简洁一点
    # for key, value in sample.items():
    #     if key in ['image', 'segmentation_mask']:
    #         print(f"- Key: '{key}' (Type: {type(value).__name__}, skipped detailed print)")
    #     else:
    #         print(f"- Key: '{key}' - Value Preview: {value}")

if __name__ == "__main__":
    main()
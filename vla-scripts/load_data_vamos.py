import sys
import os

# --- 1. 环境配置 (保持不变) ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from transformers import AutoProcessor
# 假设你的环境里有这些 prismatic 依赖
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer

# --- 2. 尝试导入 Vamos_Dataset ---
try:
    # 假设你把之前的类保存为了 vamos_dataset.py
    from prismatic.vla.datasets.vamos_dataset import Vamos_Dataset
    print("✅ 成功导入 Vamos_Dataset")
except ImportError:
    print("❌ 错误: 无法导入 Vamos_Dataset。请确保 vamos_dataset.py 在当前目录下，或者类名正确。")
    # 为了防止直接报错退出，这里允许你手动粘贴类定义调试，或者直接退出
    sys.exit(1)

def print_tensor_info(name, data):
    """辅助函数：简洁打印 Tensor/Array 信息"""
    if isinstance(data, (torch.Tensor, np.ndarray)):
        shape_str = str(list(data.shape))
        dtype_str = str(data.dtype)
        print(f"  - {name}: Shape={shape_str}, Dtype={dtype_str}")
    elif isinstance(data, list):
        print(f"  - {name}: List length={len(data)}")
    else:
        print(f"  - {name}: Type={type(data)}")

def main():
    # --- 3. 配置路径 ---
    # 指向你的 Parquet 数据集根目录
    DATA_ROOT = "data/data_splits/vamos_dataset" 
    # 指向你的模型权重目录 (用于加载 Tokenizer)
    MODEL_PATH = "./omnivla-original" 
    
    print(f"正在从 {MODEL_PATH} 加载 Processor...")
    
    try:
        # 加载 Processor 和 Tokenizer
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        base_tokenizer = processor.tokenizer
        action_tokenizer = ActionTokenizer(base_tokenizer)
        image_transform = processor.image_processor.apply_transform
    except Exception as e:
        print(f"❌ 模型组件加载失败 (可能是路径错误): {e}")
        return

    print("\n=== 开始 Vamos Dataset 数据调试 ===")
    
    # --- 4. 实例化 Dataset ---
    try:
        dataset = Vamos_Dataset(
            action_tokenizer=action_tokenizer,
            base_tokenizer=base_tokenizer, 
            image_transform=image_transform,
            prompt_builder_fn=PurePromptBuilder,
            data_root_dir=DATA_ROOT,
            data_split_type="train", # 确保目录下有 train 文件夹或直接是 parquet
            # 其他参数保持默认或根据需要调整
            len_traj_pred=8,
            dataset_name="vamos"
        )
        print(f"✅ Dataset 实例化成功，样本总数: {len(dataset)}")
    except Exception as e:
        print(f"❌ Dataset 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    if len(dataset) == 0:
        print("⚠️ 数据集为空，请检查路径下是否有 .parquet 文件。")
        return

    # --- 5. 循环检查前几个样本 ---
    # 只检查前 3 个样本，避免刷屏
    num_samples_to_check = 3
    print(f"\n[正在检查前 {num_samples_to_check} 个样本的数据结构]")

    for i in range(min(len(dataset), num_samples_to_check)):
        print(f"\n>>> Sample Index: {i}")
        try:
            sample = dataset[i]
            print(f"modality_id: {sample['modality_id']}")
            # 先查看所有的字段
            # print(f"样本字段: {list(sample.keys())}")
            # # 打印关键字段
            # # 1. 文本指令
            # print(f"  - lan_prompt: \"{sample.get('lan_prompt', 'N/A')}\"")
            
            # # 2. 图像 Tensor (pixel_values)
            # if 'pixel_values' in sample:
            #     print_tensor_info('pixel_values', sample['pixel_values'])
            
            # # 3. Input IDs (Tokenized input)
            # if 'input_ids' in sample:
            #     print_tensor_info('input_ids', sample['input_ids'])
            #     # 简单检查 label 是否有 -100
            #     labels = sample.get('labels', None)
            #     if labels is not None:
            #         valid_labels = (labels != -100).sum().item()
            #         print(f"  - labels: Valid tokens count = {valid_labels} (Active loss tokens)")

            # # 4. Actions (这是最重要的检查点)
            # if 'actions' in sample:
            #     print_tensor_info('actions', sample['actions'])
            #     # 检查 padding 是否生效 (我们设置了 MAX_TRAJECTORY_LENGTH=20)
            #     actions = sample['actions']
            #     # 打印前2个动作看看数值是否正常
            #     print(f"    -> First 2 actions: {actions[:2].tolist()}")
            
            # # 5. Modality ID
            # print(f"  - modality_id: {sample.get('modality_id', 'N/A')}")

        except Exception as e:
            print(f"❌ 读取样本 {i} 时发生错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
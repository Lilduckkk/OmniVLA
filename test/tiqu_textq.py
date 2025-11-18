# ===============================================================  # 分隔线：说明该文件用于OmniVLA推理
# OmniVLA Inference  # OmniVLA 推理
# ===============================================================  # 分隔线
#  # 说明
# Sample inference code for OmniVLA  # OmniVLA 的示例推理代码
# if you want to control the robot, you need to update the current state such as pose and image in "run_omnivla" and comment out "break" in "run".  # 如果你想控制机器人，需要在 run_omnivla 中更新当前状态（位姿、图像），并在 run 中注释掉 break。
#
# ---------------------------  # 分隔线：路径与系统设置
# Paths and System Setup  # 路径与系统设置
# ---------------------------  # 分隔线
import sys, os  # 导入系统与操作系统模块
sys.path.insert(0, '..')  # 将上一级目录加入模块搜索路径，便于本地包导入

import time, math, json  # 导入时间、数学、JSON 库
from typing import Optional, Tuple, Type, Dict  # 导入类型注解
from dataclasses import dataclass  # 导入数据类装饰器

import numpy as np  # 导入NumPy，用于数值计算
from PIL import Image  # PIL 图像处理
import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块
from torch.nn.utils.rnn import pad_sequence  # 序列填充工具
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式数据并行（别名）
import torchvision.transforms as transforms  # torchvision 图像变换
import matplotlib.pyplot as plt  # 画图工具
import utm  # 经纬度与 UTM 坐标转换

# ---------------------------  # 分隔线：自定义导入
# Custom Imports  # 自定义导入
# ---------------------------  # 分隔线
from prismatic.vla.action_tokenizer import ActionTokenizer  # 动作 tokenizer
from prismatic.models.projectors import ProprioProjector  # 本体感知投影器
from prismatic.models.action_heads import L1RegressionActionHead_idcat, L1RegressionDistHead  # 动作预测头
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1  # OpenVLA 模型
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # OpenVLA 配置
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor  # 处理器
from prismatic.models.backbones.llm.prompting import PurePromptBuilder  # 提示构建器
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask  # 掩码工具
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE  # 常量

from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor  # HF 自动类
from transformers.modeling_outputs import CausalLMOutputWithPast  # 新增这行
from typing import List


# ===============================================================  # 分隔线
# Utility Functions  # 工具函数
# ===============================================================  # 分隔线
def remove_ddp_in_checkpoint(state_dict: dict) -> dict:  # 去除DDP保存权重中的 "module." 前缀
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}  # 字典推导式移除前缀

def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:  # 加载某个模块的检查点
    if not os.path.exists(os.path.join(path, f"{module_name}--{step}_checkpoint.pt")) and module_name == "pose_projector":  # 若姿态投影器不存在，兼容旧命名
        module_name = "proprio_projector"  # 切换成 proprio_projector 名称
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")  # 拼接路径
    print(f"Loading checkpoint: {checkpoint_path}")  # 打印加载路径
    state_dict = torch.load(checkpoint_path, map_location=device)  # 加载权重到指定设备
    return remove_ddp_in_checkpoint(state_dict)  # 去除DDP前缀后返回

def count_parameters(module: nn.Module, name: str) -> None:  # 统计可训练参数数量
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)  # 累加可训练参数数量
    print(f"# trainable params in {name}: {num_params}")  # 打印可训练参数数

def init_module(  # 初始化模块（可选加载权重、转bf16、转到设备）
    module_class: Type[nn.Module],
    module_name: str,
    cfg: "InferenceConfig",
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
) -> DDP:
    module = module_class(**module_args)  # 根据提供参数实例化模块
    count_parameters(module, module_name)  # 打印参数数量

    if cfg.resume:  # 若开启恢复
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)  # 加载checkpoint
        module.load_state_dict(state_dict)  # 加载权重到模块

    if to_bf16:  # 若要求转换为 bfloat16
        module = module.to(torch.bfloat16)  # 转换数据类型
    module = module.to(device_id)  # 移动到指定设备（GPU/CPU）
    return module  # 返回模块


# ===============================================================
# 新增：纯语言推理类（仅输入文本，输出QKV）
# ===============================================================
# ...existing code...

class LanguageOnlyInference:
    """
    专门用于纯语言输入的推理类
    - 不使用图像、位姿、动作序列等多模态输入
    - 仅处理语言指令并提取文本QKV
    - 适用于测试语言模型分支的独立行为
    """
    IGNORE_INDEX = -100  # 与训练时保持一致

    def __init__(self, save_dir, lan_inst_prompt, processor, test_id, qkv_save_dir):
        """
        参数:
            save_dir: 保存结果的根目录
            lan_inst_prompt: 语言指令文本
            processor: HF处理器（用于文本tokenization）
            test_id: 当前测试编号
            qkv_save_dir: QKV保存目录
            action_seed: 随机种子（为None时使用test_id）
        """
        self.lan_inst_prompt = lan_inst_prompt
        self.processor = processor
        self.test_id = test_id
        self.qkv_save_dir = qkv_save_dir
        self.save_dir = save_dir
        # self.action_seed = action_seed if action_seed is not None else test_id
        
        # print(f"\n[LanguageOnlyInference] 初始化测试 #{test_id}")
        # print(f"  - 语言指令: {lan_inst_prompt}")
        # print(f"  - 随机种子: {self.action_seed}")
        # print(f"  - QKV保存路径: {qkv_save_dir}")

    def run(self):
        """执行一次纯语言推理"""
        print(f"\n[测试 #{self.test_id}] 开始纯语言推理...")
        self.run_language_only_inference()
        print(f"[测试 #{self.test_id}] 推理完成\n")

    def run_language_only_inference(self):
        """
        纯语言推理流程：
        1. 构建语言输入（不包含图像、位姿、动作）
        2. 前向传播（仅使用语言模型）
        3. 提取并保存文本QKV
        """
        batch = self.build_language_only_batch()
        self.run_forward_pass_language_only(batch)

    def build_language_only_batch(self) -> Dict[str, torch.Tensor]:
        """
        构建仅包含语言的输入batch（无动作、无图像、无位姿）
        
        返回:
            包含 input_ids, attention_mask, labels 的字典
        """
        print(f"  [构建输入] 纯语言模式（无动作序列）...")
        
        # 构建对话格式（无动作回复）
        conversation = [
            # {"from": "human", "value": f"What action should the robot take to {self.lan_inst_prompt}?"},
            {"from": "human", "value": f"{self.lan_inst_prompt}"},
            # {"from": "gpt", "value": ""},  # 空回复
        ]
        
        # 构建提示词
        prompt_builder = PurePromptBuilder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        final_prompt = prompt_builder.get_prompt()
        
        print(f"  [提示词] 前100字符: {final_prompt[:100]}")
        print(f"  [提示词] 总长度: {len(final_prompt)} 字符")
        
        # Tokenization
        tokenized = self.processor.tokenizer(final_prompt, add_special_tokens=True)
        input_ids = torch.tensor(tokenized.input_ids).unsqueeze(0)  # [1, seq_len]
        attention_mask = torch.ones_like(input_ids)
        
        # 构建labels（全部设为IGNORE_INDEX，因为无动作输出）
        labels = input_ids.clone()
        labels[:] = self.IGNORE_INDEX
        
        # print(f"  [Token化] input_ids 形状: {input_ids.shape}, 有效标签数: 0 (纯输入模式)")
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def run_forward_pass_language_only(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        执行纯语言模型的前向传播并提取QKV
        
        参数:
            batch: 包含 input_ids, attention_mask, labels 的字典
        """
        # print(f"  [前向传播] 使用纯语言输入...")
        
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # 直接调用语言模型（跳过VLA的多模态处理）
            language_model_output = vla.language_model(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                labels=batch["labels"].to(device_id),
                output_hidden_states=True,
                # output_attentions=True,
            )
        
        # 提取最后一层隐藏状态
        # last_hidden_states = language_model_output.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        # text_hidden_states = last_hidden_states[:, :-1]  # 去除最后的stop token
        # print(f"  [隐藏状态] 形状: {text_hidden_states.shape}")
        # # 提取QKV并保存
        # self._extract_and_save_qkv(text_hidden_states)

        # 提取所有层的QKV并保存        
        all_hidden_states = language_model_output.hidden_states  # 所有层的隐藏状态
        text_all_hidden_states = [hs[:, :-1] for hs in all_hidden_states] # 去除每层的stop token
        self._extract_and_save_all_qkv(text_all_hidden_states)



    def _extract_and_save_qkv(self, text_hidden_states: torch.Tensor) -> None:
        """
        从文本隐藏状态提取QKV并保存到本地
        
        参数:
            text_hidden_states: 文本部分的隐藏状态 [batch_size, seq_len, hidden_dim]
        """
        print(f"  [提取QKV] 从最后一层注意力层计算...")
        
        llm = vla.language_model
        
        # 获取最后一层
        if hasattr(llm, 'model') and hasattr(llm.model, 'layers'):
            last_layer = llm.model.layers[-1]
        elif hasattr(llm, 'layers'):
            last_layer = llm.layers[-1]
        else:
            print("  [错误] 无法找到语言模型的层结构")
            return
        
        # 提取self-attention层
        if not hasattr(last_layer, 'self_attn'):
            print("  [错误] 最后一层没有self_attn属性")
            return
        
        self_attn = last_layer.self_attn
        
        # 计算QKV
        if hasattr(self_attn, 'q_proj'):
            text_queries = self_attn.q_proj(text_hidden_states.to(torch.bfloat16))
            text_keys = self_attn.k_proj(text_hidden_states.to(torch.bfloat16))
            text_values = self_attn.v_proj(text_hidden_states.to(torch.bfloat16))
            
            print(f"  [QKV形状] Q: {text_queries.shape}, K: {text_keys.shape}, V: {text_values.shape}")
            
            # 统计QKV的数值范围
            # print(f"  [统计] Queries - 最小值: {text_queries.min().item():.4f}, "
            #       f"最大值: {text_queries.max().item():.4f}, "
            #       f"均值: {text_queries.mean().item():.4f}, "
            #       f"标准差: {text_queries.std().item():.4f}")
            # print(f"  [统计] Keys    - 最小值: {text_keys.min().item():.4f}, "
            #       f"最大值: {text_keys.max().item():.4f}, "
            #       f"均值: {text_keys.mean().item():.4f}, "
            #       f"标准差: {text_keys.std().item():.4f}")
            # print(f"  [统计] Values  - 最小值: {text_values.min().item():.4f}, "
            #       f"最大值: {text_values.max().item():.4f}, "
            #       f"均值: {text_values.mean().item():.4f}, "
            #       f"标准差: {text_values.std().item():.4f}")
            
            # 保存到本地
            queries_path = os.path.join(self.qkv_save_dir, f"queries_test_{self.test_id}.npy")
            keys_path = os.path.join(self.qkv_save_dir, f"keys_test_{self.test_id}.npy")
            values_path = os.path.join(self.qkv_save_dir, f"values_test_{self.test_id}.npy")
            
            np.save(queries_path, text_queries.detach().to(torch.float32).cpu().numpy())
            np.save(keys_path, text_keys.detach().to(torch.float32).cpu().numpy())
            np.save(values_path, text_values.detach().to(torch.float32).cpu().numpy())
            
            # print(f"  [保存] Queries -> {queries_path}")
            # print(f"  [保存] Keys    -> {keys_path}")
            # print(f"  [保存] Values  -> {values_path}")
        else:
            print("  [错误] self_attn层没有q_proj属性")
# ...existing code...
    def _extract_and_save_all_qkv(self, text_all_hidden_states: List[torch.Tensor]) -> None:
        """
        从语言模型的所有层提取 Q, K, V 并保存。
        每层都使用该层对应的 hidden state（即进入该层前的表示）。
        
        参数:
            text_all_hidden_states: list，每个元素为 [batch, seq_len, hidden_dim]
        """
        # print(f"  [提取QKV] 基于各层 hidden_states 计算所有层的 Q, K, V ...")

        llm = vla.language_model

        # 获取层结构
        if hasattr(llm, 'model') and hasattr(llm.model, 'layers'):
            layers = llm.model.layers
        elif hasattr(llm, 'layers'):
            layers = llm.layers
        else:
            print("  [错误] 无法找到语言模型层结构 (model.layers)")
            return

        num_layers = len(layers)
        print(f"  [信息] 检测到语言模型共有 {num_layers} 层")
        
        # 对齐 hidden_states 与层数
        # hidden_states 通常长度为 num_layers + 1（含 embedding 输出）
        if len(text_all_hidden_states) != num_layers + 1:
            print(f"  [警告] hidden_states 数量 {len(text_all_hidden_states)} 与层数 {num_layers} 不匹配，可能多出 embedding 层")
        assert len(text_all_hidden_states) >= num_layers, "hidden_states 数量不足"

        # 遍历每一层
        for idx, layer in enumerate(layers):
            if not hasattr(layer, 'self_attn'):
                print(f"  [警告] 第 {idx} 层无 self_attn，跳过")
                continue

            self_attn = layer.self_attn
            if not (hasattr(self_attn, 'q_proj') and hasattr(self_attn, 'k_proj') and hasattr(self_attn, 'v_proj')):
                print(f"  [警告] 第 {idx} 层缺少 q/k/v 投影层，跳过")
                continue

            # 当前层使用进入该层前的 hidden_state
            hidden_state = text_all_hidden_states[idx].to(torch.bfloat16)

            # 计算当前层 Q, K, V
            with torch.no_grad():
                q = self_attn.q_proj(hidden_state)
                k = self_attn.k_proj(hidden_state)
                v = self_attn.v_proj(hidden_state)

            # print(f"    [层 {idx:02d}] Q: {q.shape}, K: {k.shape}, V: {v.shape}")

            # 保存文件
            q_path = os.path.join(self.qkv_save_dir, f"layer_{idx:02d}_Q_test_{self.test_id}.npy")
            k_path = os.path.join(self.qkv_save_dir, f"layer_{idx:02d}_K_test_{self.test_id}.npy")
            v_path = os.path.join(self.qkv_save_dir, f"layer_{idx:02d}_V_test_{self.test_id}.npy")

            np.save(q_path, q.detach().to(torch.float32).cpu().numpy())
            np.save(k_path, k.detach().to(torch.float32).cpu().numpy())
            np.save(v_path, v.detach().to(torch.float32).cpu().numpy())

            # 释放显存
            del q, k, v, hidden_state
            torch.cuda.empty_cache()

        print(f"  [完成] 所有 {num_layers} 层 QKV 已保存至 {self.qkv_save_dir}")

# ===============================================================  # 分隔线
# Inference Configuration  # 推理配置
# ===============================================================  # 分隔线
class InferenceConfig:  # 推理配置类
    resume: bool = True  # 是否从 checkpoint 恢复
    # vla_path: str = "./omnivla-original"  # 原始模型路径（示例）
    # resume_step: Optional[int] = 120000     # 恢复步数（示例）
    vla_path: str = "./omnivla-finetuned-cast"     # 实际使用的模型路径
    resume_step: Optional[int] = 210000  # 恢复步数
    use_l1_regression: bool = True  # 使用 L1 回归动作头
    use_diffusion: bool = False  # 不使用扩散头
    use_film: bool = False  # 是否使用 FiLM
    num_images_in_input: int = 2  # 输入图像数量（当前+目标）
    use_lora: bool = True  # 使用 LoRA
    lora_rank: int = 32  # LoRA 秩
    lora_dropout: float = 0.0  # LoRA dropout

def define_model(cfg: InferenceConfig) -> None:  # 定义并加载模型、处理器、投影器与动作头
    cfg.vla_path = cfg.vla_path.rstrip("/")  # 去除末尾斜杠
    print(f"Loading OpenVLA Model `{cfg.vla_path}`")  # 打印加载信息

    # GPU setup  # GPU 设置
    device_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备
    torch.cuda.set_device(device_id)  # 设置当前 CUDA 设备（若可用）
    torch.cuda.empty_cache()  # 清空显存缓存

    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPOSE_DIM: {POSE_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )  # 打印重要常量

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)  # 将自定义模型注册到 HF 自动类（若非 Hub）
    AutoConfig.register("openvla", OpenVLAConfig)  # 注册配置
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)  # 注册图像处理器
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)  # 注册处理器 图像+文本
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)  # 注册模型 AutoModelForVision2Seq 是用于视觉 - 序列任务（如图像到文本生成、视觉指令跟随）的自动类
    
    # Load processor and VLA  # 加载处理器与模型
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)  # 从路径加载处理器
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device_id) #            trust_remote_code=True,  # 加载并放到设备上 加载 OmniVLA 的核心模型，用于从多模态输入预测机器人动作。
    
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)  # 设置输入图像数量
    vla.to(dtype=torch.bfloat16, device=device_id)  # 将模型转为 bfloat16 与目标设备
    
    pose_projector = init_module(  # 初始化位姿投影器
        ProprioProjector,
        "pose_projector",
        cfg,
        device_id,
        {"llm_dim": vla.llm_dim, "proprio_dim": POSE_DIM},            # 配置参数 
    )
    # 对应的输出
    # trainable params in pose_projector: 16801792
    # Loading checkpoint: ./omnivla-finetuned-cast/proprio_projector--210000_checkpoint.pt

    if cfg.use_l1_regression:  # 若使用 L1 回归动作头
        action_head = init_module(
            L1RegressionActionHead_idcat,
            "action_head",
            cfg,
            device_id,
            {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": ACTION_DIM},            # 输入/隐藏/动作维度
            to_bf16=True,  # 转 bfloat16
        )            
    # 对应的输出
    # trainable params in action_head: 100753412
    # Loading checkpoint: ./omnivla-finetuned-cast/action_head--210000_checkpoint.pt 
    
    # Get number of vision patches  # 获取视觉 patch 总数
    NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()     # patch 数量 * 图像数
    NUM_PATCHES += 1 #for goal pose  # 额外加 1（用于目标位姿 token）
    # print(f"Number of vision patches: {NUM_PATCHES}")  # 打印 patch 数量
    # Create Action Tokenizer  # 创建动作 tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)  # 以文本 tokenizer 为基础

    return vla, action_head, pose_projector, device_id, NUM_PATCHES, action_tokenizer, processor  # 返回模型与组件


# ===============================================================
# 主入口：仅运行纯语言推理（输出文本QKV）
# ===============================================================
# ...existing code...

if __name__ == "__main__":
    # 设置所有随机种子以确保完全可复现

    
    print("=" * 60)
    print("纯语言推理测试 - 仅使用 Language 分支")
    print("=" * 60)
    
    # 加载模型
    cfg = InferenceConfig()
    vla, action_head, pose_projector, device_id, NUM_PATCHES, action_tokenizer, processor = define_model(cfg)
    
    # ===========================
    # 测试1：相同语言指令的QKV一致性验证
    # ===========================
    test_times = 1
    qkv_save_dir = "./test/qkv_language_only"
    os.makedirs(qkv_save_dir, exist_ok=True)
    
    # lan_inst_prompt = "move toward white office cabinet"
    lan_inst_prompt = "move toward the blue bed and then toward the black chair"
    print(f"\n{'='*60}")
    print(f"测试1：相同语言指令的QKV一致性验证")
    print(f"{'='*60}")
    
    for test_id in range(test_times):
        print(f"\n{'='*60}")
        print(f"第 {test_id+1}/{test_times} 次测试")
        print(f"{'='*60}")
        
        inference = LanguageOnlyInference(
            save_dir="./inference",
            lan_inst_prompt=lan_inst_prompt,
            processor=processor,
            test_id=test_id,
            qkv_save_dir=qkv_save_dir,
            # action_seed=GLOBAL_SEED
        )
        inference.run()
    
    # 对比QKV差异
    # print("\n" + "="*60)
    # print("QKV差异对比结果（相同语言输入）")
    # print("="*60)
    # /home/pcl/OmniVLA/test/qkv_language_only/layer_00_Q_test_0.npy
    q0 = np.load(os.path.join(qkv_save_dir, "layer_00_Q_test_0.npy"))
    print("q0:",q0.shape)
    # for i in range(1, test_times):
    #     qi = np.load(os.path.join(qkv_save_dir, f"queries_test_{i}.npy"))
    #     ki = np.load(os.path.join(qkv_save_dir, f"keys_test_{i}.npy"))
    #     vi = np.load(os.path.join(qkv_save_dir, f"values_test_{i}.npy"))
        
    #     q_diff = np.mean(np.abs(q0 - qi))
    #     k_diff = np.mean(np.abs(k0 - ki))
    #     v_diff = np.mean(np.abs(v0 - vi))
        
    #     print(f"\n第0次 vs 第{i}次测试:")
    #     print(f"  Queries差异: {q_diff:.10f}")
    #     print(f"  Keys差异:    {k_diff:.10f}")
    #     print(f"  Values差异:  {v_diff:.10f}")
        
    #     if max(q_diff, k_diff, v_diff) < 1e-6:
    #         print(f"  → 结论: QKV完全一致 ✅ (确定性验证通过)")
    #     else:
    #         print(f"  → 结论: 存在差异 ❌ (可能有随机性或精度问题)")
    
    
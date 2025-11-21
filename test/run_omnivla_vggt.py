# ===============================================================
# OmniVLA Inference
# ===============================================================
# 
# Sample inference code for OmniVLA
# if you want to control the robot, you need to update the current state such as pose and image in "run_omnivla" and comment out "break" in "run".
#
# ---------------------------
# Paths and System Setup
# ---------------------------
import sys, os
sys.path.insert(0, '..')

import time, math, json
from typing import Optional, Tuple, Type, Dict
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import utm

# ---------------------------
# Custom Imports
# ---------------------------
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.projectors import ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead_idcat, L1RegressionDistHead
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1, OpenVLAForActionPrediction_MMNv2
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE

from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast  # 新增这行

# 获取 JanusVLN/src 的绝对路径（根据你的目录结构推导）
# 你的脚本路径：/home/pcl/OmniVLA/test/run_omnivla_vggt.py
# JanusVLN 路径：/home/pcl/JanusVLN
janus_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../JanusVLN/src"))
# 将路径加入 sys.path
sys.path.insert(0, janus_src_path)
from JanusVLN.src.qwen_vl.model.vggt.models.vggt import VGGT
from JanusVLN.src.qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images
import OmniVLA.test.VGGTv2 as VGGTv2

import random
import glob
import copy
import torch.nn.functional as F
import argparse
from pathlib import Path
import time

# 配置CUDA
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ===============================================================
# Utility Functions
# ===============================================================
def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    if not os.path.exists(os.path.join(path, f"{module_name}--{step}_checkpoint.pt")) and module_name == "pose_projector":
        module_name = "proprio_projector"
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)

def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")

def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: "InferenceConfig",
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
) -> DDP:
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return module

# ===============================================================
# Inference Class
# ===============================================================
class Inference:
    def __init__(self, save_dir, lan_inst_prompt, goal_utm, goal_compass, goal_image_PIL, action_tokenizer, processor):
        self.tick_rate = 3
        self.lan_inst_prompt = lan_inst_prompt
        self.goal_utm = goal_utm
        self.goal_compass = goal_compass
        self.goal_image_PIL = goal_image_PIL
        self.action_tokenizer = action_tokenizer
        self.processor = processor
        self.count_id = 0
        self.linear, self.angular = 0.0, 0.0
        self.datastore_path_image = save_dir
        self.vggt_feature = None
    # ----------------------------
    # Static Utility Methods
    # ----------------------------
    @staticmethod
    def calculate_relative_position(x_a, y_a, x_b, y_b):
        return x_b - x_a, y_b - y_a

    @staticmethod
    def rotate_to_local_frame(delta_x, delta_y, heading_a_rad):
        rel_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        rel_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return rel_x, rel_y

    # ----------------------------
    # Main Loop
    # ----------------------------
    def run(self):
        loop_time = 1 / self.tick_rate
        start_time = time.time()
        while True:
            if time.time() - start_time > loop_time:
                self.tick()
                start_time = time.time()
                break

    def tick(self):
        self.linear, self.angular = self.run_omnivla()

    # ----------------------------
    # OmniVLA Inference
    # ----------------------------
    def run_omnivla(self):
        thres_dist = 30.0
        metric_waypoint_spacing = 0.1

        # Load current GPS & heading
        current_lat = 37.87371258374039
        current_lon = -122.26729417226024
        current_compass = 270.0
        cur_utm = utm.from_latlon(current_lat, current_lon)
        cur_compass = -float(current_compass) / 180.0 * math.pi  # inverted compass

        # Local goal position
        delta_x, delta_y = self.calculate_relative_position(
            cur_utm[0], cur_utm[1], self.goal_utm[0], self.goal_utm[1]
        )
        relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, cur_compass)
        radius = np.sqrt(relative_x**2 + relative_y**2)
        if radius > thres_dist:
            relative_x *= thres_dist / radius
            relative_y *= thres_dist / radius

        goal_pose_loc_norm = np.array([
            relative_y / metric_waypoint_spacing,
            -relative_x / metric_waypoint_spacing,
            np.cos(self.goal_compass - cur_compass),
            np.sin(self.goal_compass - cur_compass)
        ])

        # Load current image
        current_image_path = "./inference/current_img.jpg"
        current_image_PIL = Image.open(current_image_path).convert("RGB")

        # Language instruction
        lan_inst = self.lan_inst_prompt if lan_prompt else "xxxx"

        # Prepare batch
        batch = self.data_transformer_omnivla(
            current_image_PIL, lan_inst, self.goal_image_PIL, goal_pose_loc_norm,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor
        )

        # Run forward pass
        actions, modality_id = self.run_forward_pass(
            vla=vla.eval(),
            action_head=action_head.eval(),
            noisy_action_projector=None,
            pose_projector=pose_projector.eval(),
            batch=batch,
            action_tokenizer=self.action_tokenizer,
            device_id=device_id,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_patches=NUM_PATCHES,
            compute_diffusion_l1=False,
            num_diffusion_steps_train=None,
            mode="train",
            idrun=self.count_id,
        )
        self.count_id += 1

        waypoints = actions.float().cpu().numpy()

        # Select waypoint
        waypoint_select = 4
        chosen_waypoint = waypoints[0][waypoint_select].copy()
        chosen_waypoint[:2] *= metric_waypoint_spacing
        dx, dy, hx, hy = chosen_waypoint

        # PD controller
        EPS = 1e-8
        DT = 1 / 3
        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            linear_vel_value = 0
            angular_vel_value = 1.0 * clip_angle(np.arctan2(hy, hx)) / DT
        elif np.abs(dx) < EPS:
            linear_vel_value = 0
            angular_vel_value = 1.0 * np.sign(dy) * np.pi / (2 * DT)
        else:
            linear_vel_value = dx / DT
            angular_vel_value = np.arctan(dy / dx) / DT

        linear_vel_value = np.clip(linear_vel_value, 0, 0.5)
        angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)

        # Velocity limitation
        maxv, maxw = 0.3, 0.3
        if np.abs(linear_vel_value) <= maxv:
            if np.abs(angular_vel_value) <= maxw:
                linear_vel_value_limit = linear_vel_value
                angular_vel_value_limit = angular_vel_value
            else:
                rd = linear_vel_value / angular_vel_value
                linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                angular_vel_value_limit = maxw * np.sign(angular_vel_value)
        else:
            if np.abs(angular_vel_value) <= 0.001:
                linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                angular_vel_value_limit = 0.0
            else:
                rd = linear_vel_value / angular_vel_value
                if np.abs(rd) >= maxv / maxw:
                    linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                    angular_vel_value_limit = maxv * np.sign(angular_vel_value) / np.abs(rd)
                else:
                    linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                    angular_vel_value_limit = maxw * np.sign(angular_vel_value)

        # Save behavior
        self.save_robot_behavior(
            current_image_PIL, self.goal_image_PIL, goal_pose_loc_norm, waypoints[0],
            linear_vel_value_limit, angular_vel_value_limit, metric_waypoint_spacing, modality_id.cpu().numpy()
        )

        print("linear angular", linear_vel_value_limit, angular_vel_value_limit)
        return linear_vel_value_limit, angular_vel_value_limit

    # ----------------------------
    # Save Robot Behavior Visualization
    # ----------------------------
    def save_robot_behavior(self, cur_img, goal_img, goal_pose, waypoints,
                            linear_vel, angular_vel, metric_waypoint_spacing, mask_number):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2, 2)
        ax_ob = fig.add_subplot(gs[0, 0])
        ax_goal = fig.add_subplot(gs[1, 0])
        ax_graph_pos = fig.add_subplot(gs[:, 1])

        ax_ob.imshow(np.array(cur_img).astype(np.uint8))
        ax_goal.imshow(np.array(goal_img).astype(np.uint8))

        x_seq = waypoints[:, 0] #generated trajectory is on the robot coordinate. X is front and Y is left. 
        y_seq_inv = -waypoints[:, 1]           
        ax_graph_pos.plot(np.insert(y_seq_inv, 0, 0.0), np.insert(x_seq, 0, 0.0), linewidth=4.0, markersize=12, marker='o', color='blue')

        # Mask annotation
        mask_type = int(mask_number[0])
        mask_texts = [
            "satellite only", "pose and satellite", "satellite and image", "all",
            "pose only", "pose and image", "image only", "language only", "language and pose"
        ]
        if mask_type < len(mask_texts):
            ax_graph_pos.annotate(mask_texts[mask_type], xy=(1.0, 0.0), xytext=(-20, 20), fontsize=18, textcoords='offset points')

        ax_ob.set_title("Egocentric current image", fontsize=18)
        ax_goal.set_title("Egocentric goal image", fontsize=18)
        ax_graph_pos.tick_params(axis='x', labelsize=15) 
        ax_graph_pos.tick_params(axis='y', labelsize=15) 
        
        if int(mask_number[0]) == 1 or int(mask_number[0]) == 3 or int(mask_number[0]) == 4 or int(mask_number[0]) == 5 or int(mask_number[0]) == 8:
            ax_graph_pos.plot(-goal_pose[1], goal_pose[0], marker = '*', color='red', markersize=15)  
        else:                           
            ax_graph_pos.set_xlim(-3.0, 3.0)
            ax_graph_pos.set_ylim(-0.1, 10.0)
        ax_graph_pos.set_xlim(-3.0, 3.0)
        ax_graph_pos.set_ylim(-0.1, 10.0)
                        
        ax_graph_pos.set_title("Normalized generated 2D trajectories from OmniVLA", fontsize=18)
        
        save_path = os.path.join(self.datastore_path_image, f"{self.count_id}_ex.jpg")
        plt.savefig(save_path)

    # ----------------------------
    # Custom Collator
    # ----------------------------
    def collator_custom(self, instances, model_max_length, pad_token_id, padding_side="right", pixel_values_dtype=torch.float32):
        IGNORE_INDEX = -100
        input_ids = pad_sequence([inst["input_ids"] for inst in instances], batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence([inst["labels"] for inst in instances], batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = input_ids[:, :model_max_length], labels[:, :model_max_length]
        attention_mask = input_ids.ne(pad_token_id)

        pixel_values = [inst["pixel_values_current"] for inst in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [inst["dataset_name"] for inst in instances]
        else:
            dataset_names = None

        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_goal" in instances[0]:
                pixel_values_goal = [inst["pixel_values_goal"] for inst in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_goal)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type: {type(pixel_values)}")

        actions = torch.stack([torch.from_numpy(np.copy(inst["actions"])) for inst in instances])
        goal_pose = torch.stack([torch.from_numpy(np.copy(inst["goal_pose"])) for inst in instances])

        output = dict(
            pixel_values=pixel_values.to(),
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            goal_pose=goal_pose,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output

    # ----------------------------
    # Transform Data to Dataset Format
    # ----------------------------
    def transform_datatype(self, inst_obj, actions, goal_pose_cos_sin,
                           current_image_PIL, goal_image_PIL, prompt_builder, action_tokenizer,
                           base_tokenizer, image_transform, predict_stop_token=True):
        IGNORE_INDEX = -100
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(action_tokenizer(future_actions))
        current_action_string = action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        if inst_obj == "xxxx":
            conversation = [
                {"from": "human", "value": "No language instruction"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {inst_obj}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]

        prompt_builder = prompt_builder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = torch.tensor(base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX
        if not predict_stop_token:
            labels[-1] = IGNORE_INDEX

        pixel_values_current = image_transform(current_image_PIL)
        pixel_values_goal = image_transform(goal_image_PIL)
        dataset_name = "lelan"

        return dict(
            pixel_values_current=pixel_values_current,
            pixel_values_goal=pixel_values_goal,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=torch.as_tensor(actions),
            goal_pose=goal_pose_cos_sin,
            img_PIL=current_image_PIL,
            inst=inst_obj,
        )

    # ----------------------------
    # Data Transformer for OmniVLA
    # ----------------------------
    def data_transformer_omnivla(self, current_image_PIL, lan_inst, goal_image_PIL, goal_pose_loc_norm,
                                 prompt_builder, action_tokenizer, processor):
        actions = np.random.rand(8, 4)  # dummy actions
        goal_pose_cos_sin = goal_pose_loc_norm

        batch_data = self.transform_datatype(
            lan_inst, actions, goal_pose_cos_sin,
            current_image_PIL, goal_image_PIL,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
        )

        batch = self.collator_custom(
            instances=[batch_data],
            model_max_length=processor.tokenizer.model_max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            padding_side="right"
        )
        return batch

    # ----------------------------
    # Run Forward Pass
    # ----------------------------
    def run_forward_pass(self, vla, action_head, noisy_action_projector, pose_projector,
                         batch, action_tokenizer, device_id, use_l1_regression, use_diffusion,
                         use_film, num_patches, compute_diffusion_l1=False,
                         num_diffusion_steps_train=None, mode="vali", idrun=0) -> Tuple[torch.Tensor, Dict[str, float]]:

        metrics = {}
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

        # Determine modality
        if satellite and not lan_prompt and not pose_goal and not image_goal:
            modality_id = torch.as_tensor([0], dtype=torch.float32)
        elif satellite and not lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([1], dtype=torch.float32)
        elif satellite and not lan_prompt and not pose_goal and image_goal:
            modality_id = torch.as_tensor([2], dtype=torch.float32)
        elif satellite and not lan_prompt and pose_goal and image_goal:
            modality_id = torch.as_tensor([3], dtype=torch.float32)
        elif not satellite and not lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([4], dtype=torch.float32)
        elif not satellite and not lan_prompt and pose_goal and image_goal:
            modality_id = torch.as_tensor([5], dtype=torch.float32)
        elif not satellite and not lan_prompt and not pose_goal and image_goal:
            modality_id = torch.as_tensor([6], dtype=torch.float32)
        elif not satellite and lan_prompt and not pose_goal and not image_goal:
            modality_id = torch.as_tensor([7], dtype=torch.float32)
        elif not satellite and lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([8], dtype=torch.float32)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                modality_id=modality_id.to(torch.bfloat16).to(device_id),
                labels=batch["labels"].to(device_id),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(device_id),
                proprio_projector=pose_projector,
                noisy_actions=noisy_actions if use_diffusion else None,
                noisy_action_projector=noisy_action_projector if use_diffusion else None,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
                use_film=use_film,
                vggt_feature=self.vggt_feature,
            )

        # Prepare data for metrics
        ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
        # ===================== 关键修改 1：获取原始文本序列的真实长度 =====================
        # 动作掩码的长度 = 原始文本序列长度（52），直接从掩码中获取
        text_original_len = current_action_mask.shape[1]  # 52
        # ==============================================================================   
 
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)

        
        # ===================== 关键修改 2：动态获取拼接后的特征数量 =====================
        # 从模型输出的 projector_features 中获取（拼接后的真实特征数量，如 1539）
        # projector_features 是你拼接后的 projected_patch_embeddings，形状为 (1, 1539, 4096)
        projected_features_len = output.projector_features.shape[1]  # 1539
        # ==============================================================================

        # ===================== 关键修改 3：精准计算文本截取范围 =====================
        # text_start_idx = 1 + projected_features_len  # 文本开始位置（1+BOS + 拼接特征）
        text_start_idx = projected_features_len
        print("text_start_idx:", text_start_idx)
        # 截取文本部分：从 text_start_idx 到 text_end_idx，长度=text_original_len（52）
        text_hidden_states = last_hidden_states[:, text_start_idx:-1]
        # ==============================================================================

        print("last_hidden_states shape:", last_hidden_states.shape)
        # Get hidden states for text portion of prompt+response (after the vision patches)
        # text_hidden_states = last_hidden_states[:, num_patches:-1]
        # 我想知道:, num_patches:-1表示多少
        
        # Get hidden states for action portion of response
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )  # (B, act_chunk_len, D)

        with torch.no_grad():
            predicted_actions = action_head.predict_action(actions_hidden_states, modality_id.to(torch.bfloat16).to(device_id))                                 

        # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
        return predicted_actions, modality_id

                
# ===============================================================
# Inference Configuration
# ===============================================================
class InferenceConfig:
    resume: bool = True
    # vla_path: str = "./omnivla-original"
    # resume_step: Optional[int] = 120000    
    vla_path: str = "./omnivla-finetuned-cast"    
    resume_step: Optional[int] = 210000
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

def define_model(cfg: InferenceConfig) -> None:
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Loading OpenVLA Model `{cfg.vla_path}`")

    # GPU setup
    device_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPOSE_DIM: {POSE_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv2)
    
    # Load processor and VLA
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device_id) #            trust_remote_code=True,
    
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.to(dtype=torch.bfloat16, device=device_id)
    
    pose_projector = init_module(
        ProprioProjector,
        "pose_projector",
        cfg,
        device_id,
        {"llm_dim": vla.llm_dim, "proprio_dim": POSE_DIM},            
    )
    
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead_idcat,
            "action_head",
            cfg,
            device_id,
            {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": ACTION_DIM},            
            to_bf16=True,
        )            
 
    # Get number of vision patches
    NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()    
    NUM_PATCHES += 1 #for goal pose

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    return vla, action_head, pose_projector, device_id, NUM_PATCHES, action_tokenizer, processor

# ====================================== VGGT ======================================
# 新增
def parse_args():
    parser = argparse.ArgumentParser(description="VGGT with Text-Image Matching")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_ba", action="store_true", default=False)
    parser.add_argument("--max_reproj_error", type=float, default=8.0)
    parser.add_argument("--shared_camera", action="store_true", default=False)
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE")
    parser.add_argument("--vis_thresh", type=float, default=0.2)
    parser.add_argument("--query_frame_num", type=int, default=8)
    parser.add_argument("--max_query_pts", type=int, default=4096)
    parser.add_argument("--fine_tracking", action="store_true", default=True)
    parser.add_argument("--conf_thres_value", type=float, default=5.0)
    # 新增路径参数，保留原默认值
    parser.add_argument("--scene_dir", type=str, 
                        default="../JanusVLN/images/images_wy/images",
                        help="图像场景目录")
    parser.add_argument("--text_q_path", type=str,
                        default="test/qkv_language_only/layer_31_Q_test_0.npy",
                        help="文本Q路径")
    return parser.parse_args()

def load_text_q(text_q_path, device, dtype) -> torch.Tensor:
    """加载文本Q矩阵（修改：接收路径参数）"""
    if not os.path.exists(text_q_path):
        raise FileNotFoundError(f"文本Q文件不存在: {text_q_path}")
    
    text_q_np = np.load(text_q_path)
    text_q = torch.from_numpy(text_q_np).to(dtype).to(device)
    print(f"✅ 加载文本Q: 形状={text_q.shape}, 数据类型={text_q.dtype}")
    return text_q


def split_image_kv_per_image(past_kv, num_images):
    """按实际图像数量拆分KV（自动适配图像数量）"""
    layer_k_list = []
    layer_v_list = []
    
    for layer_idx, (k, v) in enumerate(past_kv):
        if k is None or v is None:
            continue
        
        # 打印第一层信息
        if layer_idx == 0:
            print(f"🔍 第一层 K 形状: K={k.shape}, 数据类型: K={k.dtype}")
            print(f"🔍 第一层 V 形状: V={v.shape}, 数据类型: V={v.dtype}")
        
        # 按图像维度（第3维）拆分
        image_ks = []
        image_vs = []
        for img_idx in range(num_images):
            image_k = k[:, :, img_idx, :, :]  # [1, 16, seq_len, 64]
            image_v = v[:, :, img_idx, :, :]  # [1, 16, seq_len, 64]
            image_ks.append(image_k)
            image_vs.append(image_v)
        
        layer_k_list.append(image_ks)
        layer_v_list.append(image_vs)
    
    return layer_k_list, layer_v_list


def compute_text_image_matching(
    text_q: torch.Tensor,
    layer_k_list: list,
    layer_v_list: list,
    num_heads: int = 16,
    num_images: int = None,
    d_model: int = 4096
) -> list[float]:
    """计算文本Q与每张图像KV的匹配度（自动适配图像数量）"""
    d_head_text = d_model // num_heads  # 256
    batch_size, seq_len_text, _ = text_q.shape
    device = text_q.device
    dtype = text_q.dtype
    
    # 文本Q拆分多头
    text_q_heads = text_q.view(batch_size, seq_len_text, num_heads, d_head_text)
    text_q_heads = text_q_heads.transpose(1, 2)
    print(f"文本Q多头形状: {text_q_heads.shape}, 数据类型: {text_q_heads.dtype}")
    
    # 存储每层的匹配分数
    num_layers = len(layer_k_list)
    layer_scores = np.zeros((num_layers, num_images))
    
    for layer_idx in range(num_layers):
        image_ks = layer_k_list[layer_idx]
        image_vs = layer_v_list[layer_idx]
        d_head_img = image_ks[0].shape[-1]
        
        # 定义投影层
        proj = torch.nn.Linear(d_head_img, d_head_text, device=device, dtype=dtype)
        
        for img_idx in range(num_images):
            # 获取单张图像的K并投影
            img_k = image_ks[img_idx].to(dtype)
            img_k_proj = proj(img_k)
            
            # 计算注意力分数
            attn_scores = torch.matmul(text_q_heads, img_k_proj.transpose(2, 3))
            attn_scores = attn_scores / torch.sqrt(torch.tensor(d_head_text, device=device, dtype=dtype))
            
            # 聚合分数
            patch_avg = torch.mean(attn_scores, dim=-1)
            token_avg = torch.mean(patch_avg, dim=-1)
            layer_img_score = torch.mean(token_avg).item()
            
            layer_scores[layer_idx, img_idx] = layer_img_score
    
    # 融合所有层的分数（取平均）
    final_scores = np.mean(layer_scores, axis=0)
    return final_scores.tolist()


def test_kv_cache_with_real_images(model, images, dtype, device, resolution=518):
    """提取图像KV缓存（自动适配图像数量）"""
    print("\n" + "="*80)
    print("🧪 Testing KV Cache Mechanism with Real Images")
    print("="*80)
    
    total_frames = len(images)
    print(f"\n📷 Processing {total_frames} real images...")
    
    # 第1帧：初始化缓存
    print("\n" + "-"*80)
    print(f"🔵 Frame 1: Initialize KV Cache (共 {total_frames} 帧)")
    print("-"*80)
    
    past_key_values = [None] * model.aggregator.depth
    print(f"Initialized past_key_values: {len(past_key_values)} layers, all None")
    
    # 处理第1帧
    frame1 = images[0:1]
    frame1 = F.interpolate(frame1, size=(resolution, resolution), mode="bilinear", align_corners=False).to(dtype)
    
    start_time = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            frame1_input = frame1[None]
            output_list, ps_idx, past_key_values = model.aggregator(
                frame1_input,
                past_key_values=past_key_values,
                use_cache=True,
                past_frame_idx=0
            )
    
    elapsed = time.time() - start_time
    print(f"✅ Frame 1 processed in {elapsed:.3f}s")

    # 后续帧：复用缓存
    for frame_idx in range(1, total_frames):
        print(f"🟢 Frame {frame_idx + 1}: Reuse KV Cache (共 {total_frames} 帧)")
        
        current_frame = images[frame_idx:frame_idx+1]
        current_frame = F.interpolate(
            current_frame, 
            size=(resolution, resolution), 
            mode="bilinear", 
            align_corners=False
        ).to(dtype)
        
        start_time = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                frame_input = current_frame[None]
                output_list, ps_idx, past_key_values = model.aggregator(
                    frame_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    past_frame_idx=frame_idx
                )
        
        elapsed = time.time() - start_time
        print(f"✅ Frame {frame_idx + 1} processed in {elapsed:.3f}s")
        print(f"   Output: {len(output_list)} layers, last layer shape: {output_list[-2].shape}")
        features = output_list[-2][0,:, ps_idx:] 
        print(f"   Extracted features from layer {model.aggregator.depth - 2} at ps_idx {ps_idx}")
        print(f"   Extracted features shape: {features.shape}, dtype: {features.dtype}")
    return output_list, past_key_values, features


def run_text_image_matching_pipeline(args, model, device, dtype):
    """完整的文本-图像匹配流程（自动适配图像数量）"""
    # 1. 从args获取路径（删除硬编码）
    scene_dir = args.scene_dir
    text_q_path = args.text_q_path
    
    # 检查图像目录是否存在
    if not os.path.exists(scene_dir):
        raise NotADirectoryError(f"图像目录不存在: {scene_dir}")
    
    # 获取所有图像路径
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_path_list = []
    for ext in image_extensions:
        image_path_list.extend(glob.glob(os.path.join(scene_dir, ext)))
    
    # 去重并排序
    image_path_list = sorted(list(set(image_path_list)))
    num_images = len(image_path_list)
    
    if num_images == 0:
        raise ValueError(f"在 {scene_dir} 中未找到任何图像文件")
    print(f"\n📂 找到 {num_images} 张图像:")
    for i, path in enumerate(image_path_list[:5]):
        print(f"   {i+1}. {os.path.basename(path)}")
    if num_images > 5:
        print(f"   ... 还有 {num_images - 5} 张图像")

    # 2. 预处理图像
    print(f"\n⏳ 加载并预处理 {num_images} 张图像...")
    images = load_and_preprocess_images(image_path_list).to(device)
    print(f"✅ 加载图像完成，形状: {images.shape}, 数据类型: {images.dtype}")

    # 3. 提取图像KV缓存
    print("\n" + "="*80)
    print(f"🔍 提取 {num_images} 张图像的KV特征")
    print("="*80)
    _, past_kv, _ = test_kv_cache_with_real_images(model, images, dtype, device)
    
    # 过滤空缓存层
    valid_past_kv = [(k, v) for k, v in past_kv if k is not None and v is not None]
    print(f"✅ 有效KV层数量: {len(valid_past_kv)}")
    if not valid_past_kv:
        raise ValueError("没有有效的KV缓存层，请检查模型输出")

    # 4. 按图像拆分KV
    print("\n" + "="*80)
    print(f"🔄 按图像拆分KV (共 {num_images} 张图像)")
    print("="*80)
    layer_k_list, layer_v_list = split_image_kv_per_image(valid_past_kv, num_images=num_images)
    if not layer_k_list:
        raise ValueError("KV拆分失败，没有生成有效的图像KV列表")

    # 5. 加载文本Q（传入路径参数）
    print("\n" + "="*80)
    print("📖 加载文本Q")
    print("="*80)
    text_q = load_text_q(text_q_path, device, dtype)

    # 6. 计算匹配度
    print("\n" + "="*80)
    print(f"🧮 计算文本与 {num_images} 张图像的匹配度")
    print("="*80)
    start_time = time.time()    
    matching_scores = compute_text_image_matching(
        text_q=text_q,
        layer_k_list=layer_k_list,
        layer_v_list=layer_v_list,
        num_heads=16,
        num_images=num_images,
        d_model=4096
    )
    elapsed = time.time() - start_time
    print(f"✅ 计算匹配度耗时: {elapsed:.3f}s")

    # 7. 输出结果
    print("\n" + "="*80)
    print("🏁 文本-图像匹配结果")
    print("="*80)
    for i, score in enumerate(matching_scores):
        print(f"图像 {i+1} ({os.path.basename(image_path_list[i])}) 与文本的匹配度: {score:.6f}")
    
    # 排序和最匹配图像
    sorted_indices = sorted(range(len(matching_scores)), key=lambda i: matching_scores[i], reverse=True)
    print(f"匹配度排序（从高到低）: {[i+1 for i in sorted_indices]}")
    
    max_score_idx = sorted_indices[0]
    print(f"\n最匹配的图像是: 图像 {max_score_idx + 1} ({os.path.basename(image_path_list[max_score_idx])})")

    # 详细排序输出
    print("\n匹配度从高到低排序:")
    for idx in sorted_indices:
        print(f"图像 {idx + 1} ({os.path.basename(image_path_list[idx])}): {matching_scores[idx]:.6f}")

    return matching_scores

def get_current_image_feature(args, model, device, dtype):
    # 1. 从args获取路径（删除硬编码）
    scene_dir = args.scene_dir
    # 检查图像目录是否存在
    if not os.path.exists(scene_dir):
        raise NotADirectoryError(f"图像目录不存在: {scene_dir}")
    
    # 获取所有图像路径
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_path_list = []
    for ext in image_extensions:
        image_path_list.extend(glob.glob(os.path.join(scene_dir, ext)))
    
    # 去重并排序
    image_path_list = sorted(list(set(image_path_list)))
    num_images = len(image_path_list)
    
    if num_images == 0:
        raise ValueError(f"在 {scene_dir} 中未找到任何图像文件")
    print(f"\n📂 找到 {num_images} 张图像:")
    for i, path in enumerate(image_path_list[:5]):
        print(f"   {i+1}. {os.path.basename(path)}")
    if num_images > 5:
        print(f"   ... 还有 {num_images - 5} 张图像")

    # 2. 预处理图像
    print(f"\n⏳ 加载并预处理 {num_images} 张图像...")
    images = load_and_preprocess_images(image_path_list).to(device)
    print(f"✅ 加载图像完成，形状: {images.shape}, 数据类型: {images.dtype}")

    # 3. 提取图像KV缓存
    print("\n" + "="*80)
    print(f"🔍 提取 {num_images} 张图像的KV特征")
    print("="*80)
    output_list, past_kv, features = test_kv_cache_with_real_images(model, images, dtype, device)
    print(f"   Extracted features shape: {features.shape}, dtype: {features.dtype}")

    return features    

def demo_fn(args):
    print("Arguments:", vars(args))

    # 随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"Setting seed as: {args.seed}")

    # 设备和数据类型
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # # 加载模型
    model = VGGT()
    model.load_state_dict(torch.load("/home/pcl/vggt/weight/model.pt", map_location=device))
    model = model.to(device).to(dtype)
    model.eval()
    print(f"Model loaded, 数据类型: {dtype}")
    print(f"PyTorch 总占用（含缓存）: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # 运行流程
    run_text_image_matching_pipeline(args, model, device, dtype)
    print(f"PyTorch 总占用（含缓存）: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # 清理显存
    torch.cuda.empty_cache()
    print("\n" + "="*80)
    print("✅ 所有流程完成!")
    print("="*80)

    return True

def demo_fn_new(args):
    print("Arguments:", vars(args))

    # 随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"Setting seed as: {args.seed}")

    # 设备和数据类型
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # # 加载模型
    model = VGGT()
    model.load_state_dict(torch.load("/home/pcl/vggt/weight/model.pt", map_location=device))
    model = model.to(device).to(dtype)
    model.eval()
    print(f"Model loaded, 数据类型: {dtype}")
    print(f"PyTorch 总占用（含缓存）: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # 获取当前图像特征
    feature = get_current_image_feature(args, model, device, dtype)
    print("当前图像特征 shape:", feature.shape)
    print(f"PyTorch 总占用（含缓存）: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # 清理显存
    torch.cuda.empty_cache()
    print("\n" + "="*80)
    print("✅ 所有流程完成!")
    print("="*80)

    # return True
    return feature
# ===============================================================
# Main Entry
# ===============================================================
if __name__ == "__main__":
    # 新增
    args = parse_args()
    with torch.no_grad():
        vggt_feature = demo_fn_new(args)
        print("提取的当前图像特征:", vggt_feature.shape)



    print("原始 VGG-T 特征:", vggt_feature.shape)  # [1, 1369, 2048]

    # 初始化 VGGTMerger
    merger = VGGTv2.VGGTMerger(
        output_dim=4096,
        hidden_dim=4096,
        context_dim=2048,
        spatial_merge_size=2
    ).to(vggt_feature.device)

    # 获取 image_embeds_3d
    image_embeds_3d = merger(vggt_feature)

    print("最终 image_embeds_3d:", image_embeds_3d.shape)  # [1, 361, 4096]


    # select modality
    pose_goal = False
    satellite = False
    image_goal = False
    lan_prompt = True

    # Goal definitions
    # lan_inst_prompt = "move toward blue trash bin"
    # lan_inst_prompt = "turn right and go straight"
    # lan_inst_prompt = "move toward black tv monitor"
    lan_inst_prompt = "move toward white office cabinet"
    # lan_inst_prompt = "turn right and move forward"
    # lan_inst_prompt = "turn right"
    goal_lat, goal_lon, goal_compass = 37.8738930785863, -122.26746181032362, 0.0
    goal_utm = utm.from_latlon(goal_lat, goal_lon)
    goal_compass = -float(goal_compass) / 180.0 * math.pi
    goal_image_PIL = Image.open("./inference/goal_img.jpg").convert("RGB")

    # Define models (VLA, action_head, pose_projector, processor, etc.)
    cfg = InferenceConfig()
    vla, action_head, pose_projector, device_id, NUM_PATCHES, action_tokenizer, processor = define_model(cfg)

    # Run inference
    inference = Inference(
        save_dir="./inference",
        lan_inst_prompt=lan_inst_prompt,
        goal_utm=goal_utm,
        goal_compass=goal_compass,
        goal_image_PIL=goal_image_PIL,
        action_tokenizer=action_tokenizer,
        processor=processor,
    )
    inference.vggt_feature = image_embeds_3d.to(device_id)
    inference.run()
    print(f"PyTorch 总占用（含缓存）: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    # 睡眠5秒以确保所有输出完成
    # time.sleep(5)


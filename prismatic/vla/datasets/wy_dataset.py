import os
import random
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset
from typing import Tuple, Type, Dict
import math
import utm
import torchvision.transforms.functional as TF

# OpenVLA/Prismatic 依赖
from prismatic.vla.constants import IGNORE_INDEX
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.vision import ImageTransform

class WY_Dataset(Dataset):
    def __init__(
        self,
        action_tokenizer: PreTrainedTokenizerBase,
        base_tokenizer: ActionTokenizer,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        data_root_dir: str = "data/data_splits/navitrace_dataset",
        data_split_type: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        len_traj_pred: int = 8,
        predict_stop_token: bool = True,
        use_embodiment_prompt: bool = True,
        dataset_name="navitrace",
        len_context: int = 8,
    ):
        self.context_size = 5
        self.data_root_dir = data_root_dir
        self.data_split_type = data_split_type
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.prompt_builder = prompt_builder_fn
        self.predict_stop_token = predict_stop_token
        self.image_transform = image_transform
        self.len_traj_pred = len_traj_pred  # 保存轨迹长度用于生成 mask
        # 1. 定位 .arrow 文件
        split_dir = os.path.join(self.data_root_dir, self.data_split_type)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Directory not found: {split_dir}")
        
        arrow_files = glob.glob(os.path.join(split_dir, "*.arrow"))
        if not arrow_files:
            raise FileNotFoundError(f"No .arrow files found in {split_dir}")
            
        print(f"Loading raw arrow files from {split_dir}...")
        
        # 2. 加载数据集
        self.dataset = load_dataset("arrow", data_files=arrow_files, split="train")

        # ------------------------------------------------------------------
        # 3. 核心修改：构建索引映射并过滤无效数据 (Filter Invalid Data)
        # ------------------------------------------------------------------
        self.index_map = []
        
        # 同时加载 embodiments 和 ground_truth 列进行检查
        all_embodiments = self.dataset['embodiments']
        all_ground_truths = self.dataset['ground_truth']
        
        print("Building index map and filtering invalid trajectories...")
        skipped_count = 0

        # 使用 zip 同时遍历，避免在循环中反复查询 dataset
        for row_idx, (embs_list, gt_dict) in enumerate(zip(all_embodiments, all_ground_truths)):
            if not embs_list:
                continue
                
            for emb_name in embs_list:
                # --- 关键修复开始 ---
                # 检查 ground_truth 中是否有该机器人的数据，且数据不为空
                is_valid = False
                if gt_dict and emb_name in gt_dict:
                    traj = gt_dict[emb_name]
                    # 确保轨迹列表存在且长度大于0
                    if traj is not None and len(traj) > 0:
                        is_valid = True
                
                if is_valid:
                    self.index_map.append((row_idx, emb_name))
                else:
                    skipped_count += 1
                # --- 关键修复结束 ---

        print(f"Original samples: {len(self.dataset)}")
        print(f"Skipped invalid/empty trajectories: {skipped_count}")
        print(f"Final Expanded samples: {len(self.index_map)} (Valid trajectories only)")
    
    def calculate_relative_position(self, x_a, y_a, x_b, y_b):
        return x_b - x_a, y_b - y_a

    def rotate_to_local_frame(self, delta_x, delta_y, heading_a_rad):
        rel_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        rel_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return rel_x, rel_y

    def _resize_norm(self, image, size):
        return TF.resize(image, size)
        
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict:

            thres_dist = 30.0
            metric_waypoint_spacing = 0.1 
            predict_stop_token=True

            original_row_idx, target_embodiment = self.index_map[idx]
            sample = self.dataset[original_row_idx]
            
            # 原始数据
            task = sample['task']
            image = sample['image']
            current_image_PIL = image
            goal_image_PIL = image
            ground_truth = sample['ground_truth']
            # metadata = sample['metadata']

            # 3. 生成针对【当前机器人】的任务描述
            embod_task = f"Generate the trajectory for {target_embodiment} to {task}."

            # 4. 归一化轨迹数据
            W, H = image.size
            
            # 初始化为空，但在 __init__ 过滤后，这里理论上一定会有数据
            normalized_traj = np.zeros((1, 4), dtype=np.float64) 

            # 直接获取数据，因为我们在 __init__ 里保证了它存在且不为空
            if target_embodiment in ground_truth:
                raw_trajs_list = ground_truth[target_embodiment]
                if len(raw_trajs_list) > 0:
                    raw_traj = np.array(raw_trajs_list[0]) # 取第一条 (原始数据)
                    
                    # 复制原始数据用于归一化，确保长度是原始长度 (e.g., 9 步)
                    if raw_traj.ndim >= 2 and raw_traj.shape[-1] >= 2:
                        # 归一化 x/W, y/H
                        traj_norm_orig = np.zeros_like(raw_traj, dtype=np.float64)
                        traj_norm_orig[:, 0] = raw_traj[:, 0] / W
                        traj_norm_orig[:, 1] = raw_traj[:, 1] / H
                        # 补零变成 4 维 (POSE_DIM)
                        original_normalized_traj = np.pad(traj_norm_orig, ((0, 0), (0, 2)), mode='constant', constant_values=0.0)
                    else:
                        # 如果维度不对，创建一个占位符
                        original_normalized_traj = np.zeros((len(raw_traj) if raw_traj.ndim >= 2 else 1, 4), dtype=np.float64)
                    # === [新增逻辑] 规定 original_normalized_traj 为 20 个点 ===
                    MAX_TRAJECTORY_LENGTH = 20
                    current_length = original_normalized_traj.shape[0]

                    if current_length > MAX_TRAJECTORY_LENGTH:
                        # 截断
                        original_normalized_traj = original_normalized_traj[:MAX_TRAJECTORY_LENGTH, :]

                    elif current_length < MAX_TRAJECTORY_LENGTH:
                        # 补零
                        num_pad = MAX_TRAJECTORY_LENGTH - current_length
                        # 创建一个用 0 填充的 (num_pad, 4) 矩阵
                        padding = np.zeros((num_pad, 4), dtype=original_normalized_traj.dtype)
                        original_normalized_traj = np.vstack([original_normalized_traj, padding])

                    # === [新增逻辑结束] ===
                    # 将路径长度固定为8步
                    if len(raw_traj) < self.len_traj_pred:
                        num_pad = self.len_traj_pred - len(raw_traj)
                        last_point = raw_traj[-1]
                        padding = np.tile(last_point, (num_pad, 1))
                        raw_traj = np.vstack([raw_traj, padding])
                    elif len(raw_traj) > self.len_traj_pred:
                        raw_traj = raw_traj[:self.len_traj_pred]
                        
                    # 计算 8 步的 normalized_traj (用于 actions)
                    if raw_traj.ndim >= 2 and raw_traj.shape[-1] >= 2:
                        # 归一化 x/W, y/H
                        traj_norm = np.zeros_like(raw_traj, dtype=np.float64)
                        traj_norm[:, 0] = raw_traj[:, 0] / W
                        traj_norm[:, 1] = raw_traj[:, 1] / H 
                        # 补零变成4维
                        normalized_traj = np.pad(traj_norm, ((0, 0), (0, 2)), mode='constant', constant_values=0.0)
                    else:
                        raise ValueError(f"Unexpected empty trajectory after cropping for {target_embodiment} at index {original_row_idx}")
                
                else:
                    # 如果代码跑到这里，说明 __init__ 过滤逻辑有漏网之鱼，抛出异常方便调试
                    raise ValueError(f"Unexpected empty trajectory list for {target_embodiment} at index {original_row_idx}")
            
            modality_id = 7
            inst_obj = embod_task
            actions = normalized_traj

            # 虚拟导航目标逻辑 (保持不变)
            current_lat, current_lon, current_compass = 37.87371258374039, -122.26729417226024, 270.0
            cur_utm = utm.from_latlon(current_lat, current_lon)
            cur_compass = -float(current_compass) / 180.0 * math.pi
            
            goal_lat, goal_lon, goal_compass = 37.8738930785863, -122.26746181032362, 0.0
            goal_utm = utm.from_latlon(goal_lat, goal_lon)
            goal_compass = -float(goal_compass) / 180.0 * math.pi
            
            delta_x, delta_y = self.calculate_relative_position(
                cur_utm[0], cur_utm[1], goal_utm[0], goal_utm[1]
            )
            relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, cur_compass)
            radius = np.sqrt(relative_x**2 + relative_y**2)
            if radius > thres_dist:
                relative_x *= thres_dist / radius
                relative_y *= thres_dist / radius
            goal_pose_loc_norm = np.array([
                relative_y / metric_waypoint_spacing,
                -relative_x / metric_waypoint_spacing,
                np.cos(goal_compass - cur_compass),
                np.sin(goal_compass - cur_compass)
            ])        

            goal_pose_cos_sin = goal_pose_loc_norm 

            # 构建对话 Prompt
            IGNORE_INDEX = -100
            
            # 二次安全检查
            if len(actions) == 0:
                # 如果这里报错，说明上面的归一化逻辑有问题
                raise ValueError(f"Computed actions are empty for {target_embodiment} at index {original_row_idx}")

            current_action = actions[0]
            future_actions = actions[1:]
            future_actions_string = ''.join(self.action_tokenizer(future_actions))
            current_action_string = self.action_tokenizer(current_action)
            action_chunk_string = current_action_string + future_actions_string
            action_chunk_len = len(action_chunk_string)

            conversation = [
                {"from": "human", "value": f"{inst_obj}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]

            prompt_builder = self.prompt_builder("openvla")
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            # Tokenize
            input_ids = torch.tensor(self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
            labels = input_ids.clone()
            labels[:-(action_chunk_len + 1)] = IGNORE_INDEX
            if not predict_stop_token:
                labels[-1] = IGNORE_INDEX

            # Images for MBRA model
            image_obs_list = []
            for ih in range(self.context_size + 1):
                image_obs_list.append(self._resize_norm(TF.to_tensor(current_image_PIL), (96, 96)))  #In our real code, image_obs_list is list of history image. In this dummy dataset code, we feed current images. The detail implementation is same as ViNT, NoMaD code base. 
            image_obs = torch.cat(image_obs_list)     
            image_goal = self._resize_norm(TF.to_tensor(goal_image_PIL), (96, 96))

            # Data augmentation (random cropping)
            voffset = int(224.0*0.2*random.random())
            hoffset = int(224.0*0.1*random.random())            
            PILbox = (hoffset, voffset, 224-hoffset, 224-voffset)
            current_image_PIL = current_image_PIL.crop(PILbox).resize((224,224)) 
            goal_image_PIL = goal_image_PIL.crop(PILbox).resize((224,224))      

            # Data augmentation (horizontal flipping)
            if random.random() > 0.5:
                current_image_PIL = current_image_PIL.transpose(Image.FLIP_LEFT_RIGHT)
                goal_image_PIL = goal_image_PIL.transpose(Image.FLIP_LEFT_RIGHT)
                actions[:,1] = -actions[:,1]
                actions[:,3] = -actions[:,3]
                goal_pose_cos_sin[1] = -goal_pose_cos_sin[1]
                goal_pose_cos_sin[3] = -goal_pose_cos_sin[3]  
                
                image_obs = torch.flip(image_obs, dims=[2])
                image_goal = torch.flip(image_goal, dims=[2])  
                            
            pixel_values_current = self.image_transform(current_image_PIL)
            pixel_values_goal = self.image_transform(goal_image_PIL) 
            # pixel_values_current = self.image_transform(image)
            # pixel_values_goal = self.image_transform(image)

            #action select 1.0: raw action, 0.0: MBRA synthetic action
            action_select_mask = torch.tensor(1.0)


            return {
                "sample_id": sample['sample_id'],
                "task": sample['task'],
                "embodiments": sample['embodiments'],
                "image": sample['image'],
                "segmentation_mask": sample['segmentation_mask'],
                "ground_truth": sample['ground_truth'],
                "category": sample['category'],
                "context": sample['context'],
                "metadata": sample['metadata'],
                "embod_task": embod_task,
                "normalized_trajectory": normalized_traj,
                "original_normalized_trajectory": torch.as_tensor(original_normalized_traj),

                "pixel_values": pixel_values_current, # 建议加上这个键，很多模型默认读取这个
                "pixel_values_goal": pixel_values_goal,
                "input_ids": input_ids,
                "labels": labels,
                "dataset_name": "wy_dataset",
                "modality_id": modality_id,
                "actions": torch.as_tensor(actions),
                "action_select_mask": action_select_mask, # 修复本次报错的关键
                "goal_pose": goal_pose_cos_sin, # 修复 Proprio 缺失
                "obj_pose_norm": goal_pose_cos_sin[0:2],  # 修复 Loss 计算缺失
                "img_PIL": current_image_PIL, 
                "gimg_PIL": goal_image_PIL,
                "cur_image":image_obs,
                "goal_image_8":image_goal,
                "temp_dist": 10.0,             # 修复 Loss 计算缺失
                "lan_prompt": inst_obj
            }

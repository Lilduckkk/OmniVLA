import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import io
import random
import torchvision.transforms.functional as TF
from typing import Tuple, Dict, Type, List, Optional
import re
import math
import cv2

# 假设这些类已经定义，保留类型提示
# from transformers import PreTrainedTokenizerBase
# class ActionTokenizer: pass
# class ImageTransform: pass
# class PromptBuilder: pass

class Vamos_Dataset(Dataset):
    def __init__(
        self,
        action_tokenizer: any, # 替换为实际类型
        base_tokenizer: any,   # 替换为实际类型
        image_transform: any,  # 替换为实际类型
        prompt_builder_fn: Type[any],
        data_root_dir: str = "data/data_splits/vamos_dataset",
        data_split_type: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        len_traj_pred: int = 8,
        predict_stop_token: bool = True,
        use_embodiment_prompt: bool = True,
        dataset_name="vamos",
        len_context: int = 8,
        filter_modality_id: int = 7,  # 新增：指定要筛选的modality_id
    ):
        # --- 这一部分严格保留你提供的代码 ---
        self.context_size = 5
        self.data_root_dir = data_root_dir
        self.data_split_type = data_split_type
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.prompt_builder = prompt_builder_fn
        self.predict_stop_token = predict_stop_token
        self.image_transform = image_transform
        self.len_traj_pred = len_traj_pred  # 保存轨迹长度用于生成 mask
        self.MAX_TRAJECTORY_LENGTH = 200 # 统一轨迹长度为20
        self.filter_modality_id = filter_modality_id  # 新增：保存筛选条件
        # --- 以下是追加的数据加载逻辑 ---
        
        # 拼接路径: data_root_dir/train/*.parquet
        search_path = os.path.join(self.data_root_dir, self.data_split_type)
        self.file_paths = sorted(glob.glob(os.path.join(search_path, "*.parquet")))
        
        if not self.file_paths:
            # 兼容：如果没有子文件夹，尝试直接在 root 下找
            search_path = self.data_root_dir
            self.file_paths = sorted(glob.glob(os.path.join(search_path, "*.parquet")))
            
        self.data_frames = []
        self.sample_map = [] 

        # print(f"Loading Vamos Dataset from {search_path}...")
        # for df_idx, path in enumerate(self.file_paths):
        #     try:
        #         # 读取 parquet
        #         df = pd.read_parquet(path, engine='auto')
        #         self.data_frames.append(df)
        #         for row_idx in range(len(df)):
        #             self.sample_map.append((df_idx, row_idx))
        #     except Exception as e:
        #         print(f"Error loading {path}: {e}")
        
        # print(f"Total samples: {len(self.sample_map)}")

        # 可以筛选modality_id的样本
        print(f"Loading Vamos Dataset from {search_path}...")
        print(f"Will filter samples with modality_id = {self.filter_modality_id}")  # 新增：提示筛选条件
        for df_idx, path in enumerate(self.file_paths):
            try:
                # 读取 parquet
                df = pd.read_parquet(path, engine='auto')
                self.data_frames.append(df)
                # 遍历每行数据，先过滤再加入sample_map
                for row_idx in range(len(df)):
                    # 新增：提前获取task并计算modality_id，只保留符合条件的样本
                    sample_text = df.iloc[row_idx]['text']
                    _, modality_id = self._process_task(sample_text)
                    # 只保留modality_id等于目标值的样本
                    if modality_id == self.filter_modality_id:
                        self.sample_map.append((df_idx, row_idx))
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        print(f"Total filtered samples (modality_id={self.filter_modality_id}): {len(self.sample_map)}")  # 新增：打印过滤后的样本数
    
    def __len__(self) -> int:
        return len(self.sample_map)

    def _decode_image(self, img_data):
        """解码 Parquet 中的图像 bytes 或 dict"""
        try:
            if isinstance(img_data, dict) and 'bytes' in img_data:
                return Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
            elif isinstance(img_data, bytes):
                return Image.open(io.BytesIO(img_data)).convert("RGB")
            # 保护性返回黑图
            return Image.new('RGB', (224, 224), (0, 0, 0))
        except Exception as e:
            print(f"Image decode error: {e}")
            return Image.new('RGB', (224, 224), (0, 0, 0))

    def _process_task(self, sample_text):
        # 定义正则表达式来匹配 "Navigate to x=<loc>, y=<loc>"
        location_pattern = r"Navigate to x=<[^>]+>, y=<[^>]+>"
        
        # 查找任务中的坐标部分
        location_match = re.match(location_pattern, sample_text)
        
        if location_match:
            # 如果包含坐标部分
            task_location = location_match.group(0)  # 获取坐标部分
            task_language = sample_text[len(task_location):].strip()  # 获取语言部分并去除前后空格
            
            if task_language:  # 如果有语言指令部分
                modality_id = 7
                return task_language.lstrip(".").strip(), modality_id
            else:
                modality_id = 4
                return "xxxx", modality_id  # 如果没有语言部分
        else:
            return "Invalid task format", None  # 如果格式不匹配

    def calculate_relative_position(self, x_a, y_a, x_b, y_b):
        return x_b - x_a, y_b - y_a

    def rotate_to_local_frame(self, delta_x, delta_y, heading_a_rad):
        rel_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        rel_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return rel_x, rel_y

    def _resize_norm(self, image, size):
        return TF.resize(image, size)
        
    def __getitem__(self, i: int) -> Dict:
        df_idx, row_idx = self.sample_map[i]
        sample = self.data_frames[df_idx].iloc[row_idx]
        # 查看所有的字段
        # print(f"Row fields: {list(sample.index)}")
        # task = sample['text']
        task, modality_id = self._process_task(sample['text'])
        # print(f"task: {task}")
        # print(f"modality_id: {modality_id}")
        image_data = sample['image']
        image = self._decode_image(image_data)
        current_image_PIL = image
        goal_image_PIL = image
        ground_truth = sample['shorter_trajectory_2d']

        inst_obj = task
        embod_task = task

        # print(f"len of ground_truth: {len(ground_truth)}")
        # print(f"ground_truth: {ground_truth}")

        # 4. 归一化轨迹数据
        W, H = image.size
        # 初始化为空，但在 __init__ 过滤后，这里理论上一定会有数据
        normalized_traj = np.zeros((1, 4), dtype=np.float64) 
        original_normalized_traj_20 = np.zeros((self.MAX_TRAJECTORY_LENGTH, 4), dtype=np.float64)

        # 直接获取数据，因为我们在 __init__ 里保证了它存在且不为空
        raw_trajs_list = ground_truth
        if len(raw_trajs_list) > 0:
            try:
                # 尝试直接堆叠，如果是 list of arrays 或 object array
                raw_traj = np.stack(raw_trajs_list).astype(np.float64)
            except ValueError:
                # 如果 stack 失败（如形状不统一），强制转 list 再转 numpy
                raw_traj = np.array(list(raw_trajs_list), dtype=np.float64)
            
            # 确保维度是 (N, 2)
            if raw_traj.ndim == 1:
                raw_traj = raw_traj.reshape(-1, 2)
            
            # --- 逻辑分支 A: 处理 20 步长的 original_normalized_traj ---
            # 归一化
            traj_norm_orig = np.zeros_like(raw_traj, dtype=np.float64)
            traj_norm_orig[:, 0] = raw_traj[:, 0] / W
            traj_norm_orig[:, 1] = raw_traj[:, 1] / H
            # 补零变成 4 维 (x, y, 0, 0)
            original_normalized_traj = np.pad(traj_norm_orig, ((0, 0), (0, 2)), mode='constant')

            # 截断或填充到 20 步
            current_length = original_normalized_traj.shape[0]
            if current_length > self.MAX_TRAJECTORY_LENGTH:
                original_normalized_traj_20 = original_normalized_traj[:self.MAX_TRAJECTORY_LENGTH, :]
            elif current_length < self.MAX_TRAJECTORY_LENGTH:
                num_pad = self.MAX_TRAJECTORY_LENGTH - current_length
                last_point = original_normalized_traj[-1]
                padding = np.tile(last_point, (num_pad, 1))
                original_normalized_traj_20 = np.vstack([original_normalized_traj, padding])
            else:
                original_normalized_traj_20 = original_normalized_traj

            # --- 逻辑分支 B: 处理 8 步长的 raw_traj (用于 Action Tokenizer) ---
            # 为了不影响上面的 raw_traj，这里拷贝一份处理
            traj_for_action = raw_traj.copy()
            
            if len(traj_for_action) < self.len_traj_pred:
                num_pad = self.len_traj_pred - len(traj_for_action)
                last_point = traj_for_action[-1]
                padding = np.tile(last_point, (num_pad, 1))
                traj_for_action = np.vstack([traj_for_action, padding])
            elif len(traj_for_action) > self.len_traj_pred:
                traj_for_action = traj_for_action[:self.len_traj_pred]
            
            # 归一化并扩展到 4维
            traj_norm = np.zeros_like(traj_for_action, dtype=np.float64)
            traj_norm[:, 0] = traj_for_action[:, 0] / W
            traj_norm[:, 1] = traj_for_action[:, 1] / H
            normalized_traj = np.pad(traj_norm, ((0, 0), (0, 2)), mode='constant')

        else:
            # 数据为空时的处理（虽然 init 过滤了，但为了安全）
            print(f"Warning: Empty trajectory at {row_idx}")
            raw_traj = np.zeros((1, 2)) # 避免下面报错

        actions = normalized_traj
        # print(f"len of actions: {len(actions)}")
        # print(f"actions: {actions}")

        current_x, current_y, current_compass = actions[0][0], actions[0][1], 0.0
        # print(f" current_x:{current_x} , current_y:{current_y} , current_compass:{current_compass} ")
        # 逆时针为正
        goal_compass = np.arctan2(
            original_normalized_traj[-1, 1] - original_normalized_traj[-2, 1],  # Delta Y (垂直位移)
            original_normalized_traj[-1, 0] - original_normalized_traj[-2, 0]   # Delta X (水平位移)
        )
        goal_x, goal_y, goal_compass = original_normalized_traj[-1,0], original_normalized_traj[-1,1], goal_compass
        # print(f" goal_x:{goal_x} , goal_y:{goal_y} , goal_compass:{goal_compass} ")
        delta_x, delta_y = self.calculate_relative_position(
            current_x, current_y, goal_x, goal_y
        ) 
        relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, current_compass)    
        # print(" relative_x:", relative_x, " relative_y:", relative_y)   
        goal_pose_loc_norm = np.array([
            relative_x ,
            relative_y,
            np.cos(goal_compass - current_compass),
            np.sin(goal_compass - current_compass)
        ])             
        goal_pose_cos_sin = goal_pose_loc_norm   

        # 构建对话 Prompt
        IGNORE_INDEX = -100

        # 二次安全检查
        if len(actions) == 0:
            # 如果这里报错，说明上面的归一化逻辑有问题
            raise ValueError(f"Computed actions are empty for at index {row_idx}")

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
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX 
        
        # Images for MBRA model
        image_obs_list = []
        for ih in range(self.context_size + 1):
            image_obs_list.append(self._resize_norm(TF.to_tensor(current_image_PIL), (96, 96)))  #In our real code, image_obs_list is list of history image. In this dummy dataset code, we feed current images. The detail implementation is same as ViNT, NoMaD code base. 
        image_obs = torch.cat(image_obs_list)     
        image_goal = self._resize_norm(TF.to_tensor(goal_image_PIL), (96, 96))

        # 可以不修改形状，保留原始的形状方便可视化
        current_image_PIL = current_image_PIL.resize((1000, 1000))  # 直接修改为224×224
        goal_image_PIL = goal_image_PIL.resize((1000, 1000))        # 目标图同步处理 

        pixel_values_current = self.image_transform(current_image_PIL)
        pixel_values_goal = self.image_transform(goal_image_PIL)

        action_select_mask = torch.tensor(1.0)
        # 随机生成 segmentation_mask 作为示例
        segmentation_mask = np.random.randint(0, 2, (1000, 1000))  # 二值掩码示例
        segmentation_mask = np.array(segmentation_mask, dtype=np.uint8)
        target_size = (1000, 1000)
        resized_mask_cv2 = cv2.resize(
            segmentation_mask,
            target_size,  # 保持与图像一致的大小
            interpolation=cv2.INTER_NEAREST
        )


        # return dict(
        #     task=sample['lang_goal'],

        # )
        return {
            # "sample_id": sample['sample_id'],
            # "task": sample['task'],
            # "embodiments": sample['embodiments'],
            # "image": sample['image'],
            "segmentation_mask": torch.as_tensor(resized_mask_cv2),
            # "ground_truth": sample['ground_truth'],
            # "category": sample['category'],
            # "context": sample['context'],
            # "metadata": sample['metadata'],
            "embod_task": embod_task,
            "normalized_trajectory": normalized_traj,
            "original_normalized_trajectory": torch.as_tensor(original_normalized_traj_20),

            "pixel_values": pixel_values_current, # 建议加上这个键，很多模型默认读取这个
            "pixel_values_goal": pixel_values_goal,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": "navitrace_dataset",
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
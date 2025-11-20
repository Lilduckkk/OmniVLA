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

# ===============================================================  # 分隔线
# Inference Class  # 推理类
# ===============================================================  # 分隔线
class Inference:  # 定义推理流程封装类
    def __init__(self, save_dir, lan_inst_prompt, goal_utm, goal_compass, goal_image_PIL, action_tokenizer, processor):  # 构造函数
        self.tick_rate = 3  # 控制循环频率（Hz）
        self.lan_inst_prompt = lan_inst_prompt  # 语言指令
        self.goal_utm = goal_utm  # 目标 UTM 坐标
        self.goal_compass = goal_compass  # 目标朝向（弧度）
        self.goal_image_PIL = goal_image_PIL  # 目标图像 PIL
        self.action_tokenizer = action_tokenizer  # 动作 tokenizer
        self.processor = processor  # 处理器（图像+文本）
        self.count_id = 0  # 计数器（用于保存）
        self.linear, self.angular = 0.0, 0.0  # 线速度与角速度初始化
        self.datastore_path_image = save_dir  # 可视化保存目录
    # ----------------------------  # 分隔线
    # Static Utility Methods  # 静态工具方法
    # ----------------------------  # 分隔线
    @staticmethod
    def calculate_relative_position(x_a, y_a, x_b, y_b):  # 计算 B 相对 A 的平面位置差
        return x_b - x_a, y_b - y_a  # 返回 Δx, Δy

    @staticmethod
    def rotate_to_local_frame(delta_x, delta_y, heading_a_rad):  # 旋转到 A 的车体坐标系
        rel_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)  # 旋转后的 x
        rel_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)  # 旋转后的 y
        return rel_x, rel_y  # 返回相对坐标

    # ----------------------------  # 分隔线
    # Main Loop  # 主循环
    # ----------------------------  # 分隔线
    def run(self):  # 运行一次 tick（默认只跑一次以示例）
        loop_time = 1 / self.tick_rate  # 循环周期
        start_time = time.time()  # 起始时间
        while True:  # 循环
            if time.time() - start_time > loop_time:  # 到达一个周期
                self.tick()  # 执行一次 tick
                start_time = time.time()  # 重置时间
                break  # 示例中只运行一次，实际应用可移除此行

    def tick(self):  # 单步更新
        self.linear, self.angular = self.run_omnivla()  # 调用主推理，得到线/角速度

    # ----------------------------  # 分隔线
    # OmniVLA Inference  # OmniVLA 推理
    # ----------------------------  # 分隔线
    def run_omnivla(self):  # 推理主流程
        thres_dist = 30.0  # 最大截断半径（米）
        metric_waypoint_spacing = 0.1  # 轨迹点单位间隔（米/单位）

        # Load current GPS & heading  # 加载当前 GPS 与朝向（示例）
        current_lat = 37.87371258374039  # 当前纬度（示例）
        current_lon = -122.26729417226024  # 当前经度（示例）
        current_compass = 270.0  # 当前罗盘角（度，示例）
        cur_utm = utm.from_latlon(current_lat, current_lon)  # 转换为 UTM  坐标cur_utm[0]：东向坐标（x 轴），单位米；cur_utm[1]：北向坐标（y 轴），单位米。
        cur_compass = -float(current_compass) / 180.0 * math.pi  # 转为弧度并取反（坐标系约定） 

        # Local goal position  # 计算目标在车体坐标系下的位置
        delta_x, delta_y = self.calculate_relative_position(
            cur_utm[0], cur_utm[1], self.goal_utm[0], self.goal_utm[1]
        )  # 世界坐标下 Δx, Δy
        relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, cur_compass)  # 旋转到车体坐标系
        radius = np.sqrt(relative_x**2 + relative_y**2)  # 与目标的距离
        if radius > thres_dist:  # 距离超过阈值则截断
            relative_x *= thres_dist / radius  # x 缩放
            relative_y *= thres_dist / radius  # y 缩放

        goal_pose_loc_norm = np.array([
            relative_y / metric_waypoint_spacing,  # 按间距归一化：前进方向为 x（此处坐标系定义：x 前进，y 左） 无量纲了 就是代表多少个间距单位 参考1_ex.jpg
            -relative_x / metric_waypoint_spacing,  # 横向偏移（符号按坐标系约定取反）
            np.cos(self.goal_compass - cur_compass),  # 目标相对朝向的 cos
            np.sin(self.goal_compass - cur_compass)  # 目标相对朝向的 sin
        ])  # 作为本体输入（目标位姿）

        # Load current image  # 加载当前相机图像
        current_image_path = "./inference/current_img.jpg"  # 当前图像路径
        current_image_PIL = Image.open(current_image_path).convert("RGB")  # 打开并转为 RGB

        # Language instruction  # 语言指令选择
        lan_inst = self.lan_inst_prompt if lan_prompt else "xxxx"  # 若启用语言，则用传入指令，否则用占位 "xxxx"

        # Prepare batch  # 组装 batch 数据
        batch = self.data_transformer_omnivla(
            current_image_PIL, lan_inst, self.goal_image_PIL, goal_pose_loc_norm,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor
        )  # 通过数据转换器构建模型输入
        # print(f"Pixel values shape after processor: {batch['pixel_values'].shape}")  # 打印图像张量形状

        # Run forward pass  # 前向推理
        actions, modality_id = self.run_forward_pass(
            vla=vla.eval(),  # VLA 切 eval 模式
            action_head=action_head.eval(),  # 动作头 eval
            noisy_action_projector=None,  # 未使用扩散时置空
            pose_projector=pose_projector.eval(),  # 位姿投影器 eval
            batch=batch,  # 输入 batch
            action_tokenizer=self.action_tokenizer,  # 动作 tokenizer
            device_id=device_id,  # 设备
            use_l1_regression=True,  # 使用 L1 回归头
            use_diffusion=False,  # 不使用扩散
            use_film=False,  # 不使用 FiLM
            num_patches=NUM_PATCHES,  # 视觉 patch 数量
            compute_diffusion_l1=False,  # 不计算扩散 L1
            num_diffusion_steps_train=None,  # 扩散训练步（未用）
            mode="train",  # 模式名（此处用于分支选择）
            idrun=self.count_id,  # 运行 id
        )
        self.count_id += 1  # 计数自增

        waypoints = actions.float().cpu().numpy()  # 将预测动作转为 numpy（形状：[B, N, 4]）

        # Select waypoint  # 选择某个时间步的轨迹点
        waypoint_select = 4  # 选择第 5 个点
        chosen_waypoint = waypoints[0][waypoint_select].copy()  # 取该点的拷贝
        chosen_waypoint[:2] *= metric_waypoint_spacing  # 还原到米制单位（x,y）
        dx, dy, hx, hy = chosen_waypoint  # 解包：位置(dx,dy)与朝向向量(hx,hy)

        # PD controller  # 简单的基于几何的控制器
        EPS = 1e-8  # 小阈值
        DT = 1 / 3  # 时间步（与 tick_rate 对应）
        if np.abs(dx) < EPS and np.abs(dy) < EPS:  # 若几乎在目标点
            linear_vel_value = 0  # 不前进
            angular_vel_value = 1.0 * clip_angle(np.arctan2(hy, hx)) / DT  # 根据目标朝向转向（注意：clip_angle 未在本文件定义）
        elif np.abs(dx) < EPS:  # 若前向距离几乎为零
            linear_vel_value = 0  # 不前进
            angular_vel_value = 1.0 * np.sign(dy) * np.pi / (2 * DT)  # 快速转向至90度
        else:  # 一般情况
            linear_vel_value = dx / DT  # 前向速度按位移/时间
            angular_vel_value = np.arctan(dy / dx) / DT  # 转角速度按斜率/时间

        linear_vel_value = np.clip(linear_vel_value, 0, 0.5)  # 限制线速度范围 [0, 0.5]
        angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)  # 限制角速度范围 [-1, 1]

        # Velocity limitation  # 速度联合限制（考虑线角速度耦合）
        maxv, maxw = 0.3, 0.3  # 线速度和角速度上限
        if np.abs(linear_vel_value) <= maxv:  # 若线速度未超限
            if np.abs(angular_vel_value) <= maxw:  # 且角速度未超限
                linear_vel_value_limit = linear_vel_value  # 直接使用
                angular_vel_value_limit = angular_vel_value  # 直接使用
            else:  # 角速度超限
                rd = linear_vel_value / angular_vel_value  # 线角半径比
                linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)  # 调整线速度
                angular_vel_value_limit = maxw * np.sign(angular_vel_value)  # 限制角速度到上限
        else:  # 线速度超限
            if np.abs(angular_vel_value) <= 0.001:  # 角速度极小
                linear_vel_value_limit = maxv * np.sign(linear_vel_value)  # 线速度夹到上限
                angular_vel_value_limit = 0.0  # 角速度置零
            else:  # 一般情况
                rd = linear_vel_value / angular_vel_value  # 半径比
                if np.abs(rd) >= maxv / maxw:  # 若半径比足够大
                    linear_vel_value_limit = maxv * np.sign(linear_vel_value)  # 线速度到上限
                    angular_vel_value_limit = maxv * np.sign(angular_vel_value) / np.abs(rd)  # 角速度按比例
                else:  # 半径比不够大
                    linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)  # 线速度按角速度上限调整
                    angular_vel_value_limit = maxw * np.sign(angular_vel_value)  # 角速度到上限

        # Save behavior  # 保存可视化行为与结果
        self.save_robot_behavior(
            current_image_PIL, self.goal_image_PIL, goal_pose_loc_norm, waypoints[0],
            linear_vel_value_limit, angular_vel_value_limit, metric_waypoint_spacing, modality_id.cpu().numpy()
        )  # 生成并保存图像

        print("linear angular", linear_vel_value_limit, angular_vel_value_limit)  # 打印最终速度
        return linear_vel_value_limit, angular_vel_value_limit  # 返回线速度与角速度

    # ----------------------------  # 分隔线
    # Save Robot Behavior Visualization  # 保存机器人行为可视化
    # ----------------------------  # 分隔线
    def save_robot_behavior(self, cur_img, goal_img, goal_pose, waypoints,
                            linear_vel, angular_vel, metric_waypoint_spacing, mask_number):  # 保存可视化图像
        fig = plt.figure(figsize=(34, 16), dpi=80)  # 创建大图
        gs = fig.add_gridspec(2, 2)  # 网格 2x2
        ax_ob = fig.add_subplot(gs[0, 0])  # 子图：当前图像
        ax_goal = fig.add_subplot(gs[1, 0])  # 子图：目标图像
        ax_graph_pos = fig.add_subplot(gs[:, 1])  # 子图：轨迹与坐标

        ax_ob.imshow(np.array(cur_img).astype(np.uint8))  # 显示当前图像
        ax_goal.imshow(np.array(goal_img).astype(np.uint8))  # 显示目标图像

        x_seq = waypoints[:, 0] #generated trajectory is on the robot coordinate. X is front and Y is left.  # 取轨迹 x 序列（车体坐标：x 前进 y 左）
        y_seq_inv = -waypoints[:, 1]           # y 取负以适配绘图方向
        ax_graph_pos.plot(np.insert(y_seq_inv, 0, 0.0), np.insert(x_seq, 0, 0.0), linewidth=4.0, markersize=12, marker='o', color='blue')  # 绘制轨迹（起点为原点）

        # Mask annotation  # 模态掩码注释
        mask_type = int(mask_number[0])  # 掩码类型 id
        mask_texts = [
            "satellite only", "pose and satellite", "satellite and image", "all",
            "pose only", "pose and image", "image only", "language only", "language and pose"
        ]  # 掩码对应文本
        if mask_type < len(mask_texts):  # 范围检查
            ax_graph_pos.annotate(mask_texts[mask_type], xy=(1.0, 0.0), xytext=(-20, 20), fontsize=18, textcoords='offset points')  # 右下角标注

        ax_ob.set_title("Egocentric current image", fontsize=18)  # 当前图像标题
        ax_goal.set_title("Egocentric goal image", fontsize=18)  # 目标图像标题
        ax_graph_pos.tick_params(axis='x', labelsize=15)  # x 轴字体
        ax_graph_pos.tick_params(axis='y', labelsize=15)  # y 轴字体
        
        if int(mask_number[0]) == 1 or int(mask_number[0]) == 3 or int(mask_number[0]) == 4 or int(mask_number[0]) == 5 or int(mask_number[0]) == 8:  # 若包含位姿信息
            ax_graph_pos.plot(-goal_pose[1], goal_pose[0], marker = '*', color='red', markersize=15)  # 绘制目标点
        else:                           # 否则固定坐标轴范围
            ax_graph_pos.set_xlim(-3.0, 3.0)  # x 轴范围
            ax_graph_pos.set_ylim(-0.1, 10.0)  # y 轴范围
        ax_graph_pos.set_xlim(-3.0, 3.0)  # 再次设置 x 轴范围（确保范围）
        ax_graph_pos.set_ylim(-0.1, 10.0)  # 再次设置 y 轴范围
                        
        ax_graph_pos.set_title("Normalized generated 2D trajectories from OmniVLA", fontsize=18)  # 轨迹图标题
        
        save_path = os.path.join(self.datastore_path_image, f"{self.count_id}_ex.jpg")  # 保存路径
        plt.savefig(save_path)  # 保存图像

    # ----------------------------  # 分隔线
    # Custom Collator  # 自定义组批器
    # ----------------------------  # 分隔线
    def collator_custom(self, instances, model_max_length, pad_token_id, padding_side="right", pixel_values_dtype=torch.float32):  # 将单样本打包成 batch，并做填充
        IGNORE_INDEX = -100  # 忽略标签
        print("\n===== 开始 collator_custom 组批处理 =====")
        print(f"输入样本数量: {len(instances)}")  # 打印样本数（此处为1）

        # 1. 文本处理：input_ids 填充与截断
        input_ids_list = [inst["input_ids"] for inst in instances]
        print(f"组批前 input_ids 各自长度: {[len(ids) for ids in input_ids_list]}")
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
        print(f"填充后 input_ids 形状 (未截断): {input_ids.shape}")
        input_ids = input_ids[:, :model_max_length]  # 截断至模型最大长度
        print(f"截断后 input_ids 形状: {input_ids.shape}")

        # 2. 标签处理：labels 填充与截断
        labels_list = [inst["labels"] for inst in instances]
        labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        print(f"填充后 labels 形状 (未截断): {labels.shape}")
        labels = labels[:, :model_max_length]  # 截断至模型最大长度
        print(f"截断后 labels 形状: {labels.shape}")

        # 3. 注意力掩码生成
        attention_mask = input_ids.ne(pad_token_id)
        print(f"attention_mask 形状: {attention_mask.shape}，有效 token 数量: {attention_mask.sum().item()}")

        # 4. 图像处理：当前图像与目标图像拼接
        pixel_values = [inst["pixel_values_current"] for inst in instances]
        print(f"单样本当前图像形状: {pixel_values[0].shape}")
        
        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_goal" in instances[0]:
                pixel_values_goal = [inst["pixel_values_goal"] for inst in instances]
                print(f"单样本目标图像形状: {pixel_values_goal[0].shape}")
                # 堆叠当前图像和目标图像，并在第1维度拼接（图像数量维度）
                pixel_values_stacked = torch.stack(pixel_values)
                pixel_values_goal_stacked = torch.stack(pixel_values_goal)
                print(f"堆叠后当前图像形状: {pixel_values_stacked.shape}")
                print(f"堆叠后目标图像形状: {pixel_values_goal_stacked.shape}")
                pixel_values = torch.cat((pixel_values_stacked, pixel_values_goal_stacked), dim=1)
                print(f"拼接后图像总形状: {pixel_values.shape} (batch_size, 图像数, 通道, 高, 宽)")
            else:
                pixel_values = torch.stack(pixel_values)
                print(f"仅当前图像堆叠后形状: {pixel_values.shape}")
        else:
            raise ValueError(f"Unsupported `pixel_values` type: {type(pixel_values)}")

        # 5. 动作序列堆叠
        actions = torch.stack([torch.from_numpy(np.copy(inst["actions"])) for inst in instances])
        print(f"动作序列堆叠后形状: {actions.shape} (batch_size, 时间步, 动作维度)")

        # 6. 目标位姿堆叠
        goal_pose = torch.stack([torch.from_numpy(np.copy(inst["goal_pose"])) for inst in instances])
        print(f"目标位姿堆叠后形状: {goal_pose.shape} (batch_size, 位姿维度)")

        # 7. 组装输出 batch
        output = dict(
            pixel_values=pixel_values.to(dtype=pixel_values_dtype),
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            goal_pose=goal_pose,
        )
        if "dataset_name" in instances[0]:
            output["dataset_names"] = [inst["dataset_name"] for inst in instances]
            print(f"包含数据集名称: {output['dataset_names']}")

        print(f"最终 batch 包含的键: {list(output.keys())}")
        print("===== collator_custom 组批处理结束 =====\n")
        return output

    # ----------------------------  # 分隔线
    # Transform Data to Dataset Format  # 将单条数据转换为训练/推理样本格式
    # ----------------------------  # 分隔线
    def transform_datatype(self, inst_obj, actions, goal_pose_cos_sin,
                        current_image_PIL, goal_image_PIL, prompt_builder, action_tokenizer,
                        base_tokenizer, image_transform, predict_stop_token=True):  # 构造单样本
        IGNORE_INDEX = -100  # 忽略标签
        print("\n===== 开始 transform_datatype 处理 =====")
        
        # 1. 动作序列处理
        current_action = actions[0]  # 当前动作
        future_actions = actions[1:]  # 未来动作序列
        print(f"原始动作序列形状: {actions.shape} (前1个是当前动作，后{len(future_actions)}个是未来动作)")
        # 原始动作序列形状: (8, 4) (前1个是当前动作，后7个是未来动作)

        # 动作转字符串
        future_actions_string = ''.join(action_tokenizer(future_actions))  # 未来动作编码为字符串
        current_action_string = action_tokenizer(current_action)  # 当前动作编码为字符串
        action_chunk_string = current_action_string + future_actions_string  # 合并动作序列
        action_chunk_len = len(action_chunk_string)
        print(f"合并后的动作字符串: {action_chunk_string[:50]}... (长度: {action_chunk_len})")  # 打印前50字符避免过长
        
        # 2. 对话格式构建
        if inst_obj == "xxxx":  # 无语言指令
            conversation = [
                {"from": "human", "value": "No language instruction"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:  # 有语言指令
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {inst_obj}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        print(f"构建的对话格式: {conversation}")
        
        # 3. 提示词构建
        prompt_builder = prompt_builder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        final_prompt = prompt_builder.get_prompt()
        print(f"最终提示词 (前100字符): {final_prompt[:100]}...")  # 打印前100字符
        
        # 4. 文本 token 化
        tokenized = base_tokenizer(final_prompt, add_special_tokens=True)
        input_ids = torch.tensor(tokenized.input_ids)
        labels = input_ids.clone()
        # 仅保留动作部分作为标签，其余置为 IGNORE_INDEX
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX
        if not predict_stop_token:
            labels[-1] = IGNORE_INDEX
        print(f"token化后 input_ids 长度: {len(input_ids)}, 形状: {input_ids.shape}")
        print(f"labels 中有效标签位置 (非 {IGNORE_INDEX}): {torch.sum(labels != IGNORE_INDEX).item()} 个")
        
        # 5. 图像处理
        pixel_values_current = image_transform(current_image_PIL)  # 当前图像预处理
        pixel_values_goal = image_transform(goal_image_PIL)  # 目标图像预处理
        print(f"当前图像预处理后形状: {pixel_values_current.shape} (通道数: {pixel_values_current.shape[0]})")
        print(f"目标图像预处理后形状: {pixel_values_goal.shape}")
        
        # 6. 组装单样本字典
        dataset_name = "lelan"
        result = dict(
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
        print(f"单样本字典构建完成，包含键: {list(result.keys())}")
        print("===== transform_datatype 处理结束 =====\n")
        return result

    # ----------------------------  # 分隔线
    # Data Transformer for OmniVLA  # 数据打包器：将输入组装为模型 batch
    # ----------------------------  # 分隔线
    def data_transformer_omnivla(self, current_image_PIL, lan_inst, goal_image_PIL, goal_pose_loc_norm,
                                 prompt_builder, action_tokenizer, processor):  # 构造 batch
        actions = np.random.rand(8, 4)  # dummy actions  # 随机动作序列占位（形状 N=8，每步4维）
        goal_pose_cos_sin = goal_pose_loc_norm  # 使用已归一化的位姿作为本体输入

        batch_data = self.transform_datatype(  # 构造单条样本
            lan_inst, actions, goal_pose_cos_sin,
            current_image_PIL, goal_image_PIL,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
        )

        batch = self.collator_custom(  # 用自定义 collator 组装 batch
            instances=[batch_data],
            model_max_length=processor.tokenizer.model_max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            padding_side="right"
        )
        return batch  # 返回 batch

    # ----------------------------  # 分隔线
    # Run Forward Pass  # 执行一次前向传递
    # ----------------------------  # 分隔线

    def run_forward_pass(self, vla, action_head, noisy_action_projector, pose_projector,
                            batch, action_tokenizer, device_id, use_l1_regression, use_diffusion,
                            use_film, num_patches, compute_diffusion_l1=False,
                            num_diffusion_steps_train=None, mode="vali", idrun=0) -> Tuple[torch.Tensor, Dict[str, float]]:  # 前向推理与提取动作

            metrics = {}  # 度量占位
            noise, noisy_actions, diffusion_timestep_embeddings = None, None, None  # 扩散相关占位

            # Determine modality  # 根据全局开关确定输入模态类型 id
            if satellite and not lan_prompt and not pose_goal and not image_goal:
                modality_id = torch.astensor([0], dtype=torch.float32)  # 仅卫星
            elif satellite and not lan_prompt and pose_goal and not image_goal:
                modality_id = torch.as_tensor([1], dtype=torch.float32)  # 姿态+卫星
            elif satellite and not lan_prompt and not pose_goal and image_goal:
                modality_id = torch.as_tensor([2], dtype=torch.float32)  # 卫星+图像
            elif satellite and not lan_prompt and pose_goal and image_goal:
                modality_id = torch.as_tensor([3], dtype=torch.float32)  # 全部
            elif not satellite and not lan_prompt and pose_goal and not image_goal:
                modality_id = torch.as_tensor([4], dtype=torch.float32)  # 仅姿态
            elif not satellite and not lan_prompt and pose_goal and image_goal:
                modality_id = torch.as_tensor([5], dtype=torch.float32)  # 姿态+图像
            elif not satellite and not lan_prompt and not pose_goal and image_goal:
                modality_id = torch.as_tensor([6], dtype=torch.float32)  # 仅图像
            elif not satellite and lan_prompt and not pose_goal and not image_goal:
                modality_id = torch.as_tensor([7], dtype=torch.float32)  # 仅语言
            elif not satellite and lan_prompt and pose_goal and not image_goal:
                modality_id = torch.as_tensor([8], dtype=torch.float32)  # 语言+姿态
            
            # 新增打印：模态信息与输入形状
            print("\n===== 开始 run_forward_pass 前向推理 =====")
            print(f"运行模式: {mode}, 设备ID: {device_id}")
            modality_name_map = {0:"仅卫星",1:"姿态+卫星",2:"卫星+图像",3:"全部模态",4:"仅姿态",5:"姿态+图像",6:"仅图像",7:"仅语言",8:"语言+姿态"}
            print(f"输入模态: ID={modality_id.item()}, 类型={modality_name_map.get(modality_id.item(), '未知模态')}")
            print(f"模型输入关键形状:")
            print(f"  input_ids: {batch['input_ids'].shape}, attention_mask: {batch['attention_mask'].shape}")
            print(f"  pixel_values: {batch['pixel_values'].shape}, goal_pose: {batch['goal_pose'].shape}")

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # 无梯度 + bfloat16 自动混合精度
                print(f"\n调用 VLA 模型前向推理（混合精度: bfloat16）...")
                output: CausalLMOutputWithPast = vla(  # 前向调用 VLA（类型注解依赖 transformers 的输出类型）
                    input_ids=batch["input_ids"].to(device_id),  # 文本 ids
                    attention_mask=batch["attention_mask"].to(device_id),  # 注意力掩码
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),  # 图像张量
                    modality_id=modality_id.to(torch.bfloat16).to(device_id),  # 模态 id
                    labels=batch["labels"].to(device_id),  # 标签（可计算loss）
                    output_hidden_states=True,  # 输出隐藏状态
                    # output_attentions=True,  # 新增：启用注意力权重输出 开启后导致输出不一样，不要开！！！！！！
                    proprio=batch["goal_pose"].to(torch.bfloat16).to(device_id),  # 本体输入（目标位姿）
                    proprio_projector=pose_projector,  # 本体投影器
                    noisy_actions=noisy_actions if use_diffusion else None,  # 扩散噪声动作（未用）
                    noisy_action_projector=noisy_action_projector if use_diffusion else None,  # 扩散动作投影器（未用）
                    diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,  # 扩散时间步嵌入（未用）
                    use_film=use_film,  # 是否启用 FiLM（特征调制）
                )
            print(f"VLA 推理完成，输出包含: {list(output.keys())}")

            # Prepare data for metrics  # 准备掩码与动作部分的隐藏状态
            ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)  # 标签右移一位（忽略 BOS）
            current_action_mask = get_current_action_mask(ground_truth_token_ids)  # 当前动作掩码
            next_actions_mask = get_next_actions_mask(ground_truth_token_ids)  # 后续动作掩码
            print(f"\n标签处理: 原始 labels 形状={batch['labels'].shape}, 右移后={ground_truth_token_ids.shape}")
            print(f"动作掩码: 当前动作有效数={current_action_mask.sum().item()}, 后续动作有效数={next_actions_mask.sum().item()}")
            
            # Get last layer hidden states  # 取最后一层隐藏状态
            last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)  # 形状：批大小、序列长度、隐藏维
            print(f"最后一层隐藏状态形状: {last_hidden_states.shape} (batch, seq_len, hidden_dim)")
            
            # Get hidden states for text portion of prompt+response (after the vision patches)  # 取文本部分（跳过视觉 patch）
            text_hidden_states = last_hidden_states[:, num_patches:-1]  # 从 num_patches 到倒数第二个（最后通常是 EOS）
            print(f"跳过 {num_patches} 个视觉 patch 后，文本隐藏状态形状: {text_hidden_states.shape}")

            # Get hidden states for action portion of response  # 取动作对应的隐藏状态
            batch_size = batch["input_ids"].shape[0]  # 批大小
            actions_hidden_states = (
                text_hidden_states[current_action_mask | next_actions_mask]  # 选出动作相关 token
                .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)  # 重塑为 (B, 动作标记长度, D)
                .to(torch.bfloat16)  # 转 bfloat16
            )  # (B, act_chunk_len, D)
            print(f"动作相关隐藏状态形状: {actions_hidden_states.shape} (batch, action_token_len, hidden_dim)")

            with torch.no_grad():  # 不求梯度
                print(f"\n通过 action_head 预测连续动作...")
                predicted_actions = action_head.predict_action(actions_hidden_states, modality_id.to(torch.bfloat16).to(device_id))  # 通过动作头预测连续动作
            print(f"动作预测完成，预测动作形状: {predicted_actions.shape} (batch, num_steps, action_dim)")
            print("===== run_forward_pass 前向推理结束 =====\n")


            # ===== 新增：提取文本Query向量 =====
            print("\n===== 提取文本Query向量 =====")
            llm = vla.language_model

            if hasattr(llm, 'model') and hasattr(llm.model, 'layers'):
                last_layer = llm.model.layers[-1]
            elif hasattr(llm, 'layers'):
                last_layer = llm.layers[-1]
            else:
                last_layer = None
            
            if last_layer is not None and hasattr(last_layer, 'self_attn'):
                self_attn = last_layer.self_attn
                if hasattr(self_attn, 'q_proj'):
                    text_queries = self_attn.q_proj(text_hidden_states.to(torch.bfloat16))
                    text_keys = self_attn.k_proj(text_hidden_states.to(torch.bfloat16))
                    text_values = self_attn.v_proj(text_hidden_states.to(torch.bfloat16))
                    print(f"文本Query形状: {text_queries.shape}")
                    print(f"文本Key形状: {text_keys.shape}")
                    print(f"文本Value形状: {text_values.shape}")
            print("===== Query提取完成 =====\n")
            
            # ===== 新增部分结束 =====


            # # 新增：提取文本 QKV 相关注意力信息
            # if hasattr(output, "attentions") and output.attentions is not None:
            #     print("\n===== 提取文本 QKV 相关注意力信息 =====")
            #     # 1. 获取最后一层注意力权重（也可遍历所有层）
            #     last_layer_attentions = output.attentions[-1]  # 形状：[B, num_heads, seq_len, seq_len]
            #     print(f"最后一层注意力权重形状: {last_layer_attentions.shape}")
                
            #     # 2. 确定文本部分的序列范围（跳过视觉 patch + 仅保留文本 token）
            #     # 之前已计算：text_hidden_states = last_hidden_states[:, num_patches:-1]
            #     text_start_idx = num_patches  # 文本开始位置（跳过视觉 patch）
            #     text_end_idx = -1  # 文本结束位置（排除 EOS）
            #     text_seq_len = text_end_idx - text_start_idx  # 文本序列长度（对应之前的 52）
                
            #     # 3. 提取文本部分的注意力权重（仅关注文本 token 之间的注意力）
            #     text_attentions = last_layer_attentions[:, :, text_start_idx:text_end_idx, text_start_idx:text_end_idx]
            #     print(f"文本部分注意力权重形状: {text_attentions.shape} (B, num_heads, text_seq_len, text_seq_len)")
            #     text_attentions_bf16 = text_attentions.to(torch.bfloat16)  # GPU 张量可以
            #     print(f"转换为 bfloat16 后的文本注意力权重形状: {text_attentions_bf16.shape}")
            #     text_attentions = text_attentions.to(torch.float16).cpu().numpy()
            #     # 4. 从注意力权重反推 QKV 关联（核心逻辑）
            #     # QKV 是注意力层的输入，满足：Attention(Q,K,V) = Softmax(QK^T/√d_k) * V
            #     # 虽无法直接获取原始 QKV，但可通过注意力权重分析文本 token 间的 QK 相似度、KV 贡献
            #     num_heads = text_attentions.shape[1]
            #     d_k = last_hidden_states.shape[-1] // num_heads  # 每个头的维度（隐藏层维度 / 头数）
            #     print(f"每个注意力头的维度 (d_k): {d_k}")
            #     print(f"文本 QKV 隐含形状: Q/K/V = (B, num_heads, text_seq_len, d_k)")
                
            #     # 可选：保存文本注意力权重（可后续分析 QKV 交互）
            #     text_qkv_attention_info = {
            #         "text_attentions": text_attentions,
            #         "num_heads": num_heads,
            #         "d_k": d_k,
            #         "text_seq_len": text_seq_len,
            #         "text_token_range": (text_start_idx, text_end_idx)
            #     }
            #     print("===== 文本 QKV 相关信息提取完成 =====")
            # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)  # 返回预测动作与模态 id
            return predicted_actions, modality_id  # 返回

                
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

# ===============================================================  # 分隔线
# Main Entry  # 主入口
# ===============================================================  # 分隔线
if __name__ == "__main__":  # 脚本入口
    # select modality  # 选择输入模态（全局开关）
    pose_goal = False  # 是否使用目标位姿
    satellite = False  # 是否使用卫星图像
    image_goal = False  # 是否使用目标图像
    lan_prompt = True  # 是否使用语言指令

    # Goal definitions  # 目标定义
    # lan_inst_prompt = "move toward blue trash bin"  # 示例指令
    # lan_inst_prompt = "turn right and go straight"  # 示例指令
    # lan_inst_prompt = "move toward black tv monitor"  # 示例指令
    lan_inst_prompt = "move toward white office cabinet"  # 实际使用的指令
    # lan_inst_prompt = "turn right and move forward"  # 示例指令

    goal_lat, goal_lon, goal_compass = 37.8738930785863, -122.26746181032362, 0.0  # 目标经纬与罗盘角（度）
    goal_utm = utm.from_latlon(goal_lat, goal_lon)  # 转为 UTM
    goal_compass = -float(goal_compass) / 180.0 * math.pi  # 转为弧度并取反
    goal_image_PIL = Image.open("./inference/goal_img.jpg").convert("RGB")  # 读取目标图像

    # Define models (VLA, action_head, pose_projector, processor, etc.)  # 定义/加载模型与组件
    cfg = InferenceConfig()  # 创建配置
    vla, action_head, pose_projector, device_id, NUM_PATCHES, action_tokenizer, processor = define_model(cfg)  # 加载模型组件

    # Run inference  # 运行推理
    inference = Inference(
        save_dir="./inference",  # 保存目录
        lan_inst_prompt=lan_inst_prompt,  # 语言指令
        goal_utm=goal_utm,  # 目标UTM
        goal_compass=goal_compass,  # 目标朝向
        goal_image_PIL=goal_image_PIL,  # 目标图像
        action_tokenizer=action_tokenizer,  # 动作 tokenizer
        processor=processor,  # 处理器
    )  # 实例化推理类
    inference.run()  # 执行一次推理
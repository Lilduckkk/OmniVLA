
import torch
import torch.nn.functional as F


class TrajCost:
    def __init__(self, batch):
        # self.is_map = False
        self.origin_trajectory_ref = batch['original_normalized_trajectory']
        self.segmentation_mask = batch['segmentation_mask']
        return None


    
    # def CostofTraj(self, predicted_actions):
    #     # origin_trajectory_ref和predicted_actions点的数量不相等
    #     # origin_trajectory_ref shape: torch.Size([1, 20, 4]), predicted_actions shape: torch.Size([1, 8, 4])
    #     # origin_trajectory_ref用了最后一个点填充了不足的点
    #     # print(f"origin_trajectory_ref:{origin_trajectory_ref} ")
    #     # 首先提取origin_trajectory_ref的有效点，查看最后不重复的点的位置
    #     for batch_idx in range(self.origin_trajectory_ref.shape[0]):
    #         # 假设批次大小为1，我们只处理第一个（也是唯一的）批次
    #         trajectory = self.origin_trajectory_ref[batch_idx]  # shape: [20, 4]
            
    #         # 1. 计算相邻点之间的差异
    #         # torch.diff(trajectory, dim=0) 会得到 [19, 4] 的张量
    #         # 差异为 0 表示点与前一个点相同
    #         # 只有当所有维度都为 0 时，才表示点是重复的填充点
            
    #         # 计算相邻点之间的绝对差异
    #         diffs = torch.abs(torch.diff(trajectory, dim=0)) # shape: [19, 4]

    #         # 2. 检查每一对相邻点是否完全相同 (即所有坐标的差异都为 0)
    #         # torch.all(diffs == 0, dim=1) 检查每行（即每个时间步的差异）是否全为 0
    #         is_identical = torch.all(diffs == 0, dim=1) # shape: [19] (Boolean Tensor)
    #         # True 表示 trajectory[i+1] == trajectory[i]
            
    #         # 3. 找到第一个重复点（即第一个 True）出现的位置 i
    #         # 这个 i 对应于 trajectory 的索引 i+1 (因为 diffs 是从索引 1 开始计算的)
            
    #         # 找到所有重复点的索引
    #         # torch.nonzero(is_identical) 得到重复点的索引（在 diffs/is_identical 中的索引）
    #         identical_indices = torch.nonzero(is_identical, as_tuple=True)[0]
            
    #         if identical_indices.numel() > 0:
    #             # 如果存在重复点
    #             # 第一个重复点的索引在 is_identical 中是 identical_indices[0]
    #             # 对应的有效轨迹的末尾点（即不重复的最后一个点）在 trajectory 中的索引是：
    #             # last_valid_idx = identical_indices[0] (因为 is_identical[i] 检查的是 i 和 i+1)
                
    #             # 例如：如果 trajectory[5] == trajectory[4]，那么 is_identical[4] 为 True
    #             # 此时 last_valid_idx 应该是 4
    #             last_valid_idx = identical_indices[0].item()
                
    #             # 4. 提取有效轨迹
    #             # 有效点是从索引 0 到 last_valid_idx (包含 last_valid_idx)
    #             # 例如，如果 last_valid_idx = 4，则有效点是 0, 1, 2, 3, 4，共 5 个点
    #             valid_trajectory = trajectory[:last_valid_idx + 1]
    #         else:
    #             # 如果没有找到重复点，说明整个轨迹都是有效的
    #             last_valid_idx = len(trajectory) - 1
    #             valid_trajectory = trajectory
                
    #         # 现在，valid_trajectory 包含了 origin_trajectory_ref 中的所有有效点
    #         # 它的 shape 是 [有效点数量, 4]
    #         # 如果需要保留批次维度，可以：
    #         # valid_trajectory_ref = valid_trajectory.unsqueeze(0) # shape: [1, 有效点数量, 4]
    #         # print(f"trajectory:{trajectory} ")
    #         # print(f"valid_trajectory:{valid_trajectory} ")
    #         # print(f"valid_trajectory_ref:{valid_trajectory_ref} ")
    #         # print(f"predicted_actions :{predicted_actions} ")
    #         # print(f"最后一个有效点的索引: {last_valid_idx}")
    #         # print(f"有效轨迹长度: {valid_trajectory.shape[0]}")
    #         # print(f"有效轨迹 shape: {valid_trajectory_ref.shape}")
    #     return None

    def CostofTraj(self, predicted_actions):
        total_loss = 0.0
        batch_size = self.origin_trajectory_ref.shape[0]
        # 首先提取origin_trajectory_ref的有效点，查看最后不重复的点的位置
        for batch_idx in range(batch_size):
            # 假设批次大小为1，我们只处理第一个（也是唯一的）批次
            trajectory = self.origin_trajectory_ref[batch_idx]  # shape: [20, 4]
            
            # 1. 计算相邻点之间的差异            
            # 计算相邻点之间的绝对差异
            diffs = torch.abs(torch.diff(trajectory, dim=0)) # shape: [19, 4]

            # 2. 检查每一对相邻点是否完全相同 (即所有坐标的差异都为 0)
            # torch.all(diffs == 0, dim=1) 检查每行（即每个时间步的差异）是否全为 0
            is_identical = torch.all(diffs == 0, dim=1) # shape: [19] (Boolean Tensor)
            # True 表示 trajectory[i+1] == trajectory[i]
            
            # 3. 找到第一个重复点（即第一个 True）出现的位置 i            
            # 找到所有重复点的索引
            identical_indices = torch.nonzero(is_identical, as_tuple=True)[0]
            
            if identical_indices.numel() > 0:
                # 如果存在重复点
                # 第一个重复点的索引在 is_identical 中是 identical_indices[0]
                last_valid_idx = identical_indices[0].item()
                
                # 4. 提取有效轨迹
                valid_trajectory = trajectory[:last_valid_idx + 1]
            else:
                # 如果没有找到重复点，说明整个轨迹都是有效的
                last_valid_idx = len(trajectory) - 1
                valid_trajectory = trajectory
            loss = self.calculate_dtw_torch(predicted_actions[batch_idx], valid_trajectory)
            total_loss = total_loss + loss
        avg_loss = total_loss / batch_size
        return avg_loss
        
    def CostofSegMask(self, predicted_actions):
        # print(f"seg_mask:{self.segmentation_mask.shape} ")
        return None
    

    def calculate_dtw_torch(self, prediction, ground_truth):
        """
        Strict PyTorch implementation of the numpy DTW function.
        
        Args:
            prediction: [N, 4] Tensor
            ground_truth: [M, 4] Tensor
        """
        n = prediction.shape[0]
        m = ground_truth.shape[0]
        device = prediction.device
        ground_truth = ground_truth.to(device)
        # 初始化 Cost Matrix
        # 使用 Python list 模拟矩阵，避免 PyTorch 的 inplace 操作导致梯度中断或报错
        # cost_matrix[i][j] 存储的是一个 scalar tensor
        cost_matrix = [[torch.tensor(float('inf'), device=device) for _ in range(m + 1)] for _ in range(n + 1)]
        
        # cost_matrix[0, 0] = 0
        cost_matrix[0][0] = torch.tensor(0.0, device=device)

        # 严格按照原函数的双重循环逻辑
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # 1. 计算欧氏距离 (Euclidean Distance)
                # 对应: np.linalg.norm(prediction[i-1] - ground_truth[j-1])
                euclidean_distance = torch.norm(prediction[i - 1] - ground_truth[j - 1])

                # 2. 找到左、上、左上三个格子中的最小值
                # 对应: min(insertion, deletion, match)
                # 使用 torch.stack 将三个 scalar tensor 堆叠，再求 min，确保梯度可以传导
                prev_costs = torch.stack([
                    cost_matrix[i - 1][j],      # Insertion
                    cost_matrix[i][j - 1],      # Deletion
                    cost_matrix[i - 1][j - 1]   # Match
                ])
                
                # torch.min 返回 (values, indices)，我们只需要 values
                min_prev_cost = torch.min(prev_costs)

                # 3. 累加 Cost
                # 对应: cost_matrix[i, j] = euclidean_distance + min_prev_cost
                cost_matrix[i][j] = euclidean_distance + min_prev_cost

        # 返回最终的累积代价
        return cost_matrix[n][m]
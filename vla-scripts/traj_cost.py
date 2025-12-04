
import torch
import torch.nn.functional as F


class TrajCost:
    def __init__(self):
        # self.is_map = False
        return None


    
    def CostofTraj(self, origin_trajectory_ref, predicted_actions):
        # origin_trajectory_ref和predicted_actions点的数量不相等
        # origin_trajectory_ref shape: torch.Size([1, 20, 4]), predicted_actions shape: torch.Size([1, 8, 4])
        # origin_trajectory_ref用了最后一个点填充了不足的点
        # print(f"origin_trajectory_ref:{origin_trajectory_ref} ")
        return None
    
    def CostofSegMask(self, seg_mask, predicted_actions):

        return None
    

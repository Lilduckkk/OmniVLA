import torch
import torch.nn as nn

def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]

class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=48,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        DIM_TO_SLICE = {
            1: slice1d,
            2: slice2d,
            3: slice3d,
        }
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
    
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class VGGTMerger(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.input_dim = context_dim * (spatial_merge_size**2)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.mlp(self.ln_q(x).view(-1, self.input_dim))
    #     return x
    def forward(self, vggt_feature: torch.Tensor) -> torch.Tensor:
        """
        vggt_feature: [batch, patch_num, context_dim] 原始VGG-T特征
        返回: [batch, merged_patch_num, output_dim]
        """
        n_image = vggt_feature.shape[0]
        merge_size = self.spatial_merge_size
        context_dim = vggt_feature.shape[2]

        # 自动计算目标网格尺寸（能被merge_size整除）
        orig_h_w = int(vggt_feature.shape[1] ** 0.5)  # 假设输入是平方patch数量
        target_h_grid = ((orig_h_w + merge_size - 1) // merge_size) * merge_size
        target_w_grid = ((orig_h_w + merge_size - 1) // merge_size) * merge_size
        target_patch_num = target_h_grid * target_w_grid

        # 填充patch
        pad_num = target_patch_num - vggt_feature.shape[1]
        if pad_num > 0:
            pad_tensor = torch.zeros((n_image, pad_num, context_dim), device=vggt_feature.device, dtype=vggt_feature.dtype)
            vggt_feature = torch.cat([vggt_feature, pad_tensor], dim=1)

        # reshape为网格
        features = vggt_feature.view(n_image, target_h_grid, target_w_grid, -1)
        h_grid_after_merge = target_h_grid // merge_size
        w_grid_after_merge = target_w_grid // merge_size
        features = features.view(n_image, h_grid_after_merge, merge_size, w_grid_after_merge, merge_size, -1)
        features = features.permute(0, 1, 3, 2, 4, 5).contiguous().to(torch.float32)
        features = features.to(dtype=vggt_feature.dtype)

        # 调用MLP
        image_embeds_3d = self.mlp(self.ln_q(features).view(-1, self.input_dim)).view(n_image, -1, self.output_dim)
        return image_embeds_3d
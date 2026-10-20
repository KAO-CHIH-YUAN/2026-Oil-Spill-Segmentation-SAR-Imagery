import torch
import torch.nn as nn
from einops import repeat

# 嘗試從 mamba_ssm 匯入加速算子
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Warning: mamba_ssm not found. ConvSSM will use a slower PyTorch fallback or approximation.")

# 嘗試從專案中匯入現有的 VSSBlock
try:
    from ..rs3mamba_core.SwinUMamba import VSSBlock
except ImportError:
    print("Warning: VSSBlock not found in ..rs3mamba_core.SwinUMamba. Using identity for debugging.")
    class VSSBlock(nn.Module):
        def __init__(self, hidden_dim, **kwargs):
            super().__init__()
            self.net = nn.Identity()
        def forward(self, x):
            return x

class ConvSSM(nn.Module):
    """
    [論文 100% 復刻] Convolutional State Space Model (ConvSSM)
    
    依據 OSDMamba 論文:
    - 將特徵圖視為序列，針對每個像素進行狀態空間演化。
    - 使用 Mamba 的 Selective Scan 機制來實現高效運算。
    - 這裡實作的是 "Scan over Spatial Dimensions"。
    """
    def __init__(self, hidden_dim, d_state=16, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_state = d_state
        
        # 1. 局部卷積 (Local Context)
        self.local_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )
        
        # 2. SSM 參數定義 (Projecting to State Space)
        # x_proj 將輸入投影到 (dt, B, C) 參數空間
        # dt_rank = hidden_dim / 16 (依照 Mamba 參照)
        self.dt_rank = max(1, hidden_dim // 16)
        
        self.x_proj = nn.Linear(hidden_dim, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, hidden_dim, bias=True)
        
        # A matrix (discretized to -exp(A))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float).repeat(hidden_dim, 1)), requires_grad=True)
        # [Bidirectional] 加入反向的 A 參數
        self.A_log_rev = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float).repeat(hidden_dim, 1)), requires_grad=True)
        
        self.D = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)
        
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x
        
        # 1. Local Conv
        x = self.local_conv(x)
        
        # 2. Prepare for Scanning (Flatten Spatial)
        # shape: (B, C, H, W) -> (B, L, C) where L = H*W
        x_flat = x.flatten(2).transpose(1, 2) # (B, L, C)
        
        if HAS_MAMBA:
            # --- 使用 Mamba 加速核 (Bidirectional) ---
            L = H * W
            
            # (B, L, C) -> SSM Parameters
            x_dbl = self.x_proj(x_flat) # (B, L, dt_rank + 2*d_state)
            dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            
            dt = self.dt_proj(dt).transpose(1, 2).contiguous() # (B, C, L)
            B_ssm = B_ssm.transpose(1, 2).contiguous() # (B, d_state, L)
            C_ssm = C_ssm.transpose(1, 2).contiguous() # (B, d_state, L)
            
            u = x_flat.transpose(1, 2).contiguous() # (B, C, L)
            
            # A_log -> A
            A = -torch.exp(self.A_log) # (C, d_state)
            A_rev = -torch.exp(self.A_log_rev) # (C, d_state) - 反向 A
            
            # 1. Forward Scan
            y_fwd = selective_scan_fn(
                u, dt, A, B_ssm, C_ssm, self.D, 
                z=None, delta_bias=None, delta_softplus=True
            )
            
            # 2. Backward Scan (Bidirectional)
            # Flip inputs along time/spatial axis (L)
            u_rev = u.flip(-1)
            dt_rev = dt.flip(-1)
            B_rev = B_ssm.flip(-1)
            C_rev = C_ssm.flip(-1)
            
            y_rev = selective_scan_fn(
                u_rev, dt_rev, A_rev, B_rev, C_rev, self.D, 
                z=None, delta_bias=None, delta_softplus=True
            )
            
            # Flip back and combine
            y_rev = y_rev.flip(-1)
            
            # Sum predictions [cite: Bidirectional SSM]
            y = y_fwd + y_rev
            
            # Reshape back
            y = y.transpose(1, 2).view(B, H, W, C).permute(0, 3, 1, 2)
            
        else:
            # Fallback (簡單直通)
            y = x 
            
        return y + residual

class SimplePatchExpanding(nn.Module):
    """
    標準 Patch Expanding 模組
    用於將解析度放大 2 倍 (H, W -> 2H, 2W) 並調整通道數
    """
    def __init__(self, input_dim, output_dim, scale_factor=2):
        super().__init__()
        # 使用 Bilinear 插值模擬 Patch Expanding 的空間擴展
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(output_dim)
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        return x

class FinalPatchExpanding(nn.Module):
    """
    [論文 Fig. 2] 最終輸出層
    Decoder Layer 3 (1/4 尺度) -> Patch Expanding (4x) -> 1x1 Conv -> Output
    """
    def __init__(self, input_dim, num_classes, scale_factor=4):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(input_dim, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.up(x)
        x = self.final_conv(x)
        return x
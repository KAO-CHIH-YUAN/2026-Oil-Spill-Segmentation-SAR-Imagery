import torch
import torch.nn as nn
from .modules import ConvSSM, SimplePatchExpanding, FinalPatchExpanding, VSSBlock
# 匯入您現有的 VSSMEncoder
from ..rs3mamba_core.SwinUMamba import VSSMEncoder

class PermutedVSSBlock(nn.Module):
    """
    Wrapper for VSSBlock to handle shape permutation.
    VSSBlock expects (B, H, W, C), but OSDMamba uses (B, C, H, W).
    """
    def __init__(self, hidden_dim, drop_path=0.0, **kwargs):
        super().__init__()
        self.block = VSSBlock(hidden_dim=hidden_dim, drop_path=drop_path, **kwargs)
        
    def forward(self, x):
        # x: [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.block(x)
        # x: [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class OSDMamba(nn.Module):
    def __init__(self, 
                 in_chans=1, 
                 num_classes=1, 
                 # VMamba-Tiny 設定 (Encoder)
                 depths=[2, 2, 9, 2], 
                 dims=[96, 192, 384, 768],
                 deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        # --- 1. Encoder (VMamba-Tiny) ---
        # in_chans 傳遞給 VSSMEncoder，這會自動調整第一層 PatchEmbed
        self.encoder = VSSMEncoder(
            in_chans=in_chans, 
            depths=depths, 
            dims=dims,
            use_checkpoint=True # 開啟 Checkpoint 節省訓練 VRAM
        )
        
        # [新增] 自動載入預訓練權重
        self._load_pretrained_encoder()
        
        # --- 2. Asymmetric Decoder ---
        
        # [Decoder Layer 1] - Stage 3 (Low Res, 1/16)
        # 論文: Use ConvSSM
        # Input: Stage 4 (1/32) -> Up -> Concat Stage 3 (1/16)
        self.up1 = SimplePatchExpanding(dims[3], dims[2]) # 768 -> 384
        self.dec_layer1 = nn.Sequential(
            nn.Conv2d(dims[2]*2, dims[2], 1), # 融合 Concat 通道
            ConvSSM(hidden_dim=dims[2]), # [加入] ConvSSM
            PermutedVSSBlock(hidden_dim=dims[2], drop_path=0.1),
            PermutedVSSBlock(hidden_dim=dims[2], drop_path=0.1)
        )
        self.ds1 = nn.Conv2d(dims[2], num_classes, 1) # Deep Sup Head (1/16)

        # [Decoder Layer 2] - Stage 2 (Mid Res, 1/8)
        # 論文: Use ConvSSM
        # Input: Layer 1 (1/16) -> Up -> Concat Stage 2 (1/8)
        self.up2 = SimplePatchExpanding(dims[2], dims[1]) # 384 -> 192
        self.dec_layer2 = nn.ModuleList([
            nn.Conv2d(dims[1]*2, dims[1], 1), # 融合
            # 結構: ConvSSM -> VSS -> VSS
            ConvSSM(hidden_dim=dims[1]), # [保留] ConvSSM
            PermutedVSSBlock(hidden_dim=dims[1], drop_path=0.1),
            PermutedVSSBlock(hidden_dim=dims[1], drop_path=0.1)
        ])
        self.ds2 = nn.Conv2d(dims[1], num_classes, 1) # Deep Sup Head (1/8)

        # [Decoder Layer 3] - Stage 1 (High Res, 1/4)
        # 論文: No ConvSSM
        # Input: Layer 2 (1/8) -> Up -> Concat Stage 1 (1/4)
        self.up3 = SimplePatchExpanding(dims[1], dims[0]) # 192 -> 96
        self.dec_layer3 = nn.ModuleList([
            nn.Conv2d(dims[0]*2, dims[0], 1), # 融合
            # 結構: VSS -> VSS (移除 ConvSSM)
            PermutedVSSBlock(hidden_dim=dims[0], drop_path=0.1),
            PermutedVSSBlock(hidden_dim=dims[0], drop_path=0.1)
        ])
        self.ds3 = nn.Conv2d(dims[0], num_classes, 1) # Deep Sup Head (1/4)

        # --- 3. Final Output ---
        # Decoder Layer 3 (1/4) -> Patch Expanding -> 1x1 Conv -> Output (1/1)
        # 這裡的 scale_factor=4 將 1/4 解析度直接還原回原圖大小
        self.final_head = FinalPatchExpanding(dims[0], num_classes, scale_factor=4)

    def _load_pretrained_encoder(self):
        import os
        # 預設路徑
        ckpt_path = "/home/yuan/Yuan/OIL_Project_12_7/pretrain/vmamba_tiny_e292.pth"
        
        if not os.path.exists(ckpt_path):
            print(f"[Warning] Pretrained weights not found at {ckpt_path}. Training from scratch.")
            return

        print(f"[Info] Loading VMamba pretrained weights from: {ckpt_path}")
        try:
            # 載入權重
            ckpt = torch.load(ckpt_path, map_location='cpu')
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            
            # 準備 Encoder 的 state_dict
            encoder_dict = self.encoder.state_dict()
            
            # 過濾與匹配權重
            # VMamba 官方權重通常以 'layers.0.blocks.0...' 開頭
            # 我們的 VSSMEncoder 結構也是類似的，但需要小心匹配
            
            loaded_layers = 0
            for k, v in state_dict.items():
                # 處理可能的 prefix 差異
                # 官方權重可能包含 'backbone.' 前綴，或者直接是層名稱
                # 這裡假設權重是直接對應 VSSMEncoder 的結構
                
                # 跳過分類頭 (head) 和第一層 PatchEmbed (因為 in_chans 可能不同)
                if "head" in k: continue
                if "patch_embed" in k and self.encoder.patch_embed.proj.in_channels != 3:
                    # 如果輸入通道不是 3，則跳過 patch_embed 權重
                    continue
                
                if k in encoder_dict:
                    if v.shape == encoder_dict[k].shape:
                        encoder_dict[k] = v
                        loaded_layers += 1
                    else:
                        print(f"  [Skip] Shape mismatch for {k}: {v.shape} vs {encoder_dict[k].shape}")
            
            self.encoder.load_state_dict(encoder_dict, strict=False)
            print(f"[Info] Successfully loaded {loaded_layers} layers into VSSMEncoder.")
            
        except Exception as e:
            print(f"[Error] Failed to load pretrained weights: {e}")

    def forward(self, x):
        # x: [B, C, H, W]
        
        # --- Encoder ---
        # VSSMEncoder 回傳: [x, stage1, stage2, stage3, stage4]
        enc_feats = self.encoder(x)
        # c1: 1/4, c2: 1/8, c3: 1/16, c4: 1/32
        c1, c2, c3, c4 = enc_feats[1], enc_feats[2], enc_feats[3], enc_feats[4]

        # --- Decoder ---
        
        # Layer 1 (1/16)
        x = self.up1(c4)
        x = torch.cat([x, c3], dim=1)
        x = self.dec_layer1(x)
        out_16 = self.ds1(x) # Aux 1
        
        # Layer 2 (1/8)
        x = self.up2(x)
        x = torch.cat([x, c2], dim=1)
        # ModuleList 手動執行
        for layer in self.dec_layer2:
            x = layer(x)
        out_8 = self.ds2(x) # Aux 2

        # Layer 3 (1/4)
        x = self.up3(x)
        x = torch.cat([x, c1], dim=1)
        for layer in self.dec_layer3:
            x = layer(x)
        out_4 = self.ds3(x) # Aux 3

        # Final Output (1/1)
        out_final = self.final_head(x)

        if self.training and self.deep_supervision:
            return {
                "main": out_final,
                "aux_4": out_4,   # 1/4 scale
                "aux_8": out_8,   # 1/8 scale
                "aux_16": out_16  # 1/16 scale
            }
        else:
            return out_final
"""
FA-MobileUNet (Full-Scale Aggregated MobileUNet) — PyTorch 復刻
論文: Chen Y.-T., Chang L., Wang J.-H.,
      "Full-Scale Aggregated MobileUNet: An Improved U-Net Architecture for
       SAR Oil Spill Detection." Sensors 2024, 24, 3724.

依論文 Figure 3 / 5 / 6 / 7 / 8 重建。原文未公開程式碼與部分通道寬度，下列標 [假設]
之處為依「圖中標註的尺度」與常見實作的合理重建：

  Encoder = MobileNetv3-Large (ImageNet 預訓練) 為 backbone。
    Figure 3 的五個尺度 352/176/88/44/22 對應(輸入352時, 已實測 torchvision 區塊)：
      S1 @352  : full-res stem 提供   [假設] MobileNetv3 首層即 stride2, 無原生 352 特徵
      S2 @176  : features[0:2]   (16ch,  stride2)
      S3 @88   : features[2:4]   (24ch,  stride4)
      S4 @44   : features[4:7]   (40ch,  stride8)
      Bottleneck @22 : features[7:13] (112ch, stride16)   ← 對齊圖中 22×22, 不用 stride32(11×11)
  CBAM  : 加在 S1~S4 (Figure 3 與 Figure 6)。
  ASPP  : 加在 bottleneck (Figure 7), dilation = 1,3,6,9 + image-level pooling。
  FA(全尺度聚合, Figure 8): 每個 decoder stage 把 5 個 encoder 尺度 resample 到該尺度,
        各過 Conv2D(64,3×3)+BN+ReLU, concat(5×64=320), 再 Conv2D(320,3×3)+BN+ReLU。
        decoder 採序列式 (UNet3+ 風格): 較淺 stage 額外納入較深 decoder 的上採樣 [假設,
        Figure 8 只畫最深 stage4 → 僅 encoder 輸入, 與此一致]。
  Head  : 1×1 conv → num_classes, 上採樣回輸入解析度, 輸出 raw logits (softmax 交給 loss)。

通道寬度 (stem_ch / aspp_ch / dec_ch) 皆可調; 論文宣稱 14.9M 參數, 本重建因通道寬度
未知可能略有出入(以 forward 實測 param 量為準), 可調 dec_ch/aspp_ch 逼近。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1, d=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ----------------------------------------------------------------------------
# CBAM (Figure 6): Channel Attention + Spatial Attention
# ----------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, ch, ratio=16):
        super().__init__()
        hidden = max(ch // ratio, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch, 1, bias=True),
        )

    def forward(self, x):
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        a = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * a


class CBAM(nn.Module):
    def __init__(self, ch, ratio=16, k=7):
        super().__init__()
        self.ca = ChannelAttention(ch, ratio)
        self.sa = SpatialAttention(k)

    def forward(self, x):
        return self.sa(self.ca(x))


# ----------------------------------------------------------------------------
# ASPP (Figure 7): dilation 1,3,6,9 + image pooling
# ----------------------------------------------------------------------------
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, rates=(1, 3, 6, 9)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.branches.append(conv_bn_relu(in_ch, out_ch, k=1, p=0))
            else:
                self.branches.append(conv_bn_relu(in_ch, out_ch, k=3, p=r, d=r))
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv_bn_relu(in_ch, out_ch, k=1, p=0),
        )
        self.project = conv_bn_relu(out_ch * (len(rates) + 1), out_ch, k=1, p=0)

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        gp = self.image_pool(x)
        gp = F.interpolate(gp, size=(h, w), mode='bilinear', align_corners=False)
        feats.append(gp)
        return self.project(torch.cat(feats, dim=1))


# ----------------------------------------------------------------------------
# Full-Scale Aggregation stage (Figure 8)
# ----------------------------------------------------------------------------
def _resample(x, size):
    """把特徵圖縮放到目標空間尺寸: 比目標大→adaptive maxpool; 比目標小→bilinear up; 相同→原樣。"""
    if x.shape[-2:] == size:
        return x
    if x.shape[-2] > size[0]:
        return F.adaptive_max_pool2d(x, size)        # Figure 8 用 Maxpooling 下採樣
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)  # 上採樣


class FAStage(nn.Module):
    """某個 decoder stage 的全尺度聚合 (Figure 8)。
    把所有 encoder 尺度(+可選的較深 decoder)resample 到本尺度, 各→64ch, concat, 再融合。"""
    def __init__(self, enc_channels, deeper_ch=None, cat_out=320, inter=64):
        super().__init__()
        self.skips = nn.ModuleList([conv_bn_relu(c, inter) for c in enc_channels])
        self.has_deeper = deeper_ch is not None
        if self.has_deeper:
            self.deeper = conv_bn_relu(deeper_ch, inter)
        n = len(enc_channels) + (1 if self.has_deeper else 0)
        self.fuse = conv_bn_relu(inter * n, cat_out)   # Conv2D(320,3×3)+BN+ReLU
        self.out_ch = cat_out

    def forward(self, enc_feats, size, deeper=None):
        outs = [self.skips[i](_resample(enc_feats[i], size)) for i in range(len(enc_feats))]
        if self.has_deeper and deeper is not None:
            outs.append(self.deeper(_resample(deeper, size)))
        return self.fuse(torch.cat(outs, dim=1))


# ----------------------------------------------------------------------------
# FA-MobileUNet
# ----------------------------------------------------------------------------
class FAMobileUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, pretrained=True,
                 stem_ch=32, aspp_ch=256, dec_ch=320, cbam_ratio=16):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # ---- Backbone: MobileNetv3-Large ----
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        feats = mobilenet_v3_large(weights=weights).features   # blocks [0..16]

        # 調整首層卷積以接受非 3 通道輸入 (灰階1ch時取 RGB 權重平均, 保留預訓練)
        if in_channels != 3:
            old = feats[0][0]
            new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=(old.bias is not None))
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            feats[0][0] = new

        # 依實測 stride 邊界切片 (input 352 → 176/88/44/22)
        self.enc_s2 = feats[0:2]     # 176, 16ch  (stride2)
        self.enc_s3 = feats[2:4]     # 88,  24ch  (stride4)
        self.enc_s4 = feats[4:7]     # 44,  40ch  (stride8)
        self.enc_s5 = feats[7:13]    # 22, 112ch  (stride16)  ← bottleneck (丟棄 stride32 的 [13:])
        c2, c3, c4, c5 = 16, 24, 40, 112

        # ---- Stem: full-res S1 ----
        self.stem = nn.Sequential(conv_bn_relu(in_channels, stem_ch),
                                  conv_bn_relu(stem_ch, stem_ch))
        c1 = stem_ch

        # ---- CBAM on S1..S4 ----
        self.cbam1 = CBAM(c1, cbam_ratio)
        self.cbam2 = CBAM(c2, cbam_ratio)
        self.cbam3 = CBAM(c3, cbam_ratio)
        self.cbam4 = CBAM(c4, cbam_ratio)

        # ---- ASPP on bottleneck ----
        self.aspp = ASPP(c5, aspp_ch)
        cb = aspp_ch

        enc_channels = [c1, c2, c3, c4, cb]   # 5 個 encoder 尺度餵給 FA

        # ---- FA decoder (序列式, UNet3+ 風格) ----
        # decoder 解析度: stage4@H/8, stage3@H/4, stage2@H/2, stage1@H(input)
        self.dec4 = FAStage(enc_channels, deeper_ch=None,   cat_out=dec_ch)   # 最深, 僅 encoder
        self.dec3 = FAStage(enc_channels, deeper_ch=dec_ch, cat_out=dec_ch)
        self.dec2 = FAStage(enc_channels, deeper_ch=dec_ch, cat_out=dec_ch)
        self.dec1 = FAStage(enc_channels, deeper_ch=dec_ch, cat_out=dec_ch)

        # ---- Head ----
        self.head = nn.Conv2d(dec_ch, num_classes, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        # encoder
        s1 = self.stem(x)        # @H
        s2 = self.enc_s2(x)      # @H/2
        s3 = self.enc_s3(s2)     # @H/4
        s4 = self.enc_s4(s3)     # @H/8
        s5 = self.enc_s5(s4)     # @H/16
        # attention / aspp
        e1 = self.cbam1(s1)
        e2 = self.cbam2(s2)
        e3 = self.cbam3(s3)
        e4 = self.cbam4(s4)
        e5 = self.aspp(s5)
        enc = [e1, e2, e3, e4, e5]
        # decoder (build up)
        d4 = self.dec4(enc, e4.shape[-2:])
        d3 = self.dec3(enc, e3.shape[-2:], deeper=d4)
        d2 = self.dec2(enc, e2.shape[-2:], deeper=d3)
        d1 = self.dec1(enc, e1.shape[-2:], deeper=d2)
        out = self.head(d1)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out

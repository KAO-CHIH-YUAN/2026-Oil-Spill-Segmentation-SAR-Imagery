"""
Official OSDMamba architecture: SwinUMambaD
Source: https://github.com/Multimodal-Intelligence-Lab-MIL/OSDMamba

Classes kept verbatim from the official repo:
  PatchEmbed2D, PatchExpand, FinalPatchExpand_X4, PatchMerging2D,
  SS2D, VSSBlock, VSSLayer, VSSMEncoder, UNetResDecoder

Added by this project:
  OfficialOSDMamba  – thin wrapper that returns a dict compatible with our adapters
  load_pretrained_ckpt – adapted key mapping for our encoder.* namespace
"""
import re
import math
from functools import partial
from typing import Union, List, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


# ---------------------------------------------------------------------------
# Patch utilities
# ---------------------------------------------------------------------------

class PatchEmbed2D(nn.Module):
    """Image to Patch Embedding (4×4 conv)."""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96,
                 norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)   # B C H W -> B H W C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchExpand(nn.Module):
    """2× spatial upsampling via linear + rearrange (BHWC in/out)."""
    def __init__(self, input_resolution, dim, dim_scale=2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = (nn.Linear(dim, 2 * dim, bias=False)
                       if dim_scale == 2 else nn.Identity())
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):           # x: B C H W
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c',
                       p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        x = x.reshape(B, H * 2, W * 2, C // 4)
        return x                    # B H*2 W*2 C/2  (BHWC)


class FinalPatchExpand_X4(nn.Module):
    """4× spatial upsampling for the final decoder stage (BHWC in/out)."""
    def __init__(self, input_resolution, dim, dim_scale=4,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):           # x: B C H W
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c',
                       p1=self.dim_scale, p2=self.dim_scale,
                       c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale,
                       self.output_dim)
        return x                    # B H*4 W*4 C  (BHWC)


class PatchMerging2D(nn.Module):
    """2× spatial downsampling (neighbour concat + linear)."""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):           # x: B H W C
        B, H, W, C = x.shape
        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, H // 2, W // 2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x                    # B H/2 W/2 2C  (BHWC)


# ---------------------------------------------------------------------------
# Core SSM block
# ---------------------------------------------------------------------------

class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2,
                 dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
                 dropout=0., conv_bias=True, bias=False,
                 device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_conv   = d_conv
        self.expand   = expand
        self.d_inner  = int(self.expand * self.d_model)
        self.dt_rank  = (math.ceil(self.d_model / 16)
                         if dt_rank == "auto" else dt_rank)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2,
                                  bias=bias, **factory_kwargs)
        self.conv2d  = nn.Conv2d(self.d_inner, self.d_inner,
                                  groups=self.d_inner, bias=conv_bias,
                                  kernel_size=d_conv,
                                  padding=(d_conv - 1) // 2, **factory_kwargs)
        self.act = nn.SiLU()

        x_proj_tuple = tuple(
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2,
                      bias=False, **factory_kwargs)
            for _ in range(4)
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in x_proj_tuple], dim=0))

        dt_projs = tuple(
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init,
                         dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(4)
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in dt_projs], dim=0))

        self.A_logs = self.A_log_init(self.d_state, self.d_inner,
                                       copies=4, merge=True)
        self.Ds     = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model,
                                   bias=bias, **factory_kwargs)
        self.dropout  = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32,
                                 device=device),
                   "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([
            x.view(B, -1, L),
            torch.transpose(x, 2, 3).contiguous().view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l",
                              xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl,
                                   [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l",
                            dts.view(B, K, -1, L), self.dt_projs_weight)

        xs  = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs  = Bs.float().view(B, K, -1, L)
        Cs  = Cs.float().view(B, K, -1, L)
        Ds  = self.Ds.float().view(-1)
        As  = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias, delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y   = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y    = torch.transpose(out_y[:, 1].view(B, -1, W, H), 2, 3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), 2, 3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, 1, 2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ---------------------------------------------------------------------------
# VSS Block / Layer
# ---------------------------------------------------------------------------

class VSSBlock(nn.Module):
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0,
                 norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0, d_state: int = 16, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate,
                                    d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        return input + self.drop_path(self.self_attention(self.ln_1(input)))


class VSSLayer(nn.Module):
    def __init__(self, dim, depth, attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, d_state=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=dim,
                     drop_path=(drop_path[i] if isinstance(drop_path, list)
                                else drop_path),
                     norm_layer=norm_layer,
                     attn_drop_rate=attn_drop,
                     d_state=d_state)
            for i in range(depth)
        ])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class VSSMEncoder(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, depths=[2, 2, 9, 2],
                 dims=[96, 192, 384, 768], d_state=16,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i) for i in range(self.num_layers)]
        self.embed_dim    = dims[0]
        self.num_features = dims[-1]
        self.dims         = dims

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        self.ape = False
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers     = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer], depth=depths[i_layer],
                d_state=(math.ceil(dims[0] / 6) if d_state is None else d_state),
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer, downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsamples.append(
                    PatchMerging2D(dim=dims[i_layer], norm_layer=norm_layer))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x_ret = [x]                        # x_ret[0] = original image
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for s, layer in enumerate(self.layers):
            x = layer(x)
            x_ret.append(x.permute(0, 3, 1, 2))   # BHWC -> BCHW
            if s < len(self.downsamples):
                x = self.downsamples[s](x)
        return x_ret   # [original, stage1, stage2, stage3, stage4]


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class UNetResDecoder(nn.Module):
    """
    U-Net style decoder with VSSLayer refinement at each scale.
    With deep_supervision=True returns list [main, aux_H4, aux_H8, aux_H16]
    (after internal reversal: largest prediction is always [0]).
    """
    def __init__(self, num_classes: int, deep_supervision,
                 features_per_stage: Union[Tuple[int, ...], List[int]] = None,
                 drop_path_rate: float = 0.2, d_state: int = 16):
        super().__init__()
        encoder_output_channels = features_per_stage
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        n_stages_encoder = len(encoder_output_channels)   # 4

        dpr    = [x.item() for x in
                  torch.linspace(drop_path_rate, 0, (n_stages_encoder - 1) * 2)]
        depths = [2, 2, 2, 2]

        stages         = []
        expand_layers  = []
        seg_layers     = []
        concat_back_dim = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder_output_channels[-s]
            input_features_skip  = encoder_output_channels[-(s + 1)]
            expand_layers.append(PatchExpand(
                input_resolution=None, dim=input_features_below,
                dim_scale=2, norm_layer=nn.LayerNorm))
            stages.append(VSSLayer(
                dim=input_features_skip, depth=2, attn_drop=0.,
                drop_path=dpr[sum(depths[:s - 1]):sum(depths[:s])],
                d_state=(math.ceil(2 * input_features_skip / 6)
                         if d_state is None else d_state),
                norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False))
            seg_layers.append(
                nn.Conv2d(input_features_skip, num_classes, 1, 1, 0, bias=True))
            concat_back_dim.append(
                nn.Linear(2 * input_features_skip, input_features_skip))

        # final 4× expand
        expand_layers.append(FinalPatchExpand_X4(
            input_resolution=None, dim=encoder_output_channels[0],
            dim_scale=4, norm_layer=nn.LayerNorm))
        stages.append(nn.Identity())
        seg_layers.append(
            nn.Conv2d(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages          = nn.ModuleList(stages)
        self.expand_layers   = nn.ModuleList(expand_layers)
        self.seg_layers      = nn.ModuleList(seg_layers)
        self.concat_back_dim = nn.ModuleList(concat_back_dim)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.expand_layers[s](lres_input)       # BHWC after expand
            if s < (len(self.stages) - 1):
                x = torch.cat((x, skips[-(s + 2)].permute(0, 2, 3, 1)), -1)
                x = self.concat_back_dim[s](x)
            x = self.stages[s](x).permute(0, 3, 1, 2)  # BCHW
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]   # largest first
        return seg_outputs if self.deep_supervision else seg_outputs[0]


# ---------------------------------------------------------------------------
# Combined model with dict output
# ---------------------------------------------------------------------------

class OfficialOSDMamba(nn.Module):
    """
    Wrapper combining VSSMEncoder + UNetResDecoder.

    forward() returns a dict:
      {'main': Tensor[B,C,H,W],        # full-resolution prediction
       'aux_4':  Tensor[B,C,H/4,W/4], # deep-supervision head
       'aux_8':  Tensor[B,C,H/8,W/8],
       'aux_16': Tensor[B,C,H/16,W/16]}

    (aux_* keys are only present when deep_supervision=True)
    """
    def __init__(self,
                 in_chans: int = 1,
                 num_classes: int = 1,
                 depths: list = None,
                 dims: int = 96,
                 drop_path_rate: float = 0.2,
                 d_state: int = 16,
                 deep_supervision: bool = True):
        super().__init__()
        if depths is None:
            depths = [2, 2, 9, 2]

        # compute channel list for decoder
        if isinstance(dims, int):
            features_per_stage = [int(dims * 2 ** i) for i in range(len(depths))]
        else:
            features_per_stage = list(dims)

        self.encoder = VSSMEncoder(
            patch_size=4, in_chans=in_chans, depths=depths,
            dims=dims, d_state=d_state, drop_path_rate=drop_path_rate)
        self.decoder = UNetResDecoder(
            num_classes=num_classes, deep_supervision=deep_supervision,
            features_per_stage=features_per_stage,
            drop_path_rate=drop_path_rate, d_state=d_state)

        self.deep_supervision = deep_supervision

    def forward(self, x):
        skips = self.encoder(x)
        outs  = self.decoder(skips)
        if self.deep_supervision and isinstance(outs, list):
            return {
                'main':   outs[0],
                'aux_4':  outs[1],
                'aux_8':  outs[2],
                'aux_16': outs[3],
            }
        return {'main': outs if not isinstance(outs, list) else outs[0]}

    def freeze_encoder(self):
        for name, param in self.encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Weight loading from VMamba-Tiny pretrained checkpoint
# ---------------------------------------------------------------------------

def load_pretrained_ckpt(model: OfficialOSDMamba,
                          num_input_channels: int,
                          ckpt_path: str) -> OfficialOSDMamba:
    """
    Load VMamba-Tiny pretrained weights into model.encoder.
    Handles key renaming (layers.i.downsample -> downsamples.i)
    and skips patch_embed when in_channels doesn't match (e.g. grayscale).
    """
    print(f"  - [Pretrain] Loading VMamba weights from: {ckpt_path}")
    skip_params = {"norm.weight", "norm.bias", "head.weight", "head.bias"}

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    ckpt_state = ckpt.get('model', ckpt)

    model_dict = model.state_dict()
    loaded, skipped = 0, 0

    for k, v in ckpt_state.items():
        if k in skip_params:
            skipped += 1
            continue

        # map checkpoint key -> our model key (under encoder.*)
        kr = f"encoder.{k}"

        # skip patch_embed when channel count differs
        if ("patch_embed" in k and
                ckpt_state.get("patch_embed.proj.weight") is not None and
                ckpt_state["patch_embed.proj.weight"].shape[1]
                != num_input_channels):
            skipped += 1
            continue

        # remap downsample keys
        if "downsample" in kr:
            match = re.findall(r"layers\.(\d+)\.downsample", kr)
            if match:
                i_ds = int(match[0])
                kr = kr.replace(f"layers.{i_ds}.downsample",
                                 f"downsamples.{i_ds}")

        if kr in model_dict and v.shape == model_dict[kr].shape:
            model_dict[kr] = v
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(model_dict)
    print(f"  - [Pretrain] Loaded {loaded} params, skipped {skipped}")
    return model

"""
[File] gradcam_utils.py
[Purpose] Gradient-weighted Class Activation Mapping (Grad-CAM) Utilities.
[Description]
    - Implements GradCAM for visualizing model attention maps.
    - Helps interpret model decisions by highlighting relevant image regions.
    - Supports custom target layers for different architectures.
[Author] Assistant (GitHub Copilot)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 自動尋找最後一層卷積層
        if self.target_layer is None:
            self.target_layer = self._find_last_conv_layer(self.model)

        # 註冊 Hook
        if self.target_layer:
            self.target_layer.register_forward_hook(self.save_activation)
            self.target_layer.register_full_backward_hook(self.save_gradient)

    def _find_last_conv_layer(self, model):
        """遞迴尋找最後一個 nn.Conv2d 層"""
        last_conv = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, class_idx=None):
        """產生 Grad-CAM (高層語意熱力圖)"""
        self.model.eval()
        self.model.zero_grad()
        
        with torch.set_grad_enabled(True):
            output = self.model(input_tensor)
            if hasattr(output, 'logits'): logits = output.logits
            else: logits = output

            if logits.shape[-2:] != input_tensor.shape[-2:]:
                logits = F.interpolate(logits, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)

            # 針對二元分類或多類別選取目標
            if class_idx is None:
                if logits.shape[1] == 1: score = logits.sum()
                else: score = logits[:, 1, :, :].sum() if logits.shape[1] > 1 else logits.sum()
            else:
                score = logits[:, class_idx, :, :].sum()

            score.backward()
            
            if self.gradients is None or self.activations is None:
                return np.zeros(input_tensor.shape[-2:], dtype=np.float32)

            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            activation = self.activations.detach()
            
            for i in range(activation.shape[1]):
                activation[:, i, :, :] *= pooled_gradients[i]
                
            heatmap = torch.mean(activation, dim=1)
            if heatmap.ndim == 3:
                heatmap = heatmap[0] # [B, H, W] -> [H, W] assuming B=1
            
            heatmap = F.relu(heatmap)
            heatmap = heatmap.cpu().numpy()
            
            # Robust Normalization (99% percentile to avoid outliers)
            v_max = np.percentile(heatmap, 99)
            if v_max > 1e-7: heatmap /= v_max
            heatmap = np.clip(heatmap, 0, 1)
            
            return heatmap

def compute_input_saliency(model, input_tensor):
    """
    [增強版] 計算 Input Saliency Map。
    使用 (Gradient * Input) + GaussianBlur 來產生平滑、易讀的通道貢獻圖。
    """
    model.eval()
    model.zero_grad()
    
    if not input_tensor.requires_grad:
        input_tensor.requires_grad_()
        
    with torch.set_grad_enabled(True):
        output = model(input_tensor)
        if hasattr(output, 'logits'): logits = output.logits
        else: logits = output
        
        if logits.shape[1] == 1: score = logits.sum()
        else: score = logits[:, 1, :, :].sum() if logits.shape[1] > 1 else logits.sum()
        
    score.backward()
    
    # 1. 取得梯度 (Abs)
    grads = input_tensor.grad.abs().cpu().detach().numpy()
    if grads.ndim == 4:
        grads = grads[0] # [1, C, H, W] -> [C, H, W]
    
    # 2. 取得輸入影像 (Intensity)
    inputs = input_tensor.cpu().detach().numpy()
    if inputs.ndim == 4:
        inputs = inputs[0] # [1, C, H, W] -> [C, H, W]
    
    inputs = np.abs(inputs)

    if grads.ndim == 2:
        grads = np.expand_dims(grads, axis=0)
        inputs = np.expand_dims(inputs, axis=0)
        
    saliency_maps = []
    
    # 針對每個通道計算 (先不正規化，保留原始數值大小)
    for c in range(grads.shape[0]):
        # A. Gradient * Input
        g = grads[c] * inputs[c]
        
        # B. Gaussian Blur
        g = cv2.GaussianBlur(g, (0, 0), sigmaX=3)
        
        saliency_maps.append(g)
        
    saliency_maps = np.array(saliency_maps)
    
    # C. Global Robust Normalization (全通道統一正規化，保留相對強弱)
    # 使用所有通道中的 99.5% 分位數作為最大值
    v_max = np.percentile(saliency_maps, 99.5)
    v_min = np.min(saliency_maps)
    
    if v_max - v_min > 1e-9:
        saliency_maps = (saliency_maps - v_min) / (v_max - v_min)
    else:
        saliency_maps = np.zeros_like(saliency_maps)
        
    saliency_maps = np.clip(saliency_maps, 0, 1)
        
    return saliency_maps

def apply_colormap_on_image(org_img, activation_map, colormap=cv2.COLORMAP_JET):
    """
    [補回] 將熱力圖疊加在原圖上 (Evaluation Module 需要此函式)
    org_img: (H, W, 3) BGR
    activation_map: (H, W) float 0-1
    """
    # 確保 activation_map 是 uint8 0-255
    heatmap = (activation_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # 確保原圖是 uint8
    if org_img.max() <= 1.0:
        org_img = (org_img * 255).astype(np.uint8)
        
    # Resize heatmap to match image if necessary
    if heatmap.shape[:2] != org_img.shape[:2]:
        heatmap = cv2.resize(heatmap, (org_img.shape[1], org_img.shape[0]))
        
    # 疊加 (50% 原圖 + 50% 熱力圖)
    superimposed_img = cv2.addWeighted(heatmap, 0.5, org_img, 0.5, 0)
    return superimposed_img
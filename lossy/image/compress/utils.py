import shutil
from typing import Dict

import imageio.v3 as iio
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import torch
import numpy as np
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

def _check_tool(cmd: str):
    if shutil.which(cmd) is None:
        raise RuntimeError(f'"{cmd}" not found on PATH. Please install it first.')

def _human(n: int) -> str:
    for u in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024 or u == 'TB':
            return f'{n:.2f} {u}' if u != 'B' else f'{n} {u}'
        n /= 1024
    return f'{n:.2f} TB'

def calc_metrics_psnr_ssim(original_path: str, decoded_path: str) -> Dict[str, float]:
    img1 = iio.imread(original_path)
    img2 = iio.imread(decoded_path)
    return {
        'psnr': psnr(img1, img2, data_range=255),
        'ssim': ssim(img1, img2, channel_axis=-1, data_range=255)
    }
    
def calc_metrics_psnr_msssim(original_path: str, decoded_path: str) -> Dict[str, float]:
    img1 = iio.imread(original_path).astype(np.float32) / 255.0
    img2 = iio.imread(decoded_path).astype(np.float32) / 255.0

    # torch tensor: [B, C, H, W]
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)

    # -------- Padding for ms-ssim --------
    win_size = 11
    levels = 5
    required_hw = (win_size - 1) * (2 ** (levels - 1)) + 4 # = 160

    B, C, H, W = img1_tensor.shape
    pad_h = max(0, required_hw - H)
    pad_w = max(0, required_hw - W)

    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Padding: left, right, top, bottom
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        img1_tensor = F.pad(img1_tensor, padding, mode='reflect')
        img2_tensor = F.pad(img2_tensor, padding, mode='reflect')

    # -------------------------------------

    msssim_val = ms_ssim(img1_tensor, img2_tensor, data_range=1.0).item()
    return {
        'psnr': psnr(img1 * 255.0, img2 * 255.0, data_range=255),
        'ms-ssim': msssim_val
    }
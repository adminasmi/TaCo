import os
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

def read_yuv420_frame(file, width, height, bitdepth=8):
    """读取一帧 YUV420 -> (H, W, 3)，支持 8/10-bit"""
    dtype = np.uint16 if bitdepth > 8 else np.uint8
    bytes_per_sample = 2 if bitdepth > 8 else 1
    max_val = (1 << bitdepth) - 1  # 255 for 8-bit, 1023 for 10-bit

    y_size = width * height
    uv_size = y_size // 4

    y = np.frombuffer(file.read(y_size * bytes_per_sample), dtype=dtype)
    u = np.frombuffer(file.read(uv_size * bytes_per_sample), dtype=dtype)
    v = np.frombuffer(file.read(uv_size * bytes_per_sample), dtype=dtype)

    if y.size < y_size or u.size < uv_size or v.size < uv_size:
        return None

    y = y.reshape((height, width))
    u = u.reshape((height // 2, width // 2))
    v = v.reshape((height // 2, width // 2))

    # 上采样 U/V
    u_up = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
    v_up = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)

    return np.stack([y, u_up, v_up], axis=2).astype(np.float32) / max_val


def calc_total_frames(yuv_path, width, height, bitdepth=8):
    """根据文件大小计算帧数"""
    bytes_per_sample = 2 if bitdepth > 8 else 1
    frame_size = width * height * 3 // 2 * bytes_per_sample
    file_size = os.path.getsize(yuv_path)
    return file_size // frame_size


def pad_if_needed(x, min_size=162):
    """如果图像宽高 < min_size，则自动补零 padding 到 >= min_size"""
    _, _, h, w = x.shape
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    if pad_h > 0 or pad_w > 0:
        # (left, right, top, bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return x


def calc_msssim_yuv(orig_path, rec_path, width, height, bitdepth=8, batch_size=100, device="cuda"):
    """计算 YUV420p (8-bit 或 10-bit) 的平均 MS-SSIM (自动 padding 小分辨率)"""
    total_frames = min(
        calc_total_frames(orig_path, width, height, bitdepth=8),   # 原始用 8-bit
        calc_total_frames(rec_path, width, height, bitdepth=bitdepth),  # 重建可 8/10-bit
    )
    # print(f"Detected {total_frames} frames at bitdepth={bitdepth}")

    f1 = open(orig_path, "rb")
    f2 = open(rec_path, "rb")

    scores = []
    batch1, batch2 = [], []

    for i in range(total_frames):
        frame1 = read_yuv420_frame(f1, width, height, bitdepth=8)
        frame2 = read_yuv420_frame(f2, width, height, bitdepth=bitdepth)
        if frame1 is None or frame2 is None:
            break

        max_val = (1 << bitdepth) - 1  # 255 for 8-bit, 1023 for 10-bit
        frame1 = torch.from_numpy(frame1).permute(2,0,1).unsqueeze(0).float() / 255.0
        frame2 = torch.from_numpy(frame2).permute(2,0,1).unsqueeze(0).float() / max_val

        frame1 = pad_if_needed(frame1)
        frame2 = pad_if_needed(frame2)

        batch1.append(frame1.squeeze(0))
        batch2.append(frame2.squeeze(0))

        if len(batch1) == batch_size or i == total_frames - 1:
            b1 = torch.stack(batch1).to(device)  # 用 stack
            b2 = torch.stack(batch2).to(device)

            with torch.no_grad():
                score = ms_ssim(b1, b2, data_range=1.0, size_average=False)  # 每帧独立算
            scores.extend(score.cpu().numpy().tolist())

            batch1, batch2 = [], []

    f1.close()
    f2.close()
    return float(np.mean(scores))


# 示例调用
if __name__ == '__main__':
    rec_path  = '/data/ssd/zhaoy/datasets/ObjectFolder_1.0/compressed/vvenc/rec/10_qp17_medium.yuv'
    orig_path = '/data/ssd/zhaoy/datasets/ObjectFolder_1.0/dataset-comp/test/video/yuv/10.yuv'
    print(calc_msssim_yuv(orig_path, rec_path, width=120, height=160, bitdepth=10))
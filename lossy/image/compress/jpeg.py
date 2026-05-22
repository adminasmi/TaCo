import os
import time
import csv
import shutil
import subprocess
from typing import List, Dict, Optional

from utils import _check_tool, _human, calc_metrics_psnr_msssim


# ---------------- JPEG 2000 ----------------

def compress_jpeg2000(input_path: str, output_path: str, qp: int) -> Dict:
    _check_tool('opj_compress')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = ['opj_compress', '-i', input_path, '-o', output_path, '-q', str(qp)]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    t1 = time.perf_counter()
    return {
        'qp': qp,
        'input': input_path,
        'output': output_path,
        'input_size': os.path.getsize(input_path),
        'output_size': os.path.getsize(output_path),
        'time_s': t1 - t0,
    }

def decompress_jpeg2000(input_path: str, output_path: str) -> bool:
    _check_tool('opj_decompress')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = ['opj_decompress', '-i', input_path, '-o', output_path]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f'Error decompressing {input_path}: {e}')
        return False

def benchmark_jpeg2000_multi_qp(input_path: str, out_dir: str, qps: List[int], csv_path: Optional[str] = None) -> List[Dict]:
    base = os.path.splitext(os.path.basename(input_path))[0]
    results = []

    for qp in qps:
        out_j2k = os.path.join(out_dir, f'{base}_qp{qp}.j2k')
        out_dec = os.path.join(out_dir, f'{base}_qp{qp}_dec.png')

        try:
            res = compress_jpeg2000(input_path, out_j2k, qp)
            if decompress_jpeg2000(out_j2k, out_dec):
                quality = calc_metrics_psnr_msssim(input_path, out_dec)
                res.update(quality)
                res['ratio'] = res['output_size'] / res['input_size']
                res['reduction_pct'] = 100 * (1 - res['ratio'])
                results.append(res)
                print(
                    f'JPEG2000 | QP={qp:2d} | PSNR={quality["psnr"]:.2f} | SSIM={quality["ms-ssim"]:.4f} | '
                    f'size={_human(res["output_size"])} | time={res["time_s"]:.4f}s'
                )
        except Exception as e:
            print(f'[QP={qp}] failed: {e}')

    if csv_path:
        fieldnames = ['qp', 'input', 'output', 'input_size', 'output_size', 'ratio', 'reduction_pct', 'psnr', 'ms-ssim', 'time_s']
        with open(csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerows(results)
    return results

# ---------------- JPEG XL ----------------

def compress_jpegxl(input_path: str, output_path: str, qp: int) -> Dict:
    _check_tool('cjxl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = ['cjxl', input_path, output_path, '-d', str(qp)]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    t1 = time.perf_counter()
    return {
        'qp': qp,
        'input': input_path,
        'output': output_path,
        'input_size': os.path.getsize(input_path),
        'output_size': os.path.getsize(output_path),
        'time_s': t1 - t0,
    }

def decompress_jpegxl(input_path: str, output_path: str) -> bool:
    _check_tool('djxl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = ['djxl', input_path, output_path]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f'Error decompressing {input_path}: {e}')
        return False

def benchmark_jpegxl_multi_qp(input_path: str, out_dir: str, qps: List[int], csv_path: Optional[str] = None) -> List[Dict]:
    base = os.path.splitext(os.path.basename(input_path))[0]
    results = []

    for qp in qps:
        out_jxl = os.path.join(out_dir, f'{base}_qp{qp}.jxl')
        out_dec = os.path.join(out_dir, f'{base}_qp{qp}_dec.png')

        try:
            res = compress_jpegxl(input_path, out_jxl, qp)
            if decompress_jpegxl(out_jxl, out_dec):
                quality = calc_metrics_psnr_msssim(input_path, out_dec)
                res.update(quality)
                res['ratio'] = res['output_size'] / res['input_size']
                res['reduction_pct'] = 100 * (1 - res['ratio'])
                results.append(res)
                print(
                    f'JPEG-XL  | QP={qp:2d} | PSNR={quality["psnr"]:.2f} | SSIM={quality["ms-ssim"]:.4f} | '
                    f'size={_human(res["output_size"])} | time={res["time_s"]:.4f}s'
                )
        except Exception as e:
            print(f'[QP={qp}] failed: {e}')

    if csv_path:
        fieldnames = ['qp', 'input', 'output', 'input_size', 'output_size', 'ratio', 'reduction_pct', 'psnr', 'ms-ssim', 'time_s']
        with open(csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerows(results)
    return results

# ---------------- Entry ----------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--codec', type=str, choices=['jpeg2000', 'jpegxl'], default='jpegxl')
    parser.add_argument('--dataset', default='SSVTP')
    args = parser.parse_args()

    dataset = args.dataset
    codec = args.codec

    # in_dir  = f'/data/ssd/zhaoy/datasets/{dataset}/dataset-comp/test/image'
    in_dir  = '/data/ssd/zhaoy/datasets/ActiveCloth/frames'
    out_dir = f'/data/ssd/zhaoy/datasets/{dataset}/compressed/{codec}/lossy'
    stat_dir = '/home/zhaoy/TaCo-Bench/lossy/image/statistics'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(stat_dir, exist_ok=True)

    # qps = [37, 41, 45, 49, 53, 57, 59] if codec == 'jpeg2000' else [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    if codec == 'jpeg2000':
        if 'objectfolder' in dataset.lower():
            qps = [28, 30]
            # qps = [27, 29, 31, 33, 35]
        else:
            qps = [32, 35, 38, 41, 44, 47]
    else:
        if 'objectfolder' in dataset.lower():
            qps = [2.0]
            # qps = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        else:
            qps = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    all_results = []

    from pathlib import Path
    from glob import glob
    in_dir = Path(in_dir)
    in_files = list(in_dir.rglob('*.png'))
    for file in in_files:
        in_path = os.path.join(in_dir, file)
        if codec == 'jpeg2000':
            results = benchmark_jpeg2000_multi_qp(
                input_path=in_path,
                out_dir=out_dir,
                qps=qps,
                csv_path=None
            )
        else:
            results = benchmark_jpegxl_multi_qp(
                input_path=in_path,
                out_dir=out_dir,
                qps=qps,
                csv_path=None
            )
        all_results.extend(results)

    stat_path = os.path.join(stat_dir, f'{codec}_{dataset}.csv')
    with open(stat_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_results[0].keys())
        w.writeheader()
        w.writerows(all_results)
    print(f'Saved CSV: {stat_path}')

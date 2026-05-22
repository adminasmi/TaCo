import os
import time
import csv
import subprocess
from typing import List, Dict, Optional

from utils import _check_tool, _human, calc_metrics_psnr_msssim


def compress_bpg(
    input_path: str,
    output_path: str,
    qp: int,
    fmt: str = '444',
    bit_depth: int = 8,
    encoder_extras: Optional[List[str]] = None,
) -> Dict:
    _check_tool('bpgenc')
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)

    cmd = ['bpgenc', '-q', str(qp), '-f', fmt, '-b', str(bit_depth), input_path, '-o', output_path]
    if encoder_extras:
        cmd[1:1] = encoder_extras

    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    t1 = time.perf_counter()

    in_size = os.path.getsize(input_path)
    out_size = os.path.getsize(output_path)
    ratio = out_size / in_size if in_size > 0 else float('inf')

    return {
        'qp': qp,
        'fmt': fmt,
        'bit_depth': bit_depth,
        'input': input_path,
        'output': output_path,
        'input_size': in_size,
        'output_size': out_size,
        'ratio': ratio,
        'time_s': (t1 - t0),
    }

def decompress_bpg(input_path: str, output_path: str) -> bool:
    _check_tool('bpgdec')
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    cmd = ['bpgdec', '-o', output_path, input_path]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f'Error decompressing {input_path}: {e}')
        return False

def benchmark_bpg_multi_qp(
    input_path: str,
    out_dir: str,
    qps: List[int],
    fmt: str = '444',
    bit_depth: int = 8,
    encoder_extras: Optional[List[str]] = None,
    csv_path: Optional[str] = None,
) -> List[Dict]:
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    base = os.path.splitext(os.path.basename(input_path))[0]

    for qp in qps:
        out_bpg = os.path.join(out_dir, f'{base}_qp{qp}.bpg')
        out_dec = os.path.join(out_dir, f'{base}_qp{qp}_dec.png')

        try:
            res = compress_bpg(input_path, out_bpg, qp, fmt, bit_depth, encoder_extras)
            if decompress_bpg(out_bpg, out_dec):
                quality = calc_metrics_psnr_msssim(input_path, out_dec)
                res.update(quality)
                res['reduction_pct'] = 100 * (1 - res['ratio'])
                results.append(res)
                print(
                    f'QP={qp:2d} | PSNR={quality["psnr"]:.2f} | SSIM={quality["ms-ssim"]:.4f} | '
                    f'size={_human(res["output_size"])} | ratio={res["ratio"]:.4f} | '
                    f'reduction={res["reduction_pct"]:.2f}% | time={res["time_s"]:.4f}s'
                )
        except Exception as e:
            print(f'[QP={qp}] failed: {e}')

    if csv_path:
        fieldnames = [
            'qp', 'fmt', 'bit_depth', 'input', 'output', 'input_size', 'output_size', 'ratio', 'reduction_pct', 'psnr', 'ms-ssim', 'time_s'
        ]
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f'Saved CSV: {csv_path}')

    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ObjectFolder_1.0')
    args = parser.parse_args()

    dataset = args.dataset
    in_dir  = f'/data/ssd/zhaoy/datasets/{dataset}/dataset-comp/test/image'
    out_dir = f'/data/ssd/zhaoy/datasets/{dataset}/compressed/bpg/lossy'
    stat_dir = '/home/zhaoy/TaCo-Bench/lossy/image/statistics'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(stat_dir, exist_ok=True)

    qps = [12, 18, 24, 30, 36, 42, 48]
    all_results = []

    for file in os.listdir(in_dir):
        in_path = os.path.join(in_dir, file)
        results = benchmark_bpg_multi_qp(
            input_path=in_path,
            out_dir=out_dir,
            qps=qps,
            fmt='444',
            bit_depth=8,
            encoder_extras=None,
            csv_path=None
        )
        all_results.extend(results)

    stat_path = os.path.join(stat_dir, f'bpg_{dataset}.csv')
    fieldnames = [
        'qp', 'fmt', 'bit_depth', 'input', 'output', 'input_size', 'output_size', 'ratio', 'reduction_pct', 'psnr', 'ms-ssim', 'time_s'
    ]
    with open(stat_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_results)
    print(f'saved csv: {stat_path}')

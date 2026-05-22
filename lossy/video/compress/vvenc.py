import os
from tqdm import tqdm

MODE = 'test'
VVENC = '/home/zhaoy/vvenc/bin/release-static/vvencapp'
QPs = list(range(17, 52, 5))
PRESET = 'medium'
DEBUG  = False

import time

def enc_vvenc(size, yuv_dir, rlt_root, vvenc=VVENC, qps=QPs, preset=PRESET, debug=DEBUG):
    yuvs = [_ for _ in os.listdir(yuv_dir) if _.endswith('.yuv')]
    for yuv in tqdm(yuvs):
        yuv_path = os.path.join(yuv_dir, yuv)
        os.makedirs(f'{rlt_root}/log', exist_ok=True)
        os.makedirs(f'{rlt_root}/bin', exist_ok=True)
        
        for qp in qps:
            log_path = os.path.join(rlt_root, 'log', yuv.replace('.yuv', f'_qp{qp}_{PRESET}.log'))
            bin_path = os.path.join(rlt_root, 'bin', yuv.replace('.yuv', f'_qp{qp}_{PRESET}.bin'))
            
            cmd = f'{vvenc} --size {size} --preset {preset} --qp {qp} --fps 30 --format yuv420 --input {yuv_path} --passes 1 --output {bin_path} --threads 8 > {log_path} &'
            os.system(cmd)
            if DEBUG:
                break
        time.sleep(5)
        

''' 1. Touch and Go '''
yuv_root = '/data/ssd/zhaoy/datasets/TouchandGoDataset-v2/dataset-comp'
yuv_dir  = os.path.join(yuv_root, MODE, 'video', 'yuv')
rlt_root = '/data/ssd/zhaoy/datasets/TouchandGoDataset-v2/compressed/vvenc'
enc_vvenc(size='640x480', yuv_dir=yuv_dir, rlt_root=rlt_root)


''' 2. Object Folder '''
yuv_root = '/data/ssd/zhaoy/datasets/ObjectFolder_1.0/dataset-comp'
yuv_dir  = os.path.join(yuv_root, MODE, 'video', 'yuv')
rlt_root = '/data/ssd/zhaoy/datasets/ObjectFolder_1.0/compressed/vvenc'
enc_vvenc(size='160x120', yuv_dir=yuv_dir, rlt_root=rlt_root)


''' 3. SSVTP '''
yuv_root = '/data/ssd/zhaoy/datasets/SSVTP/dataset-comp'
yuv_dir  = os.path.join(yuv_root, MODE, 'video', 'yuv')
rlt_root = '/data/ssd/zhaoy/datasets/SSVTP/compressed/vvenc'
enc_vvenc(size='240x320', yuv_dir=yuv_dir, rlt_root=rlt_root)
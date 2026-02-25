#!/usr/bin/env python3

import subprocess
import sys
import os
import argparse
from datetime import datetime
from tqdm import tqdm

MODELS = [
    'pointnet_sem_seg',
    'pointnet2_sem_seg',
    'pointnet2_sem_seg_msg',
    'pointnet2_sem_seg_trans',
    'pointnet2_sem_seg_trans_msg'
]

def train_model(model_name, test_area=5, epochs=300, batch_size=8):
    
    log_dir = model_name
    
    cmd = [
        'python', 'train_semseg3.py',
        '--model', model_name,
        '--log_dir', log_dir,
        '--test_area', str(test_area),
        '--epoch', str(epochs),
        '--batch_size', str(batch_size)
    ]
    
    print(f"\n{'='*60}")
    print(f"开始训练模型: {model_name}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, bufsize=1)
        
        pbar = tqdm(total=epochs, desc=f"{model_name} 训练进度", unit="epoch")
        current_epoch = 0
        
        for line in process.stdout:
            if "**** Epoch" in line and "/" in line and "****" in line:
                try:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        epoch_num = int(parts[2])
                        if epoch_num > current_epoch:
                            pbar.update(epoch_num - current_epoch)
                            current_epoch = epoch_num
                except:
                    pass
            
            if any(keyword in line for keyword in ["Training mean loss", "Training accuracy", "eval mean loss", "Best mIoU"]):
                tqdm.write(line.strip())
        
        process.wait()
        
        if current_epoch < epochs:
            pbar.update(epochs - current_epoch)
        pbar.close()
        
        if process.returncode == 0:
            print(f"\n模型 {model_name} 训练成功完成!")
            return True
        else:
            print(f"\n模型 {model_name} 训练失败! 返回码: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"模型 {model_name} 训练过程中出现异常: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='连续训练多个模型')
    parser.add_argument('--test_area', type=int, default=5, help='测试区域 (默认: 5)')
    parser.add_argument('--epoch', type=int, default=300, help='训练轮数 (默认: 300)')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小 (默认: 8)')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=MODELS, 
                       help='要训练的模型列表 (默认: {})'.format(' '.join(MODELS)))
    
    args = parser.parse_args()
    
    print(f"开始连续训练任务: {datetime.now()}")
    print(f"训练模型列表: {args.models}")
    
    success_models = []
    failed_models = []
    
    for i, model_name in enumerate(args.models):
        print(f"\n[{i+1}/{len(args.models)}] 训练进度")
        
        if train_model(model_name, args.test_area, args.epoch, args.batch_size):
            success_models.append(model_name)
        else:
            failed_models.append(model_name)
            
        print(f"\n{'#'*80}")
    
    print(f"\n{'='*60}")
    print("训练任务总结:")
    print(f"成功训练的模型 ({len(success_models)}个): {success_models}")
    print(f"训练失败的模型 ({len(failed_models)}个): {failed_models}")
    print(f"完成时间: {datetime.now()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
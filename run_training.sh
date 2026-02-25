#!/bin/bash

# ====================================================================
# 森林数据集训练启动脚本
# ====================================================================

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# ====================================================================
# 配置 1: 基础训练（原始模型）
# ====================================================================
baseline_training() {
    echo "======================================================================"
    echo "配置 1: 基础训练（原始模型）"
    echo "======================================================================"
    
    python train_forest_optimized.py \
        --model pointnet2_sem_seg \
        --batch_size 8 \
        --epoch 100 \
        --learning_rate 0.001 \
        --optimizer Adam \
        --scheduler cosine \
        --npoint 16384 \
        --input_channels 6 \
        --log_dir baseline \
        --gpu 0
}

# ====================================================================
# 配置 2: 优化模型 - 速度优先
# ====================================================================
speed_optimized_training() {
    echo "======================================================================"
    echo "配置 2: 优化模型 - 速度优先"
    echo "======================================================================"
    
    python train_forest_optimized.py \
        --model pointnet2_sem_seg \
        --use_optimized \
        --dropout 0.2 \
        --batch_size 16 \
        --epoch 80 \
        --learning_rate 0.001 \
        --optimizer Adam \
        --scheduler cosine \
        --npoint 16384 \
        --input_channels 6 \
        --log_dir speed_optimized \
        --gpu 0
}

# ====================================================================
# 配置 3: 优化模型 - 平衡配置（推荐）
# ====================================================================
balanced_training() {
    echo "======================================================================"
    echo "配置 3: 优化模型 - 平衡配置（推荐）"
    echo "======================================================================"
    
    python train_forest_optimized.py \
        --model pointnet2_sem_seg \
        --use_optimized \
        --use_attention \
        --dropout 0.3 \
        --batch_size 8 \
        --epoch 100 \
        --learning_rate 0.001 \
        --min_lr 1e-5 \
        --optimizer Adam \
        --decay_rate 1e-4 \
        --scheduler cosine \
        --use_focal \
        --focal_gamma 2.0 \
        --use_class_weights \
        --npoint 16384 \
        --input_channels 6 \
        --augment \
        --early_stop \
        --patience 20 \
        --save_freq 10 \
        --eval_freq 1 \
        --log_dir balanced_optimized \
        --gpu 0
}

# ====================================================================
# 配置 4: 优化模型 - 精度优先（完整优化）
# ====================================================================
accuracy_optimized_training() {
    echo "======================================================================"
    echo "配置 4: 优化模型 - 精度优先（完整优化）"
    echo "======================================================================"
    
    python train_forest_optimized.py \
        --model pointnet2_sem_seg \
        --use_optimized \
        --use_attention \
        --use_residual \
        --dropout 0.3 \
        --batch_size 8 \
        --epoch 150 \
        --learning_rate 0.001 \
        --min_lr 1e-5 \
        --optimizer Adam \
        --decay_rate 1e-4 \
        --scheduler cosine \
        --use_focal \
        --focal_gamma 2.5 \
        --use_class_weights \
        --aux_weight 0.15 \
        --npoint 16384 \
        --input_channels 6 \
        --augment \
        --early_stop \
        --patience 25 \
        --save_freq 10 \
        --eval_freq 1 \
        --log_dir accuracy_optimized \
        --gpu 0
}

# ====================================================================
# 配置 5: 快速测试（调试用）
# ====================================================================
quick_test() {
    echo "======================================================================"
    echo "配置 5: 快速测试（调试用）"
    echo "======================================================================"
    
    python train_forest_optimized.py \
        --model pointnet2_sem_seg \
        --use_optimized \
        --use_attention \
        --batch_size 4 \
        --epoch 10 \
        --learning_rate 0.001 \
        --optimizer Adam \
        --scheduler cosine \
        --npoint 4096 \
        --input_channels 6 \
        --eval_freq 1 \
        --log_dir quick_test \
        --gpu 0
}

# ====================================================================
# 配置 6: 极度不平衡数据
# ====================================================================
imbalanced_data_training() {
    echo "======================================================================"
    echo "配置 6: 针对极度不平衡数据"
    echo "======================================================================"
    
    python train_forest_optimized.py \
        --model pointnet2_sem_seg \
        --use_optimized \
        --use_attention \
        --use_residual \
        --dropout 0.3 \
        --batch_size 8 \
        --epoch 120 \
        --learning_rate 0.001 \
        --min_lr 1e-5 \
        --optimizer Adam \
        --scheduler cosine \
        --use_focal \
        --focal_gamma 3.0 \
        --use_class_weights \
        --aux_weight 0.15 \
        --npoint 16384 \
        --input_channels 6 \
        --augment \
        --early_stop \
        --patience 25 \
        --log_dir imbalanced_optimized \
        --gpu 0
}

# ====================================================================
# 配置 7: 恢复训练
# ====================================================================
resume_training() {
    echo "======================================================================"
    echo "配置 7: 恢复训练"
    echo "======================================================================"
    
    # 修改这里的路径为实际的检查点路径
    CHECKPOINT_PATH="./log/forest_seg/balanced_optimized/checkpoints/checkpoint_epoch_50.pth"
    
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "错误: 检查点文件不存在: $CHECKPOINT_PATH"
        echo "请修改 CHECKPOINT_PATH 变量为正确的路径"
        return 1
    fi
    
    python train_forest_optimized.py \
        --model pointnet2_sem_seg \
        --use_optimized \
        --use_attention \
        --dropout 0.3 \
        --batch_size 8 \
        --epoch 100 \
        --resume "$CHECKPOINT_PATH" \
        --gpu 0
}

# ====================================================================
# 主菜单
# ====================================================================
show_menu() {
    echo ""
    echo "======================================================================"
    echo "森林数据集训练脚本"
    echo "======================================================================"
    echo "请选择训练配置:"
    echo ""
    echo "1. 基础训练（原始模型）"
    echo "2. 速度优先配置"
    echo "3. 平衡配置（推荐）⭐"
    echo "4. 精度优先配置（完整优化）"
    echo "5. 快速测试（调试）"
    echo "6. 针对极度不平衡数据"
    echo "7. 恢复训练"
    echo ""
    echo "a. 运行所有配置（消融实验）"
    echo "q. 退出"
    echo "======================================================================"
    echo -n "请输入选项: "
}

# ====================================================================
# 运行所有配置（消融实验）
# ====================================================================
run_all_experiments() {
    echo "======================================================================"
    echo "开始消融实验 - 将依次运行所有配置"
    echo "======================================================================"
    
    baseline_training
    echo "等待 10 秒..."
    sleep 10
    
    speed_optimized_training
    echo "等待 10 秒..."
    sleep 10
    
    balanced_training
    echo "等待 10 秒..."
    sleep 10
    
    accuracy_optimized_training
    
    echo "======================================================================"
    echo "所有实验完成！"
    echo "======================================================================"
}

# ====================================================================
# 主程序
# ====================================================================
main() {
    # 检查 Python 脚本是否存在
    if [ ! -f "train_forest_optimized.py" ]; then
        echo "错误: 找不到 train_forest_optimized.py"
        echo "请确保在正确的目录下运行此脚本"
        exit 1
    fi
    
    # 检查数据集
    if [ ! -d "data/forest" ]; then
        echo "警告: 找不到数据集目录 data/forest"
        echo "请确保数据集路径正确"
        echo ""
    fi
    
    while true; do
        show_menu
        read choice
        
        case $choice in
            1)
                baseline_training
                ;;
            2)
                speed_optimized_training
                ;;
            3)
                balanced_training
                ;;
            4)
                accuracy_optimized_training
                ;;
            5)
                quick_test
                ;;
            6)
                imbalanced_data_training
                ;;
            7)
                resume_training
                ;;
            a|A)
                run_all_experiments
                ;;
            q|Q)
                echo "退出程序"
                exit 0
                ;;
            *)
                echo "无效选项，请重新选择"
                ;;
        esac
        
        echo ""
        echo "按 Enter 继续..."
        read
    done
}

# 运行主程序
main
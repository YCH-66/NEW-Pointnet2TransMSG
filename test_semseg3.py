import argparse
import os
from data_utils.S3DISDataLoader3 import ScannetDatasetWholeScene
from data_utils.indoor3d_util3 import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['Areca-palm', 'Water-pipe', 'Olique-line', 'Low-vegetation']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def parse_args():
    
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--data_root', type=str, default='data/forest_output/', help='data root path [default: data/forest_output/]')
    parser.add_argument('--vis_format', type=str, default='all', choices=['obj', 'pcd', 'ply', 'all'], help='visualization format [default: all]')
    parser.add_argument('--model_name', type=str, default=None, help='specific model name to load [default: auto detect]')
    return parser.parse_args()

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

def save_point_cloud_obj(filename, points, colors, labels=None):
    
    with open(filename, 'w') as f:
        for i in range(points.shape[0]):
            color = colors[i]
            f.write('v %f %f %f %d %d %d\n' % (
                points[i, 0], points[i, 1], points[i, 2], 
                color[0], color[1], color[2]))
    print(f"保存OBJ文件: {filename}")

def save_point_cloud_ply(filename, points, colors, labels=None):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    if labels is not None:
        pass
        
    o3d.io.write_point_cloud(filename, pcd)
    print(f"保存PLY文件: {filename}")

def save_point_cloud_pcd(filename, points, colors, labels=None):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"保存PCD文件: {filename}")

def create_visualization_report(visual_dir, scene_id, metrics, class_names):
    
    report_file = os.path.join(visual_dir, f'{scene_id}_report.html')
    
    html_content = f
    
    for i, class_name in enumerate(class_names):
        iou = metrics.get('class_iou', {}).get(i, 0)
        html_content += f'<div class="class-result">{class_name}: {iou:.4f}</div>'
    
    html_content += 
    
    for file in os.listdir(visual_dir):
        if scene_id in file and file != os.path.basename(report_file):
            html_content += f'<li><a href="{file}">{file}</a></li>'
    
    html_content += 
    
    for i, class_name in enumerate(class_names):
        color = g_label2color[i]
        html_content += f
    
    html_content += 
    
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    print(f"创建报告文件: {report_file}")

def save_segmentation_statistics(visual_dir, scene_id, pred_label, gt_label, class_names):
    
    stats_file = os.path.join(visual_dir, f'{scene_id}_statistics.json')
    
    stats = {
        'scene_id': scene_id,
        'total_points': len(pred_label),
        'class_distribution': {},
        'confusion_matrix': {}
    }
    
    for i, class_name in enumerate(class_names):
        pred_count = np.sum(pred_label == i)
        gt_count = np.sum(gt_label == i)
        stats['class_distribution'][class_name] = {
            'predicted': int(pred_count),
            'ground_truth': int(gt_count),
            'accuracy': float(np.sum((pred_label == i) & (gt_label == i)) / max(gt_count, 1))
        }
    
    confusion = np.zeros((len(class_names), len(class_names)))
    for i in range(len(pred_label)):
        confusion[gt_label[i], pred_label[i]] += 1
    stats['confusion_matrix'] = confusion.tolist()
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"保存统计文件: {stats_file}")
    return stats

def create_comparison_visualization(visual_dir, scene_id, points, pred_label, gt_label):
    
    try:
        pred_colors = np.array([g_label2color[label] for label in pred_label])
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pred_pcd.colors = o3d.utility.Vector3dVector(pred_colors / 255.0)
        
        gt_colors = np.array([g_label2color[label] for label in gt_label])
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        gt_pcd.colors = o3d.utility.Vector3dVector(gt_colors / 255.0)
        
        comparison_file = os.path.join(visual_dir, f'{scene_id}_comparison.ply')
        
        pred_pcd.translate([np.max(points[:, 0]) + 2, 0, 0])
        combined_pcd = gt_pcd + pred_pcd
        
        o3d.io.write_point_cloud(comparison_file, combined_pcd)
        print(f"创建对比可视化: {comparison_file}")
        
    except Exception as e:
        print(f"创建对比可视化时出错: {e}")

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = Path(experiment_dir) / 'visual'
    visual_dir.mkdir(parents=True, exist_ok=True)

    
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(str(args))

    NUM_CLASSES = 4
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    root = args.data_root
    log_string(f"使用数据路径: {root}")

    log_string("=== 数据路径调试信息 ===")
    log_string(f"数据根目录: {root}")
    log_string(f"目录是否存在: {os.path.exists(root)}")
    
    if os.path.exists(root):
        all_items = os.listdir(root)
        log_string(f"根目录内容: {all_items}")
        
        test_dir = os.path.join(root, 'test')
        if os.path.exists(test_dir):
            test_files = os.listdir(test_dir)
            npy_files = [f for f in test_files if f.endswith('.npy')]
            log_string(f"test目录中的npy文件 ({len(npy_files)}个): {npy_files[:5]}...")
            
        train_dir = os.path.join(root, 'train')
        if os.path.exists(train_dir):
            train_files = os.listdir(train_dir)
            train_npy = [f for f in train_files if f.endswith('.npy')]
            log_string(f"train目录中的npy文件 ({len(train_npy)}个): {train_npy[:5]}...")
            
        val_dir = os.path.join(root, 'val')
        if os.path.exists(val_dir):
            val_files = os.listdir(val_dir)
            val_npy = [f for f in val_files if f.endswith('.npy')]
            log_string(f"val目录中的npy文件 ({len(val_npy)}个): {val_npy[:5]}...")

    try:
        TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=args.test_area, block_points=NUM_POINT)
        log_string("数据加载成功!")
        log_string("测试数据集文件列表: %s" % TEST_DATASET_WHOLE_SCENE.file_list)
        log_string("场景数量: %d" % len(TEST_DATASET_WHOLE_SCENE.scene_points_list))
        
        if len(TEST_DATASET_WHOLE_SCENE.scene_points_list) > 0:
            scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[0]
            scene_labels = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[0]
            log_string(f"第一个场景数据形状: {scene_data.shape}")
            log_string(f"第一个场景标签形状: {scene_labels.shape}")
            log_string(f"标签唯一值: {np.unique(scene_labels)}")
            
    except Exception as e:
        log_string(f"数据加载失败: {e}")
        import traceback
        log_string(traceback.format_exc())
        log_string("请检查数据文件是否存在且格式正确")
        return

    if len(TEST_DATASET_WHOLE_SCENE) == 0:
        log_string("错误: 没有加载到任何测试数据!")
        log_string("可能的原因:")
        log_string("1. 数据文件不存在")
        log_string("2. 数据文件格式不正确") 
        log_string("3. 数据路径错误")
        log_string("4. 数据加载器筛选条件不匹配")
        return

    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    
    if args.model_name:
        model_name = args.model_name
        log_string(f"使用用户指定的模型: {model_name}")
    else:
        log_files = os.listdir(experiment_dir + '/logs')
        log_string(f"日志目录中的文件: {log_files}")
        
        model_files = [f for f in log_files if f.endswith('.py')]
        if not model_files:
            model_files = [f for f in os.listdir(experiment_dir) if f.endswith('.py') and 'pointnet' in f and 'sem_seg' in f]
        
        if not model_files:
            log_string("警告: 在日志目录中找不到模型文件")
            model_candidates = [f for f in os.listdir('models') if 'sem_seg' in f and f.endswith('.py')]
            if model_candidates:
                if os.path.exists(experiment_dir + '/logs'):
                    log_base_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
                    matched_models = [f for f in model_candidates if log_base_name.replace('_msg', '') in f or log_base_name in f]
                    if matched_models:
                        model_name = matched_models[0][:-3]
                        log_string(f"根据日志文件名匹配模型: {model_name}")
                    else:
                        model_name = 'pointnet2_sem_seg_msg'
                        log_string(f"使用默认模型: {model_name}")
                else:
                    model_name = 'pointnet2_sem_seg_msg'
                    log_string(f"使用默认模型: {model_name}")
            else:
                log_string("错误: 在models目录中也找不到合适的模型文件")
                return
        else:
            model_name = model_files[0].split('.')[0]
            log_string(f"加载模型: {model_name}")
            
            if os.path.exists(os.path.join(experiment_dir, model_files[0])):
                spec = importlib.util.spec_from_file_location(model_name, os.path.join(experiment_dir, model_files[0]))
                MODEL = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(MODEL)
                sys.modules[model_name] = MODEL
                log_string(f"成功从日志目录直接加载模型: {model_name}")
    
    model_name = model_name.split('/')[-1]
    log_string(f"处理后的模型名称: {model_name}")
    
    if 'MODEL' not in locals():
        try:
            MODEL = importlib.import_module(model_name)
            log_string(f"成功导入模型模块: {model_name}")
        except Exception as e:
            log_string(f"无法直接导入模型模块 {model_name}: {e}")
            try:
                if experiment_dir in sys.path and os.path.exists(os.path.join(experiment_dir, model_name + '.py')):
                    spec = importlib.util.spec_from_file_location(model_name, os.path.join(experiment_dir, model_name + '.py'))
                    MODEL = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(MODEL)
                    sys.modules[model_name] = MODEL
                    log_string(f"成功从日志目录直接加载模型: {model_name}")
                else:
                    MODEL = importlib.import_module('models.' + model_name)
                    log_string(f"成功从models目录导入模型: {model_name}")
            except Exception as e2:
                log_string(f"也无法从特定路径导入模型: {e2}")
                return
    
    if not hasattr(MODEL, 'get_model'):
        log_string(f"错误: 模型模块 {model_name} 不包含 get_model 方法")
        return
    
    if len(TEST_DATASET_WHOLE_SCENE.scene_points_list) > 0:
        sample_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[0]
        total_channels = sample_data.shape[1]
        in_channel = max(0, total_channels - 3)
        log_string(f"自动检测到总特征维度: {total_channels}, in_channel参数: {in_channel}")
    else:
        in_channel = 3
        log_string(f"使用默认in_channel参数: {in_channel}")
    
    try:
        import inspect
        model_signature = inspect.signature(MODEL.get_model)
        model_params = list(model_signature.parameters.keys())
        
        log_string(f"模型支持的参数: {model_params}")
        
        if 'normal_channel' in model_params:
            classifier = MODEL.get_model(NUM_CLASSES, normal_channel=(in_channel >= 3)).cuda()
            log_string(f"成功创建部件分割模型，normal_channel参数: {in_channel >= 3}")
        elif 'in_channel' in model_params:
            classifier = MODEL.get_model(NUM_CLASSES, in_channel=6).cuda()
            log_string(f"成功创建模型，in_channel参数: 6")
        else:
            classifier = MODEL.get_model(NUM_CLASSES).cuda()
            log_string("成功创建语义分割模型")
    except Exception as e:
        log_string(f"使用自动检测的参数创建模型失败: {e}")
        try:
            classifier = MODEL.get_model(NUM_CLASSES).cuda()
            log_string("成功创建模型（使用默认参数）")
        except Exception as e2:
            log_string(f"使用默认参数创建模型也失败了: {e2}")
            try:
                classifier = MODEL.get_model(NUM_CLASSES, in_channel=6).cuda()
                log_string("成功创建模型（使用in_channel=6）")
            except Exception as e3:
                log_string(f"使用in_channel=6创建模型也失败了: {e3}")
                return

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location='cuda')
        classifier.load_state_dict(checkpoint['model_state_dict'], strict=True)
        log_string('成功使用严格模式加载模型权重')
    except Exception as e:
        log_string(f'严格模式加载失败: {e}')
        try:
            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location='cuda')
            classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)
            log_string('成功使用非严格模式加载模型权重')
        except Exception as e2:
            log_string(f'非严格模式加载也失败: {e2}')
            try:
                checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location='cuda')
                model_dict = classifier.state_dict()
                pretrained_dict = checkpoint['model_state_dict']
                
                filtered_dict = {}
                for k, v in pretrained_dict.items():
                    if k in model_dict and v.shape == model_dict[k].shape:
                        filtered_dict[k] = v
                    elif k in model_dict:
                        log_string(f'形状不匹配，跳过键 {k}: 预训练权重 {v.shape} vs 当前模型 {model_dict[k].shape}')
                
                model_dict.update(filtered_dict)
                classifier.load_state_dict(model_dict, strict=False)
                log_string('成功使用手动适配方式加载模型权重')
            except Exception as e3:
                log_string(f'手动适配方式也失败: {e3}')
                return

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION WHOLE SCENE----')

        for batch_idx in range(num_batches):
            log_string("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
            
            scene_visual_dir = visual_dir / scene_id[batch_idx]
            scene_visual_dir.mkdir(exist_ok=True)
            
            if args.visual:
                fout = open(os.path.join(scene_visual_dir, 'pred.obj'), 'w')
                fout_gt = open(os.path.join(scene_visual_dir, 'gt.obj'), 'w')

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            
            scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
            num_blocks = scene_data.shape[0]
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
            
            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                    
                    torch_data = torch.from_numpy(batch_data).float().cuda()
                    
                    current_channels = torch_data.shape[2]
                    if current_channels > 6:
                        torch_data = torch_data[:, :, :6]
                    elif current_channels < 6:
                        padding = torch.zeros(torch_data.shape[0], torch_data.shape[1], 6 - current_channels, 
                                            dtype=torch.float32, device=torch_data.device)
                        torch_data = torch.cat([torch_data, padding], dim=2)
                    
                    rgb_mask = torch_data[:, :, 3:6]
                    torch_data[:, :, 3:6] = torch.clamp(rgb_mask, 0, 255) / 255.0
                    
                    torch_data = torch_data.permute(0, 2, 1)
                    
                    import inspect
                    forward_params = list(inspect.signature(classifier.forward).parameters.keys())
                    
                    if len(forward_params) >= 2 and forward_params[1] == 'cls_label':
                        cls_label = torch.zeros(torch_data.shape[0], 16).cuda()
                        seg_pred, _ = classifier(torch_data, cls_label)
                    else:
                        seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)

            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            with np.errstate(divide='ignore', invalid='ignore'):
                iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float64) + 1e-6)
                iou_map = np.nan_to_num(iou_map)
            
            log_string(f"IoU map: {iou_map}")
            arr = np.array(total_seen_class_tmp)
            valid_iou = iou_map[arr != 0]
            if len(valid_iou) > 0:
                tmp_iou = np.mean(valid_iou)
                log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            else:
                log_string('Mean IoU of %s: No valid classes' % scene_id[batch_idx])
            log_string('----------------------------')

            if args.visual:
                scene_visual_dir = os.path.join(experiment_dir, 'visual', scene_id[batch_idx])
                if not os.path.exists(scene_visual_dir):
                    os.makedirs(scene_visual_dir)
                    
                pred_colors = np.array([g_label2color[label] for label in pred_label])
                gt_colors = np.array([g_label2color[label] for label in whole_scene_label])
                
                points_xyz = whole_scene_data[:, :3]
                
                if args.vis_format in ['obj', 'all']:
                    save_point_cloud_obj(os.path.join(scene_visual_dir, 'pred.obj'), points_xyz, pred_colors, pred_label)
                    save_point_cloud_obj(os.path.join(scene_visual_dir, 'gt.obj'), points_xyz, gt_colors, whole_scene_label)
                
                if args.vis_format in ['ply', 'all']:
                    save_point_cloud_ply(os.path.join(scene_visual_dir, 'pred.ply'), points_xyz, pred_colors, pred_label)
                    save_point_cloud_ply(os.path.join(scene_visual_dir, 'gt.ply'), points_xyz, gt_colors, whole_scene_label)
                
                if args.vis_format in ['pcd', 'all']:
                    save_point_cloud_pcd(os.path.join(scene_visual_dir, 'pred.pcd'), points_xyz, pred_colors, pred_label)
                    save_point_cloud_pcd(os.path.join(scene_visual_dir, 'gt.pcd'), points_xyz, gt_colors, whole_scene_label)
                
                create_comparison_visualization(scene_visual_dir, scene_id[batch_idx], 
                                              whole_scene_data, pred_label, whole_scene_label)
                
                stats = save_segmentation_statistics(scene_visual_dir, scene_id[batch_idx], 
                                                   pred_label, whole_scene_label, classes)
                
                metrics = {
                    'mean_iou': tmp_iou if len(valid_iou) > 0 else 0,
                    'overall_accuracy': np.sum(total_correct_class_tmp) / max(np.sum(total_seen_class_tmp), 1),
                    'class_iou': {i: iou_map[i] for i in range(NUM_CLASSES)}
                }
                create_visualization_report(scene_visual_dir, scene_id[batch_idx], metrics, classes)
                
                log_string(f"为场景 {scene_id[batch_idx]} 生成完整的可视化文件")

            filename = os.path.join(scene_visual_dir, 'predictions.txt')
            with open(filename, 'w') as pl_save:
                for i in pred_label:
                    pl_save.write(str(int(i)) + '\n')
                pl_save.close()

        with np.errstate(divide='ignore', invalid='ignore'):
            IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
            IoU = np.nan_to_num(IoU)
            
            accuracy_per_class = np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6)
            accuracy_per_class = np.nan_to_num(accuracy_per_class)

        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            if total_iou_deno_class[l] > 0:
                iou = total_correct_class[l] / float(total_iou_deno_class[l])
            else:
                iou = 0.0
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), iou)
        
        log_string(iou_per_class_str)
        log_string('eval point avg class IoU: %f' % np.mean(IoU))
        log_string('eval whole scene point avg class acc: %f' % np.mean(accuracy_per_class))
        
        total_seen = float(np.sum(total_seen_class) + 1e-6)
        if total_seen > 0:
            overall_accuracy = np.sum(total_correct_class) / total_seen
            log_string('eval whole scene point accuracy: %f' % overall_accuracy)
        else:
            log_string('eval whole scene point accuracy: No points')

        log_string("Done!")

if __name__ == '__main__':
    args = parse_args()
    main(args)
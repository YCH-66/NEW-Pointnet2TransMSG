import argparse
import os
from data_utils.forestnew import ForestNewDataset
import torch
import torch.nn.functional as F
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError as e:
    print(f"Visualization libraries not available: {e}")
    print("Plots and detailed metrics will be skipped.")
    HAS_VISUALIZATION = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['Areca-palm', 'Water-pipe', 'Olique-line', 'Low-vegetation']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {i: cat for i, cat in enumerate(seg_classes.keys())}

def inplace_relu(m):
    if m.__class__.__name__.find('ReLU') != -1:
        m.inplace = True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_trans', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=300, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--data_root', type=str, default=None, help='自定义数据根目录，优先级最高')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers [default: 4]')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 [default: 42]')
    parser.add_argument('--save_plots', action='store_true', default=True, help='保存PR曲线和混淆矩阵')
    parser.add_argument('--save_csv', action='store_true', default=True, help='保存详细结果到CSV文件')
    return parser.parse_args()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_detailed_metrics(all_predictions, all_targets, all_probabilities, num_classes, class_names):
    
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    probabilities = np.concatenate(all_probabilities)
    
    valid_mask = (targets >= 0) & (targets < num_classes)
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    probabilities = probabilities[valid_mask]
    
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(len(targets)):
        cm[targets[i], predictions[i]] += 1
    
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    iou_per_class = []
    accuracy_per_class = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        iou = tp / (tp + fp + fn + 1e-8)
        
        accuracy_class = (tp + tn) / (tp + fn + fp + tn + 1e-8)
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
        iou_per_class.append(iou)
        accuracy_per_class.append(accuracy_class)
    
    accuracy = (targets == predictions).sum() / len(targets)
    
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)
    macro_accuracy = np.mean(accuracy_per_class)
    mIoU = np.mean(iou_per_class)
    
    support = cm.sum(axis=1)
    weighted_precision = np.average(precision_per_class, weights=support)
    weighted_recall = np.average(recall_per_class, weights=support)
    weighted_f1 = np.average(f1_per_class, weights=support)
    weighted_accuracy = np.average(accuracy_per_class, weights=support)
    
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'macro_accuracy': macro_accuracy,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'weighted_accuracy': weighted_accuracy,
        'mIoU': mIoU,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'iou_per_class': iou_per_class,
        'accuracy_per_class': accuracy_per_class,
        'confusion_matrix': cm,
        'support': support,
        'probabilities': probabilities,
        'targets': targets
    }
    
    return metrics

def plot_pr_curves(metrics, class_names, save_path):
    
    if not HAS_VISUALIZATION:
        return
        
    plt.figure(figsize=(12, 8))
    
    probabilities = metrics['probabilities']
    targets = metrics['targets']
    num_classes = len(class_names)
    
    for i in range(num_classes):
        y_true = (targets == i).astype(int)
        y_scores = probabilities[:, i]
        
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        plt.plot(recall, precision, lw=2, 
                label=f'{class_names[i]} (AP = {ap:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Each Class')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path):
    
    if not HAS_VISUALIZATION:
        return
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_results_to_file(metrics, class_names, save_path):
    
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED EVALUATION METRICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:           {metrics['accuracy']:.4f}\n")
        f.write(f"Macro Accuracy:     {metrics['macro_accuracy']:.4f}\n")
        f.write(f"Weighted Accuracy:  {metrics['weighted_accuracy']:.4f}\n")
        f.write(f"Macro Precision:    {metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall:       {metrics['macro_recall']:.4f}\n")
        f.write(f"Macro F1-Score:     {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted Precision: {metrics['weighted_precision']:.4f}\n")
        f.write(f"Weighted Recall:    {metrics['weighted_recall']:.4f}\n")
        f.write(f"Weighted F1-Score:  {metrics['weighted_f1']:.4f}\n")
        f.write(f"mIoU:               {metrics['mIoU']:.4f}\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Class':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'IoU':<10} {'Support':<10}\n")
        f.write("-" * 80 + "\n")
        
        for i, name in enumerate(class_names):
            f.write(f"{name:<15} {metrics['accuracy_per_class'][i]:<10.4f} {metrics['precision_per_class'][i]:<10.4f} "
                   f"{metrics['recall_per_class'][i]:<10.4f} {metrics['f1_per_class'][i]:<10.4f} "
                   f"{metrics['iou_per_class'][i]:<10.4f} {metrics['support'][i]:<10}\n")
        
        f.write("\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 40 + "\n")
        cm = metrics['confusion_matrix']
        f.write("     " + " ".join([f"{name:>6}" for name in class_names]) + "\n")
        for i, name in enumerate(class_names):
            f.write(f"{name:<5} " + " ".join([f"{cm[i,j]:>6}" for j in range(len(class_names))]) + "\n")

def save_results_to_csv(metrics, class_names, save_path):
    
    if not HAS_VISUALIZATION:
        return None, None
        
    class_results = []
    for i, name in enumerate(class_names):
        class_results.append({
            'Class': name,
            'Accuracy': metrics['accuracy_per_class'][i],
            'Precision': metrics['precision_per_class'][i],
            'Recall': metrics['recall_per_class'][i],
            'F1-Score': metrics['f1_per_class'][i],
            'IoU': metrics['iou_per_class'][i],
            'Support': metrics['support'][i]
        })
    
    overall_results = [{
        'Metric': 'Accuracy',
        'Value': metrics['accuracy']
    }, {
        'Metric': 'Macro Accuracy',
        'Value': metrics['macro_accuracy']
    }, {
        'Metric': 'Weighted Accuracy',
        'Value': metrics['weighted_accuracy']
    }, {
        'Metric': 'Macro Precision',
        'Value': metrics['macro_precision']
    }, {
        'Metric': 'Macro Recall', 
        'Value': metrics['macro_recall']
    }, {
        'Metric': 'Macro F1-Score',
        'Value': metrics['macro_f1']
    }, {
        'Metric': 'Weighted Precision',
        'Value': metrics['weighted_precision']
    }, {
        'Metric': 'Weighted Recall',
        'Value': metrics['weighted_recall']
    }, {
        'Metric': 'Weighted F1-Score',
        'Value': metrics['weighted_f1']
    }, {
        'Metric': 'mIoU',
        'Value': metrics['mIoU']
    }]
    
    class_df = pd.DataFrame(class_results)
    overall_df = pd.DataFrame(overall_results)
    
    base_path = save_path.replace('.csv', '')
    class_df.to_csv(f'{base_path}_class_details.csv', index=False)
    overall_df.to_csv(f'{base_path}_overall_metrics.csv', index=False)
    
    return class_df, overall_df

def compute_class_weights_from_loader(dataset, num_classes, device, batch_size=16, num_workers=0, method="log_inv"):
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False, drop_last=False
    )
    counts = np.zeros(num_classes, dtype=np.int64)
    for pts, lbl in loader:
        arr = lbl.numpy().ravel()
        arr = arr[(arr >= 0) & (arr < num_classes)]
        if arr.size:
            counts += np.bincount(arr, minlength=num_classes)

    if counts.sum() == 0:
        weights = np.ones(num_classes, dtype=np.float64)
    else:
        if method == "inv_freq":
            c = counts.copy().astype(np.float64)
            c[c == 0] = 1.0
            weights = c.max() / c
        elif method == "effective":
            beta = 0.999
            eff = 1.0 - np.power(beta, counts.astype(np.float64))
            eff[eff == 0] = 1.0
            weights = (1.0 - beta) / eff
        else:
            freq = counts / counts.sum()
            weights = 1.0 / (np.log(1.2 + np.maximum(freq, 1e-12)))

    weights = weights / (weights.mean() + 1e-12)
    return torch.tensor(weights, dtype=torch.float32, device=device)

def sanitize_labels_(target_tensor, num_classes):
    
    bad = (target_tensor < 0) | (target_tensor >= num_classes)
    if torch.any(bad):
        target_tensor[bad] = 0
        return bad.sum().item()
    return 0

def load_compatible_model(module, num_classes, in_channel, device, logger):
    
    try:
        model = module.get_model(num_classes, in_channel=in_channel)
        logger.info(f"Model loaded with in_channel={in_channel}")
        return model
    except TypeError as e:
        if "unexpected keyword argument 'in_channel'" in str(e):
            logger.info("Model does not support in_channel parameter, using default initialization")
            model = module.get_model(num_classes)
            
            if in_channel != 0:
                logger.info(f"Adjusting model input channels from 3 to {3 + in_channel}")
                model = adjust_model_input_channels(model, in_channel, logger)
            return model
        else:
            raise e

def adjust_model_input_channels(model, extra_channels, logger):
    
    total_channels = 3 + extra_channels
    
    adjusted = False
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            if module.in_channels == 3:
                new_conv = torch.nn.Conv1d(
                    total_channels, 
                    module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    bias=module.bias is not None
                )
                
                with torch.no_grad():
                    if total_channels > 3:
                        new_weight = torch.cat([
                            module.weight.data,
                            torch.randn(module.out_channels, extra_channels, module.kernel_size[0]) * 0.02
                        ], dim=1)
                        new_conv.weight.data = new_weight
                    else:
                        new_conv.weight.data = module.weight.data
                    
                    if module.bias is not None:
                        new_conv.bias.data = module.bias.data
                
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                    setattr(parent, name.split('.')[-1], new_conv)
                else:
                    setattr(model, name, new_conv)
                
                logger.info(f"Adjusted input channels of {name} from 3 to {total_channels}")
                adjusted = True
    
    if not adjusted:
        logger.warning("No convolutional layers found to adjust!")
    
    return model

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        if args.log_dir.startswith('log/sem_seg/'):
            experiment_dir = Path(args.log_dir)
        else:
            experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    plots_dir = experiment_dir.joinpath('plots/')
    plots_dir.mkdir(exist_ok=True)
    results_dir = experiment_dir.joinpath('results/')
    results_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{log_dir}/{args.model}.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)

    def log_string(s):
        logger.info(s)
        print(s)

    log_string('PARAMETER ...')
    log_string(str(vars(args)))
    
    log_string(f'TRAINING MODEL: {args.model}')
    log_string('=' * 60)

    if args.data_root is not None:
        root = args.data_root
    else:
        local_data_path = os.path.join(BASE_DIR, 'date/forest_output/')
        legacy_path = '/home/y/Pointnet++/data/forest_output/'
        
        if os.path.exists(local_data_path):
            root = local_data_path
        elif os.path.exists(legacy_path):
            root = legacy_path
        else:
            root = os.path.join(BASE_DIR, 'data/forest_output/')
            print(f"警告：找不到数据目录，请确保以下路径之一存在:")
            print(f"  - {local_data_path}")
            print(f"  - {legacy_path}")
            print(f"  - {root}")

    print(f"使用的数据根目录: {root}")

    NUM_CLASSES = 4
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    num_workers = max(0, args.num_workers)
    if device.type == 'cpu':
        num_workers = min(num_workers, 2)

    print("=" * 60)
    print("start loading training data ...")
    print("=" * 60)
    try:
        TRAIN_DATASET = ForestNewDataset(
            split='train',
            data_root=root,
            num_point=NUM_POINT,
            test_area=args.test_area,
            block_size=1.0,
            sample_rate=1.0,
            transform=None
        )
    except Exception as e:
        log_string(f"Error loading training dataset: {e}")
        raise

    print("=" * 60)
    print("start loading test data ...")
    print("=" * 60)
    try:
        TEST_DATASET = ForestNewDataset(
            split='test',
            data_root=root,
            num_point=NUM_POINT,
            test_area=args.test_area,
            block_size=1.0,
            sample_rate=1.0,
            transform=None
        )
    except Exception as e:
        log_string(f"Error loading test dataset: {e}")
        raise

    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )

    weights = None
    try:
        lw = np.asarray(getattr(TRAIN_DATASET, 'labelweights'), dtype=np.float32).reshape(-1)
        if lw.size == NUM_CLASSES:
            weights = torch.tensor(lw, dtype=torch.float32, device=device)
            log_string(f"Use dataset-provided class weights (len={lw.size}): {lw}")
        else:
            log_string(f"[Warn] Dataset labelweights length {lw.size} != NUM_CLASSES {NUM_CLASSES}, will recompute...")
    except Exception as e:
        log_string(f"[Warn] No valid labelweights in dataset: {e}")

    if weights is None:
        weights = compute_class_weights_from_loader(
            TRAIN_DATASET, num_classes=NUM_CLASSES, device=device,
            batch_size=BATCH_SIZE, num_workers=min(2, num_workers), method="log_inv"
        )
        log_string(f"Recomputed class weights (len={len(weights)}): {weights.detach().cpu().numpy()}")

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    sample_points, _ = next(iter(trainDataLoader))
    detected_total_C = sample_points.shape[-1]
    assert detected_total_C >= 3, f"数据最后一维应≥3，当前 {detected_total_C}"
    in_channel = detected_total_C - 3
    log_string(f"[AutoDetect] total C={detected_total_C} -> in_channel={in_channel}")

    MODEL = importlib.import_module(f"models.{args.model}")
    try:
        shutil.copy(f'models/{args.model}.py', str(experiment_dir))
    except Exception as e:
        log_string(f'Warning: copy model file failed: {e}')
    try:
        shutil.copy('models/pointnet2_utils.py', str(experiment_dir))
    except Exception as e:
        log_string(f'Warning: copy utils file failed: {e}')

    classifier = load_compatible_model(MODEL, NUM_CLASSES, in_channel, device, logger)
    classifier = classifier.to(device)
    classifier.apply(inplace_relu)
    
    log_string("Testing model with sample input...")
    with torch.no_grad():
        sample_input = torch.randn(2, 3 + in_channel, NUM_POINT).to(device)
        try:
            output, _ = classifier(sample_input)
            log_string(f"Model test successful! Output shape: {output.shape}")
        except Exception as e:
            log_string(f"Model test failed: {e}")
            raise

    def weights_init(m):
        name = m.__class__.__name__
        if 'Conv' in name or 'Linear' in name:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.xavier_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    start_epoch = 0
    best_iou = 0.0
    ckpt_path = str(checkpoints_dir / 'best_model.pth')
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        start_epoch = checkpoint.get('epoch', 0)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
        best_iou = float(checkpoint.get('class_avg_iou', 0.0))
    except Exception:
        log_string('No existing model, starting training from scratch...')
        classifier.apply(weights_init)

    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0

    for epoch in range(start_epoch, args.epoch):
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        momentum = max(momentum, 0.01)
        print('BN momentum updated to: %f' % momentum)
        classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0.0
        classifier.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points_np = points.numpy()
            points_np[:, :, :3] = provider.rotate_point_cloud_z(points_np[:, :, :3])

            points = torch.as_tensor(points_np, dtype=torch.float32, device=device)
            target = target.to(device=device, dtype=torch.long)

            n_fixed = sanitize_labels_(target, NUM_CLASSES)
            if n_fixed > 0:
                if i == 0:
                    log_string(f"[Train] Fixed {n_fixed} out-of-range labels in current batch (set to 0).")

            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            target_flat = target.view(-1)

            if hasattr(MODEL, 'get_loss'):
                if isinstance(MODEL.get_loss, type):
                    criterion = MODEL.get_loss()
                    loss = criterion(seg_pred, target_flat, trans_feat, weights)
                else:
                    loss = MODEL.get_loss(seg_pred, target_flat, trans_feat, weights)
            else:
                loss = F.cross_entropy(seg_pred, target_flat, weight=weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.detach().cpu().argmax(1).numpy()
            batch_label = target_flat.detach().cpu().numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss.item()

        log_string('Training mean loss: %f' % (loss_sum / max(1, num_batches)))
        log_string('Training accuracy: %f' % (total_correct / float(max(1, total_seen))))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir / 'model.pth')
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_avg_iou': best_iou
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0.0
            labelweights_eval = np.zeros(NUM_CLASSES, dtype=np.float64)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            
            all_predictions = []
            all_targets = []
            all_probabilities = []
            
            classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = torch.as_tensor(points.numpy(), dtype=torch.float32, device=device)
                target = target.to(device=device, dtype=torch.long)

                n_fixed_eval = sanitize_labels_(target, NUM_CLASSES)
                if n_fixed_eval > 0 and i == 0:
                    log_string(f"[Eval] Fixed {n_fixed_eval} out-of-range labels in current batch (set to 0).")

                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.detach().cpu().numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                target_flat = target.view(-1)

                if hasattr(MODEL, 'get_loss'):
                    if isinstance(MODEL.get_loss, type):
                        criterion = MODEL.get_loss()
                        loss = criterion(seg_pred, target_flat, trans_feat, weights)
                    else:
                        loss = MODEL.get_loss(seg_pred, target_flat, trans_feat, weights)
                else:
                    loss = F.cross_entropy(seg_pred, target_flat, weight=weights)

                loss_sum += loss.item()

                pred_val_cls = np.argmax(pred_val, axis=2)
                batch_label = target.detach().cpu().numpy()
                correct = np.sum((pred_val_cls == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)

                tmp, _ = np.histogram(batch_label, bins=np.arange(NUM_CLASSES + 1))
                labelweights_eval += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val_cls == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val_cls == l) | (batch_label == l)))
                
                all_predictions.append(pred_val_cls.reshape(-1))
                all_targets.append(batch_label.reshape(-1))
                all_probabilities.append(pred_val.reshape(-1, NUM_CLASSES))

            if all_predictions:
                metrics = calculate_detailed_metrics(all_predictions, all_targets, all_probabilities, NUM_CLASSES, classes)
                
                log_string('=' * 80)
                log_string('DETAILED EVALUATION METRICS:')
                log_string('-' * 40)
                log_string(f"Accuracy:           {metrics['accuracy']:.4f}")
                log_string(f"Macro Accuracy:     {metrics['macro_accuracy']:.4f}")
                log_string(f"Weighted Accuracy:  {metrics['weighted_accuracy']:.4f}")
                log_string(f"Macro Precision:    {metrics['macro_precision']:.4f}")
                log_string(f"Macro Recall:       {metrics['macro_recall']:.4f}")
                log_string(f"Macro F1-Score:     {metrics['macro_f1']:.4f}")
                log_string(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
                log_string(f"Weighted Recall:    {metrics['weighted_recall']:.4f}")
                log_string(f"Weighted F1-Score:  {metrics['weighted_f1']:.4f}")
                log_string(f"mIoU:               {metrics['mIoU']:.4f}")
                
                log_string('-' * 40)
                log_string('PER-CLASS METRICS:')
                log_string(f"{'Class':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'IoU':<10}")
                log_string('-' * 70)
                for i, name in enumerate(classes):
                    log_string(f"{name:<15} {metrics['accuracy_per_class'][i]:<10.4f} {metrics['precision_per_class'][i]:<10.4f} "
                              f"{metrics['recall_per_class'][i]:<10.4f} {metrics['f1_per_class'][i]:<10.4f} {metrics['iou_per_class'][i]:<10.4f}")
                
                if epoch == args.epoch - 1:
                    results_path = str(results_dir / f'final_metrics.txt')
                    save_results_to_file(metrics, classes, results_path)
                    log_string(f'Saved detailed metrics to: {results_path}')
                    
                    if args.save_csv:
                        csv_path = str(results_dir / f'final_results.csv')
                        save_results_to_csv(metrics, classes, csv_path)
                        log_string(f'Saved CSV results to: {csv_path}')
                    
                    if args.save_plots and HAS_VISUALIZATION:
                        pr_curve_path = str(plots_dir / f'final_pr_curve.png')
                        plot_pr_curves(metrics, classes, pr_curve_path)
                        
                        cm_path = str(plots_dir / f'final_confusion_matrix.png')
                        plot_confusion_matrix(metrics['confusion_matrix'], classes, cm_path)
                        
                        log_string(f'Saved plots to: {pr_curve_path}, {cm_path}')

            ious = []
            for l in range(NUM_CLASSES):
                deno = float(total_iou_deno_class[l]) + 1e-6
                iou_l = float(total_correct_class[l]) / deno
                ious.append(iou_l)

            mIoU = float(np.mean(ious))

            log_string('eval mean loss: %f' % (loss_sum / max(1, num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(max(1, total_seen))))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir / 'best_model.pth')
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1

if __name__ == '__main__':
    args = parse_args()
    main(args)
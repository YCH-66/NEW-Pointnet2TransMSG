import numpy as np
import torch
from data_utils.ForestDataLoader import ForestDataset

def analyze_class_distribution():
    dataset = ForestDataset(root='data/forest', split='train', num_points=16384)
    
    all_labels = []
    for i in range(len(dataset)):
        _, labels = dataset[i]
        all_labels.append(labels.numpy())
    
    all_labels = np.concatenate(all_labels)
    
    classes = ['forest', 'shuiguan', 'xiexian', 'diaizhibei']
    print("="*60)
    print("类别分布分析")
    print("="*60)
    
    for i, cls in enumerate(classes):
        count = (all_labels == i).sum()
        ratio = count / len(all_labels) * 100
        print(f"{cls:15s}: {count:10d} points ({ratio:5.2f}%)")
    
    print("\n不平衡比率:")
    counts = [((all_labels == i).sum()) for i in range(len(classes))]
    max_count = max(counts)
    for i, cls in enumerate(classes):
        ratio = max_count / (counts[i] + 1e-8)
        print(f"{cls:15s}: 1:{ratio:.1f}")
    
    return counts

def analyze_per_class_performance(model_path):
    import importlib
    from torch.utils.data import DataLoader
    
    MODEL = importlib.import_module('pointnet2_sem_seg_transformer')
    model = MODEL.get_model(num_classes=4, use_transformer='bottleneck').cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_dataset = ForestDataset(root='data/forest', split='test', num_points=16384)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    class_tp = np.zeros(4)
    class_fp = np.zeros(4)
    class_fn = np.zeros(4)
    class_tn = np.zeros(4)
    
    with torch.no_grad():
        for points, target in test_loader:
            points = points.cuda().transpose(2, 1)
            target = target.cuda()
            
            pred, _ = model(points)
            pred_choice = pred.argmax(dim=2).view(-1)
            target = target.view(-1)
            
            for c in range(4):
                class_tp[c] += ((pred_choice == c) & (target == c)).sum().item()
                class_fp[c] += ((pred_choice == c) & (target != c)).sum().item()
                class_fn[c] += ((pred_choice != c) & (target == c)).sum().item()
    
    classes = ['forest', 'shuiguan', 'xiexian', 'diaizhibei']
    print("\n" + "="*60)
    print("每类性能分析")
    print("="*60)
    print(f"{'类别':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'IoU':<12}")
    print("-"*60)
    
    for i, cls in enumerate(classes):
        prec = class_tp[i] / (class_tp[i] + class_fp[i] + 1e-8)
        rec = class_tp[i] / (class_tp[i] + class_fn[i] + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        iou = class_tp[i] / (class_tp[i] + class_fp[i] + class_fn[i] + 1e-8)
        
        print(f"{cls:<15} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {iou:<12.4f}")

if __name__ == '__main__':
    counts = analyze_class_distribution()
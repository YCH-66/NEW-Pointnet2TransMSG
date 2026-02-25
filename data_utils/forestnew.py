import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class ForestNewDataset(Dataset):
    def __init__(self, split='train', data_root='/home/y/NEW-Pointnet++/data/forest_output', num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0, transform=None, verbose=False):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.verbose = verbose
        
        if split == 'train':
            data_root = os.path.join(data_root, 'train')
        elif split == 'test':
            data_root = os.path.join(data_root, 'test')
        elif split == 'val':
            data_root = os.path.join(data_root, 'val')
        
        files = sorted(os.listdir(data_root))
        files = [file for file in files if file.endswith('.npy')]
        
        if self.verbose:
            print(f"Found {len(files)} .npy files")
        
        files_split = files
        
        if self.verbose:
            print(f"Found {len(files_split)} files for {split} split")
        
        self.file_points, self.file_labels = [], []
        self.file_coord_min, self.file_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(4)

        for file_name in tqdm(files_split, total=len(files_split), desc=f"Loading {split} data"):
            file_path = os.path.join(data_root, file_name)
            if not os.path.exists(file_path):
                if self.verbose:
                    print(f"Warning: File path {file_path} does not exist. Skipping.")
                continue
            
            try:
                data = np.load(file_path, allow_pickle=True)
                
                if data.ndim != 2 or data.shape[1] != 7:
                    if self.verbose:
                        print(f"Warning: Data in {file_path} has wrong shape {data.shape}. Skipping.")
                    continue
                    
            except Exception as e:
                if self.verbose:
                    print(f"Error loading {file_path}: {e}")
                continue
            
            points = data[:, 0:6]
            labels = data[:, 6]
            
            nan_mask = np.isnan(labels)
            if np.any(nan_mask):
                if self.verbose:
                    print(f"Warning: Found {np.sum(nan_mask)} NaN values in labels of {file_name}. Removing them.")
                valid_mask = ~nan_mask
                points = points[valid_mask]
                labels = labels[valid_mask]
            
            inf_mask = np.isinf(labels)
            if np.any(inf_mask):
                if self.verbose:
                    print(f"Warning: Found {np.sum(inf_mask)} infinite values in labels of {file_name}. Removing them.")
                valid_mask = ~inf_mask
                points = points[valid_mask]
                labels = labels[valid_mask]
            
            if len(labels) == 0:
                if self.verbose:
                    print(f"Warning: No valid data left in {file_name} after cleaning. Skipping.")
                continue
            
            labels = np.round(labels).astype(np.int32)
            
            unique_labels = np.unique(labels)
            if self.verbose and len(unique_labels) > 0:
                print(f"Label range in {file_name}: {unique_labels.min()} - {unique_labels.max()}")
            
            tmp, _ = np.histogram(labels, range(5))
            labelweights += tmp
            
            coord_min = np.amin(points[:, 0:3], axis=0)
            coord_max = np.amax(points[:, 0:3], axis=0)
            
            if np.any(coord_max == coord_min):
                coord_max = coord_max + 1e-6
            
            self.file_points.append(points)
            self.file_labels.append(labels)
            self.file_coord_min.append(coord_min)
            self.file_coord_max.append(coord_max)
            num_point_all.append(labels.size)

        if len(self.file_points) == 0:
            raise ValueError(f"No valid data files found in {data_root}")

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights) if np.sum(labelweights) != 0 else labelweights
        self.labelweights = np.power(np.amax(labelweights) / (labelweights + 1e-8), 1 / 3.0)
        if self.verbose:
            print(f"Label weights: {self.labelweights}")

        sample_prob = np.array(num_point_all) / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        
        file_idxs = []
        for index in range(len(self.file_points)):
            file_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.file_idxs = np.array(file_idxs)
        
        if self.verbose:
            print(f"Totally {len(self.file_idxs)} samples in {split} set.")

    def __getitem__(self, idx):
        file_idx = self.file_idxs[idx]
        points = self.file_points[file_idx]
        labels = self.file_labels[file_idx]
        N_points = points.shape[0]

        attempt = 0
        while attempt < 10:
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where(
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) &
                (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1])
            )[0]
            if point_idxs.size > 1024:
                break
            attempt += 1
        
        if point_idxs.size < 1024:
            point_idxs = np.arange(N_points)

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        selected_points = points[selected_point_idxs, :]
        current_points = np.zeros((self.num_point, 9), dtype=np.float32)
        
        current_points[:, 6] = selected_points[:, 0] / (self.file_coord_max[file_idx][0] + 1e-8)
        current_points[:, 7] = selected_points[:, 1] / (self.file_coord_max[file_idx][1] + 1e-8)
        current_points[:, 8] = selected_points[:, 2] / (self.file_coord_max[file_idx][2] + 1e-8)
        
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        
        selected_points[:, 3:6] = selected_points[:, 3:6] / 255.0
        
        current_points[:, 0:6] = selected_points

        current_labels = labels[selected_point_idxs]
        
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        
        return current_points.astype(np.float32), current_labels.astype(np.int32)

    def __len__(self):
        return len(self.file_idxs)

if __name__ == '__main__':
    data_root = '/home/y/NEW-Pointnet++/data/forest_output/'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = ForestNewDataset(
        split='train', 
        data_root=data_root, 
        num_point=num_point, 
        test_area=test_area, 
        block_size=block_size, 
        sample_rate=sample_rate, 
        transform=None,
        verbose=True
    )
    
    print('Point data size:', point_data.__len__())
    print('Point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('Point label 0 shape:', point_data.__getitem__(0)[1].shape)
    print('Point data sample:', point_data.__getitem__(0)[0][:2])
    print('Point label sample:', point_data.__getitem__(0)[1][:10])
    
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)

    train_loader = torch.utils.data.DataLoader(
        point_data, 
        batch_size=16, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        worker_init_fn=worker_init_fn
    )
    
    print("\nTesting DataLoader...")
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print(f'Batch {i+1}/{len(train_loader)} -- Input shape: {input.shape}, Target shape: {target.shape}, Time: {time.time() - end:.4f}s')
            end = time.time()
            if i == 2:
                break
        break
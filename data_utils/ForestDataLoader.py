import os
import numpy as np
import torch
from torch.utils.data import Dataset

def normalize_point_cloud(pc):
    
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def random_rotate(pc):
    
    theta = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    return pc @ rotation_matrix.T

def random_scale(pc, low=0.8, high=1.2):
    
    scale = np.random.uniform(low, high)
    return pc * scale

def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    
    jitter = np.clip(sigma * np.random.randn(*pc.shape), -clip, clip)
    return pc + jitter

def color_jitter(rgb, brightness=0.1):
    
    factor = np.random.uniform(1 - brightness, 1 + brightness, size=(1, 3))
    return np.clip(rgb * factor, 0, 1)

class ForestDataset(Dataset):
    def __init__(self, root, split='train', num_points=4096, augment=False, num_classes=4):
        self.root = root
        self.num_points = num_points
        self.augment = augment
        self.num_classes = num_classes

        self.data_dir = os.path.join(root, split)
        self.files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.txt')]

        self.points_list = []
        self.labels_list = []

        for file in self.files:
            data = np.loadtxt(file)
            if data.shape[1] < 7:
                continue

            xyz = data[:, 0:3]
            rgb = data[:, 3:6] / 255.0
            labels = data[:, 6].astype(np.int64)

            mask = ~np.isnan(xyz).any(axis=1)
            xyz, rgb, labels = xyz[mask], rgb[mask], labels[mask]

            valid_mask = (labels >= 0) & (labels < self.num_classes)
            xyz, rgb, labels = xyz[valid_mask], rgb[valid_mask], labels[valid_mask]

            if len(labels) == 0:
                continue

            self.points_list.append(np.concatenate([xyz, rgb], axis=1))
            self.labels_list.append(labels)

        all_labels = np.hstack(self.labels_list)
        classes, counts = np.unique(all_labels, return_counts=True)
        freq = counts.astype(np.float32) / np.sum(counts)
        self.labelweights = np.power(np.max(freq) / freq, 1 / 3.0)

        print(f"Loaded {len(self.files)} files")
        print("唯一标签值:", np.unique(all_labels))
        print("类别分布:", dict(zip(classes, counts)))

    def __len__(self):
        return len(self.points_list)

    def __getitem__(self, idx):
        pts = self.points_list[idx]
        labels = self.labels_list[idx]

        choice = np.random.choice(len(labels), self.num_points, replace=len(labels) < self.num_points)
        pts, labels = pts[choice, :], labels[choice]

        xyz, rgb = pts[:, 0:3], pts[:, 3:6]

        if self.augment:
            xyz = random_rotate(xyz)
            xyz = random_scale(xyz)
            xyz = jitter_point_cloud(xyz)
            rgb = color_jitter(rgb)

        xyz = normalize_point_cloud(xyz)

        pts = np.concatenate([xyz, rgb], axis=1)

        return torch.from_numpy(pts).float(), torch.from_numpy(labels).long()
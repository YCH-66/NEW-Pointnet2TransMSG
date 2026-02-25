import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset

class S3DISDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        
        print(f"数据根目录: {data_root}")
        print(f"目录是否存在: {os.path.exists(data_root)}")
        
        if not os.path.exists(data_root):
            raise ValueError(f"数据目录不存在: {data_root}")
        
        if split == 'train':
            data_dir = os.path.join(data_root, 'train')
        else:
            data_dir = os.path.join(data_root, 'test')
            
        if not os.path.exists(data_dir):
            data_dir = data_root
            print(f"警告: {split}目录不存在，使用根目录")
        
        rooms = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        rooms = sorted(rooms)
        
        print(f"找到{len(rooms)}个{split}文件")
        if len(rooms) > 0:
            print(f"前5个文件: {rooms[:5]}")
        
        if split == 'train':
            rooms_split = rooms[:int(0.8 * len(rooms))]
        else:
            rooms_split = rooms[int(0.8 * len(rooms)):]
            
        print(f"最终使用{len(rooms_split)}个文件作为{split}集")

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(4)

        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_dir, room_name)
            try:
                room_data = np.load(room_path)
                print(f"文件 {room_name} 形状: {room_data.shape}")
                
                if room_data.shape[1] < 7:
                    print(f"警告: 文件 {room_name} 列数不足，跳过")
                    continue
                    
                points, labels = room_data[:, 0:6], room_data[:, 6]
                
                tmp, _ = np.histogram(labels, range(5))
                labelweights += tmp
                
                coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
                self.room_points.append(points)
                self.room_labels.append(labels)
                self.room_coord_min.append(coord_min)
                self.room_coord_max.append(coord_max)
                num_point_all.append(labels.size)
                
            except Exception as e:
                print(f"加载文件 {room_name} 时出错: {e}")
                continue

        if len(self.room_points) == 0:
            raise ValueError("没有成功加载任何数据文件")
            
        if np.sum(labelweights) > 0:
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        else:
            self.labelweights = np.ones(4)
            
        print(f"标签权重: {self.labelweights}")
        
        sample_prob = np.array(num_point_all) / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]
        labels = self.room_labels[room_idx]
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        selected_points = points[selected_point_idxs, :]
        current_points = np.zeros((self.num_point, 9))
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)

class ScannetDatasetWholeScene():
    def __init__(self, root, block_points=4096, split='test', test_area=5, stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        
        print(f"ScannetDatasetWholeScene 初始化: root={root}, split={split}")
        
        assert split in ['train', 'test']
        
        if not os.path.exists(root):
            raise ValueError(f"数据根目录不存在: {root}")
        
        if split == 'train':
            data_dir = os.path.join(root, 'train')
            if not os.path.exists(data_dir):
                print(f"警告: train目录不存在，使用根目录")
                data_dir = root
        else:
            data_dir = os.path.join(root, 'test')
            if not os.path.exists(data_dir):
                data_dir = os.path.join(root, 'val')
                if not os.path.exists(data_dir):
                    print(f"警告: test和val目录都不存在，使用根目录")
                    data_dir = root
        
        print(f"实际数据目录: {data_dir}")
        
        if os.path.exists(data_dir):
            all_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
            print(f"在 {data_dir} 中找到 {len(all_files)} 个npy文件")
            
            if len(all_files) > 0:
                self.file_list = all_files
            else:
                self.file_list = []
        else:
            self.file_list = []
            
        print(f"{split}文件列表 ({len(self.file_list)}个): {self.file_list}")
        
        if len(self.file_list) == 0:
            raise ValueError(f"在 {data_dir} 中没有找到任何npy文件")
        
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min = []
        self.room_coord_max = []
        
        for file in self.file_list:
            file_path = os.path.join(data_dir, file)
            print(f"加载文件: {file_path}")
            
            try:
                data = np.load(file_path)
                print(f"文件 {file} 形状: {data.shape}")
                
                if data.shape[1] < 7:
                    print(f"警告: 文件 {file} 列数不足，需要至少7列 (xyzrgbl)，跳过")
                    continue
                
                points = data[:, :3]
                self.scene_points_list.append(data[:, :6])
                self.semantic_labels_list.append(data[:, 6].astype(np.int32))
                
                coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
                self.room_coord_min.append(coord_min)
                self.room_coord_max.append(coord_max)
                
                print(f"文件 {file} 加载成功, 点数: {len(points)}, 标签范围: {np.unique(data[:, 6])}")
                
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")
                continue
        
        if len(self.scene_points_list) == 0:
            raise ValueError("没有成功加载任何数据文件")
        
        labelweights = np.zeros(4)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(5))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        
        print(f"标签分布: {labelweights}")
        
        if np.sum(labelweights) > 0:
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        else:
            self.labelweights = np.ones(4)
        
        print(f"标签权重: {self.labelweights}")
        print(f"成功加载 {len(self.scene_points_list)} 个场景")

    def __getitem__(self, index):
        if index >= len(self.scene_points_list):
            raise IndexError(f"索引 {index} 超出范围，最大索引为 {len(self.scene_points_list)-1}")
            
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)

if __name__ == '__main__':
    data_root = 'data/forest_output/'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    try:
        point_data = S3DISDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
        print('point data size:', point_data.__len__())
        print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
        print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    except Exception as e:
        print(f"数据加载失败: {e}")
    
    try:
        whole_scene_data = ScannetDatasetWholeScene(root=data_root, split='test')
        print('whole scene data size:', whole_scene_data.__len__())
        if whole_scene_data.__len__() > 0:
            scene_data, scene_label, scene_weight, scene_idx = whole_scene_data[0]
            print('scene data shape:', scene_data.shape)
            print('scene label shape:', scene_label.shape)
    except Exception as e:
        print(f"整个场景数据加载失败: {e}")
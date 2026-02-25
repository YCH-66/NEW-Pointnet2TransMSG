import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

class PointNet2SemSeg(nn.Module):
    
    def __init__(self, num_classes: int, in_channel: int):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = int(in_channel)

        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.1, nsample=32,
            in_channel=self.in_channel, mlp=[32, 32, 64], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.2, nsample=32,
            in_channel=64, mlp=[64, 64, 128], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.4, nsample=32,
            in_channel=128, mlp=[128, 128, 256], group_all=False
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=16, radius=0.8, nsample=32,
            in_channel=256, mlp=[256, 256, 512], group_all=False
        )

        self.fp4 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 64,  mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + self.in_channel, mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1, bias=False)
        self.bn1   = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, points):
        
        l0_xyz    = points[:, :3, :]
        l0_points = points[:, 3:3+self.in_channel, :] if points.size(1) >= 3+self.in_channel else points[:, 3:, :]

        if self.training:
            assert l0_points.size(1) == self.in_channel, \
                f"in_channel mismatch: expect {self.in_channel}, got {l0_points.size(1)}"
        else:
            if l0_points.size(1) > self.in_channel:
                l0_points = l0_points[:, :self.in_channel, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2).contiguous()
        return x, None

def get_model(num_classes: int, in_channel: int = 3):
    
    return PointNet2SemSeg(num_classes=num_classes, in_channel=in_channel)

def get_loss(pred, target, trans_feat=None, weight=None):
    
    if pred.dim() == 3:
        pred = pred.reshape(-1, pred.size(-1))
    target = target.view(-1)
    return F.cross_entropy(pred, target, weight=weight, reduction='mean')

if __name__ == '__main__':
    m = get_model(num_classes=4, in_channel=3)
    x = torch.randn(2, 6, 2048)
    y, _ = m(x)
    print(y.shape)
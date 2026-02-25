import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import (
    PointNetSetAbstraction, PointNetFeaturePropagation, square_distance, index_points
)

def knn_idx(xyz, k):
    
    B, _, N = xyz.shape
    dists = square_distance(xyz.transpose(1, 2), xyz.transpose(1, 2))
    _, idx = torch.topk(-dists, k=k, dim=-1)
    return idx

class PosMLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=128, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 1), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, out_dim, 1)
        )
    def forward(self, xyz):
        return self.net(xyz)

class LocalPointTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, k=16, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.k = k
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Conv1d(dim, dim, 1, bias=False)
        self.kv = nn.Conv1d(dim, dim * 2, 1, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv1d(dim, dim, 1, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rel_pos = nn.Sequential(
            nn.Conv2d(3, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False)
        )

        self.norm1 = nn.BatchNorm1d(dim)
        self.norm2 = nn.BatchNorm1d(dim)
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, dim * 4, 1), nn.ReLU(inplace=True),
            nn.Conv1d(dim * 4, dim, 1)
        )

    def forward(self, x, xyz):
        
        B, C, N = x.shape
        H = self.num_heads
        d = C // H

        idx = knn_idx(xyz, self.k)

        q = self.q(x).view(B, H, d, N)
        kv = self.kv(x)
        k, v = kv[:, :C, :], kv[:, C:, :]
        k = k.view(B, H, d, N)
        v = v.view(B, H, d, N)

        def gather_knn(feat_hd_n, idx_bnk):
            feat_bn_hd = feat_hd_n.permute(0, 3, 1, 2).contiguous().view(B, N, H * d)
            gathered = index_points(feat_bn_hd, idx_bnk)
            gathered = gathered.view(B, N, self.k, H, d).permute(0, 3, 4, 1, 2).contiguous()
            return gathered

        k_knn = gather_knn(k, idx)
        v_knn = gather_knn(v, idx)

        attn_logits = (q.permute(0,1,3,2).unsqueeze(-1) * k_knn.permute(0,1,3,2,4)).sum(dim=3) * self.scale

        nbr_xyz = index_points(xyz.transpose(1, 2), idx)
        center_xyz = xyz.transpose(1, 2).unsqueeze(2)
        rel = (nbr_xyz - center_xyz).permute(0, 3, 2, 1)
        rel_bias = self.rel_pos(rel)
        rel_bias = rel_bias.view(B, H, d, self.k, N).permute(0,1,4,3,2).sum(-1)
        attn_logits = attn_logits + rel_bias

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)

        out = (v_knn * attn.unsqueeze(2)).sum(dim=-1)
        out = out.reshape(B, C, N)
        out = self.proj_drop(self.proj(out))

        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x

class PTBlock(nn.Module):
    def __init__(self, dim, depth=2, heads=4, k=16):
        super().__init__()
        self.blocks = nn.ModuleList([
            LocalPointTransformer(dim=dim, num_heads=heads, k=k) for _ in range(depth)
        ])
    def forward(self, feat, xyz):
        for blk in self.blocks:
            feat = blk(feat, xyz)
        return feat

class PointNet2SemSegTrans(nn.Module):
    
    def __init__(self, num_classes: int, in_channel: int, trans_k=16, trans_depth=(2,2), trans_heads=(4,4), pe_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = int(in_channel)

        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, in_channel=self.in_channel, mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(256,  0.2, 32, in_channel=64,             mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(64,   0.4, 32, in_channel=128,            mlp=[128, 128, 256], group_all=False)
        self.sa4 = PointNetSetAbstraction(16,   0.8, 32, in_channel=256,            mlp=[256, 256, 512], group_all=False)

        self.pe_l2 = PosMLP(in_dim=3, out_dim=128, hidden=64)
        self.pe_l3 = PosMLP(in_dim=3, out_dim=256, hidden=128)

        self.tr_l2 = PTBlock(dim=128, depth=trans_depth[0], heads=trans_heads[0], k=trans_k)
        self.tr_l3 = PTBlock(dim=256, depth=trans_depth[1], heads=trans_heads[1], k=trans_k)

        self.fp4 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 64,  mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + self.in_channel, mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1, bias=False)
        self.bn1   = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, points):
        
        l0_xyz, l0_points = points[:, :3, :], points[:, 3:, :]
        assert l0_points.size(1) == self.in_channel, \
            f"in_channel mismatch: expect {self.in_channel}, got {l0_points.size(1)}"

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l2_points = l2_points + self.pe_l2(l2_xyz)
        l2_points = self.tr_l2(l2_points, l2_xyz)

        l3_points = l3_points + self.pe_l3(l3_xyz)
        l3_points = self.tr_l3(l3_points, l3_xyz)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2).contiguous()
        return x, None

def get_model(num_classes: int, in_channel: int = 3, trans_k=16, trans_depth=(2,2), trans_heads=(4,4), pe_dim=128):
    return PointNet2SemSegTrans(
        num_classes=num_classes, in_channel=in_channel,
        trans_k=trans_k, trans_depth=trans_depth, trans_heads=trans_heads, pe_dim=pe_dim
    )

def get_loss(pred, target, trans_feat=None, weight=None):
    if pred.dim() == 3:
        pred = pred.reshape(-1, pred.size(-1))
    target = target.view(-1)
    return F.cross_entropy(pred, target, weight=weight, reduction='mean')
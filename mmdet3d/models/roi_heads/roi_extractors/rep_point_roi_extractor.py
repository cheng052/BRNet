import torch
from torch import nn as nn

from mmdet3d.ops import build_sa_module
from mmdet3d.core.bbox.structures.utils import rotation_3d_in_axis
from mmdet.models.builder import ROI_EXTRACTORS

@ROI_EXTRACTORS.register_module()
class RepPointRoIExtractor(nn.Module):
    def __init__(self,
                 rep_type=None,
                 density=None,
                 seed_feat_dim=None,
                 sa_cfg=dict(
                     type='PointSAModule',
                     pool_mod='max',
                     use_xyz=True,
                     normalize_xyz=True),
                 sa_radius=0.2,
                 sa_num_sample=16,
                 num_seed_points=1024):
        super(RepPointRoIExtractor, self).__init__()
        self.rep_type = rep_type
        self.density = density
        self.num_rep_points = density * 6 if rep_type == 'ray' else density ** 3

        self.seed_aggregation = build_sa_module(
            cfg=sa_cfg,
            num_point=num_seed_points,
            radius=sa_radius,
            num_sample=sa_num_sample,
            mlp_channels=[seed_feat_dim, 128, 64, 32],
            norm_cfg=dict(type='BN2d'))

        self.reduce_dim = torch.nn.Conv1d(self.num_rep_points * 32, 128, 1)

    def forward(self,
                seed_xyz: torch.Tensor,
                seed_features: torch.Tensor,
                proposals: torch.Tensor,
                ref_points: torch.Tensor,
                distance: torch.Tensor):
        batch_size, num_proposal, _ = distance.shape
        angle = proposals[..., -1]
        if self.rep_type == 'grid':
            # (B, N, num_rep_points, 3)
            rep_points = self._get_grid_based_rep_points(
                ref_points, distance, angle, grid_size=self.density)
        elif self.rep_type == 'ray':
            # (B, N, num_rep_points, 3)
            rep_points = self._get_ray_based_rep_points(
                ref_points, distance, angle, density=self.density)
        else:
            raise NotImplementedError(f'Representative points sampling method '
                                      f'{self.rep_type} not implemented')

        # (batch_size, num_proposal * num_rep_points, 3)
        rep_points = rep_points.view(batch_size, -1, 3)

        _, rep_features, _ = self.seed_aggregation(seed_xyz, seed_features, target_xyz=rep_points)
        # (batch_size, mlp[-1], num_proposal, num_rep_points)
        rep_features = rep_features.view(batch_size, -1, num_proposal, self.num_rep_points)
        # (batch_size, mlp[-1], num_rep_points, num_proposal)
        features = rep_features.transpose(2, 3)
        # (batch_size, mlp[-1]*num_rep_points, num_proposal)
        features = features.reshape(batch_size, -1, num_proposal)
        # (batch_size, 128, num_proposal)
        features = self.reduce_dim(features)

        return features

    def _get_ray_based_rep_points(self, center, rois, angle, density=2):
        batch_size, num_proposal, _ = rois.shape  # (B, N, 6)
        surface_points = torch.zeros((batch_size, num_proposal, 6, 3)).cuda()  # (B, N, 6, 3)
        surface_points[:, :, 0, 0] = rois[:, :, 0]  # front
        surface_points[:, :, 1, 1] = rois[:, :, 1]  # left
        surface_points[:, :, 2, 2] = rois[:, :, 2]  # top
        surface_points[:, :, 3, 0] = - rois[:, :, 3]  # back
        surface_points[:, :, 4, 1] = - rois[:, :, 4]  # right
        surface_points[:, :, 5, 2] = - rois[:, :, 5]  # bottom
        surface_points = surface_points.reshape(-1, 6, 3)  # (B*N, 6, 3)
        local_points = [surface_points*float(i/density) for i in range(density, 0, -1)]
        local_points = torch.stack(local_points, dim=1)  # (B*N, density, 6, 3)
        local_points = local_points.transpose(1,2).contiguous()  # (B*N, 6, density, 3)
        local_points = local_points.reshape(-1, self.num_rep_points, 3)  # (B*N, num_rep_points, 3)
        local_points_rotated = rotation_3d_in_axis(local_points, angle.view(-1), axis=2)
        center = center.reshape(batch_size*num_proposal, 1, 3)  # (B*N, num_rep_points, 3)
        global_points = local_points_rotated + center  # (B*N, num_rep_points, 3)
        # (B, N, num_rep_points, 3)
        global_points = global_points.reshape(batch_size, num_proposal, self.num_rep_points, 3)

        return global_points

    def _get_grid_based_rep_points(self, center, rois, angle, grid_size=3):
        B, N = angle.shape
        # (B*N, gs**3, 3)
        local_grid_points = self._get_dense_grid_points(rois, B*N, grid_size)
        # (B*N, gs**3, 3) ~ add rotation
        local_grid_points = rotation_3d_in_axis(local_grid_points, angle.view(-1), axis=2)
        # (B*N, gs**3, 3)
        global_roi_grid_points = local_grid_points + center.view(-1, 3).unsqueeze(dim=1)
        # (B, N, gs**3, 3)
        global_roi_grid_points = global_roi_grid_points.view(B, N, -1, 3)
        return global_roi_grid_points

    def _get_dense_grid_points(self, rois, bs, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))  # alis gs for grid_size
        dense_idx = faked_features.nonzero(as_tuple=False)  # (gs**3, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(bs, 1, 1).float()  # (bs, gs**3, 3)

        rois_center = rois[:, :, 3:6].view(-1, 3)  # (bs, 3)
        local_roi_size = rois[:, :, 0:3] + rois[:, :, 3:6]  # (batch_size, num_proposal, 3)
        local_roi_size = local_roi_size.view(-1, 3)  # (bs, 3)
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1)  # (bs, gs**3, 3)
        roi_grid_points = roi_grid_points - rois_center.unsqueeze(dim=1)  # (bs, gs**3, 3)
        return roi_grid_points

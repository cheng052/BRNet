import numpy as np
import torch
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core.bbox.structures import (DepthInstance3DBoxes,
                                          LiDARInstance3DBoxes,
                                          rotation_3d_in_axis)
from mmdet3d.models.builder import build_loss
from mmdet3d.models.losses import chamfer_distance
from mmdet.core import multi_apply
from mmdet.models import HEADS
from .vote_head import VoteHead

@HEADS.register_module()
class CAVoteHead(VoteHead):
    r"""
    Class agnostic vote head
    """
    def __init__(self,
                 num_classes,
                 bbox_coder,
                 train_cfg=None,
                 test_cfg=None,
                 vote_module_cfg=None,
                 vote_aggregation_cfg=None,
                 pred_layer_cfg=None,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 objectness_loss=None,
                 dir_class_loss=None,
                 dir_res_loss=None,
                 size_res_loss=None,
                 semantic_loss=None,
                 center_loss=None,
                 size_class_loss=None
                 ):
        super().__init__(
            num_classes,
            bbox_coder,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            vote_module_cfg=vote_module_cfg,
            vote_aggregation_cfg=vote_aggregation_cfg,
            pred_layer_cfg=pred_layer_cfg,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            objectness_loss=objectness_loss,
            center_loss=None,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_class_loss=None,
            size_res_loss=size_res_loss,
            semantic_loss=semantic_loss
        )

    def _get_cls_out_channels(self):
        if hasattr(self, 'semantic_loss'):
            return self.num_classes + 2
        else:
            return 2

    def _get_reg_out_channels(self):
        """Return the channel number of regression outputs."""
        # size regression (6)
        # heading class+residual (num_dir_bins*2)
        return 6 + self.num_dir_bins * 2

    @force_fp32(apply_to=('bbox_preds', ))
    def loss(self,
             bbox_preds,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             gt_bboxes_ignore=None,
             ret_target=False):
        targets = self.get_targets(points, gt_bboxes_3d, gt_labels_3d,
                                   pts_semantic_mask, pts_instance_mask, bbox_preds)

        (vote_targets, vote_target_masks, dir_class_targets,
        dir_res_targets, mask_targets, objectness_targets, objectness_weights,
        box_loss_weights, distance_targets, centerness_targets, dir_targets) = targets

        # calculate vote loss
        vote_loss = self.vote_module.get_loss(bbox_preds['seed_points'],
                                              bbox_preds['vote_points'],
                                              bbox_preds['seed_indices'],
                                              vote_target_masks, vote_targets)

        # calculate objectness loss
        objectness_loss = self.objectness_loss(
            bbox_preds['obj_scores'].transpose(2, 1),
            objectness_targets,
            weight=objectness_weights)

        # calculate distance loss
        size_res_loss = self.size_res_loss(
            bbox_preds['distance'],
            distance_targets,
            weight=box_loss_weights.unsqueeze(-1).repeat(1, 1, 6)
        )

        # calculate direction class loss
        dir_class_loss = self.dir_class_loss(
            bbox_preds['dir_class'].transpose(2, 1),
            dir_class_targets,
            weight=box_loss_weights)

        # calculate direction residual loss
        batch_size, proposal_num = dir_class_targets.shape[:2]
        heading_label_one_hot = vote_targets.new_zeros(
            (batch_size, proposal_num, self.num_dir_bins))
        heading_label_one_hot.scatter_(2, dir_class_targets.unsqueeze(-1), 1)
        dir_res_norm = torch.sum(
            bbox_preds['dir_res_norm'] * heading_label_one_hot, -1)
        dir_res_loss = self.dir_res_loss(
            dir_res_norm, dir_res_targets, weight=box_loss_weights)

        if hasattr(self, 'semantic_loss'):
            # calculate semantic loss
            semantic_loss = self.semantic_loss(
                bbox_preds['sem_scores'].transpose(2, 1),
                mask_targets,
                weight=box_loss_weights)

        losses = dict(
            vote_loss=vote_loss,
            objectness_loss=objectness_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_res_loss=size_res_loss)

        if hasattr(self, 'semantic_loss'):
            losses.update(semantic_loss=semantic_loss)

        if ret_target:
            losses['targets'] = targets

        return losses

    def get_targets(self,
                    points,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    bbox_preds=None):
        # find empty example
        valid_gt_masks = list()
        gt_num = list()
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)
                valid_gt_masks.append(gt_labels_3d[index].new_zeros(1))
                gt_num.append(1)
            else:
                valid_gt_masks.append(gt_labels_3d[index].new_ones(
                    gt_labels_3d[index].shape))
                gt_num.append(gt_labels_3d[index].shape[0])
        max_gt_num = max(gt_num)

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for i in range(len(gt_labels_3d))]
            pts_instance_mask = [None for i in range(len(gt_labels_3d))]

        aggregated_points = [
            bbox_preds['aggregated_points'][i]
            for i in range(len(gt_labels_3d))
        ]

        (vote_targets, vote_target_masks, size_res_targets,
         dir_class_targets, dir_res_targets, centerness_targets,
         mask_targets, objectness_targets, objectness_masks,
         distance_targets, centerness_targets, dir_targets) = multi_apply(
            self.get_targets_single, points, gt_bboxes_3d, gt_labels_3d,
            pts_semantic_mask, pts_instance_mask, aggregated_points
        )

        vote_targets = torch.stack(vote_targets)
        vote_target_masks = torch.stack(vote_target_masks)

        objectness_targets = torch.stack(objectness_targets)
        objectness_weights = torch.stack(objectness_masks)
        objectness_weights /= (torch.sum(objectness_weights) + 1e-6)
        box_loss_weights = objectness_targets.float() / (
            torch.sum(objectness_targets).float() + 1e-6)
        dir_class_targets = torch.stack(dir_class_targets)
        dir_res_targets = torch.stack(dir_res_targets)
        dir_targets = torch.stack(dir_targets)
        mask_targets = torch.stack(mask_targets)
        distance_targets = torch.stack(distance_targets)
        centerness_targets = torch.stack(centerness_targets)

        return (vote_targets, vote_target_masks, dir_class_targets,
                dir_res_targets, mask_targets, objectness_targets, objectness_weights,
                box_loss_weights, distance_targets, centerness_targets,
                dir_targets)

    def get_targets_single(self,
                           points,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           pts_semantic_mask=None,
                           pts_instance_mask=None,
                           aggregated_points=None):
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)

        # generate votes target
        num_points = points.shape[0]
        if self.bbox_coder.with_rot:
            vote_targets = points.new_zeros([num_points, 3 * self.gt_per_seed])
            vote_target_masks = points.new_zeros([num_points],
                                                 dtype=torch.long)
            vote_target_idx = points.new_zeros([num_points], dtype=torch.long)
            box_indices_all = gt_bboxes_3d.points_in_boxes(points)
            for i in range(gt_labels_3d.shape[0]):
                box_indices = box_indices_all[:, i]
                indices = torch.nonzero(
                    box_indices, as_tuple=False).squeeze(-1)
                selected_points = points[indices]
                vote_target_masks[indices] = 1
                vote_targets_tmp = vote_targets[indices]
                votes = gt_bboxes_3d.gravity_center[i].unsqueeze(
                    0) - selected_points[:, :3]

                for j in range(self.gt_per_seed):
                    column_indices = torch.nonzero(
                        vote_target_idx[indices] == j,
                        as_tuple=False).squeeze(-1)
                    vote_targets_tmp[column_indices,
                                     int(j * 3):int(j * 3 +
                                                    3)] = votes[column_indices]
                    if j == 0:
                        vote_targets_tmp[column_indices] = votes[
                            column_indices].repeat(1, self.gt_per_seed)

                vote_targets[indices] = vote_targets_tmp
                vote_target_idx[indices] = torch.clamp(
                    vote_target_idx[indices] + 1, max=2)
        elif pts_semantic_mask is not None:
            vote_targets = points.new_zeros([num_points, 3])
            vote_target_masks = points.new_zeros([num_points],
                                                 dtype=torch.long)

            for i in torch.unique(pts_instance_mask):
                indices = torch.nonzero(
                    pts_instance_mask == i, as_tuple=False).squeeze(-1)
                if pts_semantic_mask[indices[0]] < self.num_classes:
                    selected_points = points[indices, :3]
                    center = 0.5 * (
                        selected_points.min(0)[0] + selected_points.max(0)[0])
                    vote_targets[indices, :] = center - selected_points
                    vote_target_masks[indices] = 1
            vote_targets = vote_targets.repeat((1, self.gt_per_seed))
        else:
            raise NotImplementedError

        (center_targets, size_targets, dir_class_targets,
         dir_res_targets, dir_targets) = self.bbox_coder.encode(
            gt_bboxes_3d, gt_labels_3d, ret_dir_target=True)

        proposal_num = aggregated_points.shape[0]
        distance1, _, assignment, _ = chamfer_distance(
            aggregated_points.unsqueeze(0),
            center_targets.unsqueeze(0),
            reduction='none')
        assignment = assignment.squeeze(0)
        euclidean_distance1 = torch.sqrt(distance1.squeeze(0) + 1e-6)

        objectness_masks = points.new_zeros((proposal_num))
        objectness_masks[
            euclidean_distance1 < self.train_cfg['pos_distance_thr']] = 1.0
        objectness_masks[
            euclidean_distance1 > self.train_cfg['neg_distance_thr']] = 1.0

        center_targets = center_targets[assignment]
        dir_class_targets = dir_class_targets[assignment]
        dir_res_targets = dir_res_targets[assignment]
        dir_res_targets /= (np.pi / self.num_dir_bins)
        size_res_targets = size_targets[assignment]
        dir_targets = dir_targets[assignment]

        mask_targets = gt_labels_3d[assignment]

        # Centerness loss targets
        canonical_xyz = aggregated_points - center_targets
        # print(canonical_xyz.shape)
        # print(gt_bboxes_3d.yaw[assignment].shape)
        if self.bbox_coder.with_rot:
            canonical_xyz = rotation_3d_in_axis(
                canonical_xyz.unsqueeze(0).transpose(0, 1),
                -gt_bboxes_3d.yaw[assignment], 2).squeeze(1)

        distance_front  = size_res_targets[:, 0] - canonical_xyz[:, 0]
        distance_left   = size_res_targets[:, 1] - canonical_xyz[:, 1]
        distance_top    = size_res_targets[:, 2] - canonical_xyz[:, 2]
        distance_back   = size_res_targets[:, 0] + canonical_xyz[:, 0]
        distance_right  = size_res_targets[:, 1] + canonical_xyz[:, 1]
        distance_bottom = size_res_targets[:, 2] + canonical_xyz[:, 2]

        distance_targets = torch.cat(
            (distance_front.unsqueeze(-1),
             distance_left.unsqueeze(-1),
             distance_top.unsqueeze(-1),
             distance_back.unsqueeze(-1),
             distance_right.unsqueeze(-1),
             distance_bottom.unsqueeze(-1)),
            dim=-1
        )
        inside_mask = (distance_targets >= 0.).all(dim=-1)

        objectness_targets = points.new_zeros((proposal_num), dtype=torch.long)
        pos_mask = (euclidean_distance1 < self.train_cfg['pos_distance_thr']) & inside_mask
        objectness_targets[pos_mask] = 1

        distance_targets.clamp_(min=0)
        deltas = torch.cat(
            (distance_targets[:, 0:3, None], distance_targets[:, 3:6, None]),
            dim=-1
        )
        nominators = deltas.min(dim=-1).values.prod(dim=-1)
        denominators = deltas.max(dim=-1).values.prod(dim=-1) + 1e-6
        centerness_targets = (nominators / denominators + 1e-6) ** (1/3)
        centerness_targets = torch.clamp(centerness_targets, min=0, max=1)

        return (
            vote_targets, vote_target_masks, size_res_targets,
            dir_class_targets, dir_res_targets, centerness_targets, mask_targets.long(),
            objectness_targets, objectness_masks, distance_targets, centerness_targets,
            dir_targets
        )

    def get_bboxes(self,
                   points,
                   bbox_preds,
                   input_metas,
                   rescale=False,
                   use_nms=True):
        if hasattr(self, 'semantic_loss'):
            return super(CAVoteHead, self).get_bboxes(
                points, bbox_preds, input_metas, rescale=rescale, use_nms=use_nms
            )
        else:
            # decode boxes
            bbox3d = self.bbox_coder.decode(bbox_preds, mode='rpn')
            assert not use_nms
            return bbox3d














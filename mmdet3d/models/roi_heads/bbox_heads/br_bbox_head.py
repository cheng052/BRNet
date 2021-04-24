import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.utils.dir_refine_loss import get_dir_refine_loss
from mmdet3d.core.post_processing import aligned_3d_nms
from mmdet3d.models.builder import build_loss
from mmdet3d.models.dense_heads.base_conv_bbox_head import BaseConvBboxHead
from mmdet.core import build_bbox_coder
from mmdet.models import HEADS


@HEADS.register_module()
class BRBboxHead(nn.Module):
    def __init__(self,
                 num_classes,
                 bbox_coder,
                 train_cfg=None,
                 test_cfg=None,
                 pred_layer_cfg=None,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 dir_res_loss=None,
                 size_res_loss=None,
                 semantic_loss=None
                 ):
        super(BRBboxHead, self).__init__()
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.dir_res_loss = build_loss(dir_res_loss)
        self.size_res_loss = build_loss(size_res_loss)
        self.semantic_loss = build_loss(semantic_loss)

        self.bbox_coder = build_bbox_coder(bbox_coder)

        # Bbox classification and regression
        self.conv_pred = BaseConvBboxHead(
            **pred_layer_cfg,
            num_cls_out_channels=self._get_cls_out_channels(),
            num_reg_out_channels=self._get_reg_out_channels())

    def _get_cls_out_channels(self):
        return self.num_classes

    def _get_reg_out_channels(self):
        # distance residual(6),
        # heading residual(1)
        return 1 + 6

    def forward(self, feats_dict):
        cls_predictions, reg_predictions = self.conv_pred(feats_dict['fused_feats'])
        bbox_preds = self.bbox_coder.split_refined_pred(
            feats_dict, cls_predictions, reg_predictions)

        return bbox_preds

    def loss(self,
             feats_dict,
             bbox_preds):
        targets = self.get_targets(feats_dict)
        (mask_targets, distance_targets, dir_targets, box_loss_weights) = targets

        # calculate distance refine loss
        size_refine_loss = self.size_res_loss(
            bbox_preds['refined_distance'],
            distance_targets,
            weight=box_loss_weights.unsqueeze(-1).repeat(1, 1, 6)
        )

        # calculate direction refine loss
        if not self.bbox_coder.with_rot:
            dir_refine_loss = self.dir_res_loss(
                bbox_preds['refined_angle'],
                dir_targets,
                weight=box_loss_weights
            )
        else:
            dir_refine_loss = get_dir_refine_loss(
                bbox_preds['refined_angle'],
                dir_targets,
                self.bbox_coder.num_dir_bins
            )
            dir_refine_loss = torch.sum(dir_refine_loss * box_loss_weights)


        # calculate semantic loss
        semantic_loss = self.semantic_loss(
            bbox_preds['sem_scores'].transpose(2, 1),
            mask_targets,
            weight=box_loss_weights
        )

        losses = dict(
            size_refine_loss=size_refine_loss,
            dir_refine_loss=dir_refine_loss,
            semantic_loss=semantic_loss
        )

        return losses

    def get_targets(self, feats_dict):
        (vote_targets, vote_target_masks, dir_class_targets, dir_res_targets,
         mask_targets, objectness_targets, objectness_weights, box_loss_weights,
         distance_targets, centerness_targets, dir_targets) = feats_dict['targets']

        return (mask_targets, distance_targets, dir_targets, box_loss_weights)

    def get_bboxes(self,
                   points,
                   bbox_preds,
                   input_metas,
                   rescale=False,
                   use_nms=True):
        # decode boxes
        obj_scores = F.softmax(bbox_preds['obj_scores'], dim=-1)[..., -1]
        sem_scores = F.softmax(bbox_preds['sem_scores'], dim=-1)
        bbox3d = self.bbox_coder.decode(bbox_preds, mode='rcnn')  # [center, bbox_size, dir_angle]

        if use_nms:
            batch_size = bbox3d.shape[0]
            results = list()
            for b in range(batch_size):
                bbox_selected, score_selected, labels = \
                    self.multiclass_nms_single(obj_scores[b], sem_scores[b],
                                               bbox3d[b], points[b, ..., :3],
                                               input_metas[b])
                bbox = input_metas[b]['box_type_3d'](
                    bbox_selected,
                    box_dim=bbox_selected.shape[-1],
                    with_yaw=self.bbox_coder.with_rot)
                results.append((bbox, score_selected, labels))

            return results
        else:
            return bbox3d

    def multiclass_nms_single(self, obj_scores, sem_scores, bbox, points,
                              input_meta):
        bbox = input_meta['box_type_3d'](
            bbox,
            box_dim=bbox.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 0.5))
        box_indices = bbox.points_in_boxes(points)

        corner3d = bbox.corners
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        nonempty_box_mask = box_indices.T.sum(1) > 5

        bbox_classes = torch.argmax(sem_scores, -1)
        nms_selected = aligned_3d_nms(minmax_box3d[nonempty_box_mask],
                                      obj_scores[nonempty_box_mask],
                                      bbox_classes[nonempty_box_mask],
                                      self.test_cfg.nms_thr)

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores > self.test_cfg.score_thr)
        nonempty_box_inds = torch.nonzero(
            nonempty_box_mask, as_tuple=False).flatten()
        nonempty_mask = torch.zeros_like(bbox_classes).scatter(
            0, nonempty_box_inds[nms_selected], 1)
        selected = (nonempty_mask.bool() & scores_mask.bool())

        if self.test_cfg.per_class_proposal:
            bbox_selected, score_selected, labels = [], [], []
            for k in range(sem_scores.shape[-1]):
                bbox_selected.append(bbox[selected].tensor)
                score_selected.append(obj_scores[selected] *
                                      sem_scores[selected][:, k])
                labels.append(
                    torch.zeros_like(bbox_classes[selected]).fill_(k))
            bbox_selected = torch.cat(bbox_selected, 0)
            score_selected = torch.cat(score_selected, 0)
            labels = torch.cat(labels, 0)
        else:
            bbox_selected = bbox[selected].tensor
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]

        return bbox_selected, score_selected, labels

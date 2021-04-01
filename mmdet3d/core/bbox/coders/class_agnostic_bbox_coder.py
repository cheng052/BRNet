import numpy as np
import torch

from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet.core.bbox.builder import BBOX_CODERS
from .anchor_free_bbox_coder import AnchorFreeBBoxCoder

@BBOX_CODERS.register_module()
class ClassAgnosticBBoxCoder(AnchorFreeBBoxCoder):
    def __init__(self, num_dir_bins, with_rot):
        super(ClassAgnosticBBoxCoder, self).__init__(
            num_dir_bins, with_rot=with_rot
        )

    def encode(self, gt_bboxes_3d, gt_labels_3d, ret_dir_target=False):
        # generate center target
        center_target = gt_bboxes_3d.gravity_center

        # generate dir target
        size_res_target = gt_bboxes_3d.dims / 2

        # generate dir target
        box_num = gt_labels_3d.shape[0]

        if self.with_rot:
            (dir_class_target,
             dir_res_target) = self.angle2class(gt_bboxes_3d.yaw)
            dir_target = gt_bboxes_3d.yaw
        else:
            dir_class_target = gt_labels_3d.new_zeros(box_num)
            dir_res_target = gt_bboxes_3d.tensor.new_zeros(box_num)
            dir_target = gt_bboxes_3d.tensor.new_zeros(box_num)

        if ret_dir_target:
            return (center_target, size_res_target, dir_class_target,
                    dir_res_target, dir_target)
        else:
            return (center_target, size_res_target, dir_class_target,
                    dir_res_target)

    def decode(self, bbox_out, mode='rpn'):
        assert mode in ['rpn', 'rcnn']
        prefix = 'refined_' if mode == 'rcnn' else ''

        distance = bbox_out[prefix+'distance']  # (B, N, 6)
        batch_size, num_proposal, _ = distance.shape

        if self.with_rot:
            if mode == 'rpn':
                dir_class = torch.argmax(bbox_out['dir_class'], -1).detach()
                dir_res = torch.gather(bbox_out['dir_res'], -1,
                                       dir_class.unsqueeze(-1))
                dir_res.squeeze_(-1)  # (batch_size, num_proposal)
                dir_angle = self.class2angle(dir_class, dir_res).reshape(
                    batch_size, num_proposal, 1)
            elif mode == 'rcnn':
                dir_angle = bbox_out[prefix+'angle'].reshape(
                    batch_size, num_proposal, 1)
            else:
                raise NotImplementedError
        else:
            dir_angle = distance.new_zeros(batch_size, num_proposal, 1)

        dir_angle = dir_angle % (2*np.pi)

        # decode bbox size
        bbox_size = distance[..., 0:3] + distance[..., 3:6]
        bbox_size = torch.clamp(bbox_size, min=0.1)

        # decode bbox center
        canonical_xyz = (distance[..., 3:6] -
                         distance[..., 0:3]) / 2  # (batch_size, num_proposal, 3)

        shape = canonical_xyz.shape

        canonical_xyz = rotation_3d_in_axis(
            canonical_xyz.view(-1, 3).unsqueeze(1),
            dir_angle.view(-1),
            axis=2
        ).squeeze(1).view(shape)

        ref_points = bbox_out['ref_points']
        center = ref_points - canonical_xyz

        bbox3d = torch.cat([center, bbox_size, dir_angle], dim=-1)
        return bbox3d


    def split_pred(self, cls_preds, reg_preds, ref_points):
        results = {}
        start, end = 0, 0

        cls_preds_trans = cls_preds.transpose(2, 1)
        reg_preds_trans = reg_preds.transpose(2, 1)

        with_sem = True if cls_preds_trans.shape[-1] > 2 else False

        # decode distance
        end += 6
        # (batch_size, num_proposal, 6)
        results['distance'] = reg_preds_trans[..., start:end].exp().contiguous()
        start = end

        # decode directions
        end += self.num_dir_bins
        results['dir_class'] = reg_preds_trans[..., start:end].contiguous()
        start = end

        end += self.num_dir_bins
        dir_res_norm = reg_preds_trans[..., start:end].contiguous()
        start = end

        results['dir_res_norm'] = dir_res_norm
        results['dir_res'] = dir_res_norm * (np.pi / self.num_dir_bins)

        # decode objectness scores
        start = 0
        end = 2
        results['obj_scores'] = cls_preds_trans[..., start:end].contiguous()
        start = end

        # decode semantic scores
        if with_sem:
            results['sem_scores'] = cls_preds_trans[..., start:].contiguous()

        results['ref_points'] = ref_points  # (batch_size, num_proposal, 3)

        return results

    def split_refined_pred(self, feats_dict, cls_preds, reg_preds):
        results = {}

        cls_preds_trans = cls_preds.transpose(2, 1)  # (B, N, num_classes)
        reg_preds_trans = reg_preds.transpose(2, 1)  # (B, N, ...)

        # 1. Get proposals from roi head
        distance = feats_dict['distance']  # (B, N, 6)
        angle = feats_dict['proposal_list'][..., -1]  # (B, N)

        # 2. Get refined proposals
        # (B, N, num_classes)
        results['sem_scores'] = cls_preds_trans.contiguous()

        start, end = 0, 0
        end += 1
        # (B, N)
        angle_delta = reg_preds_trans[..., :end].squeeze(-1)

        start = end
        # (B, N, 6)
        distance_delta = reg_preds_trans[..., start:].contiguous()

        results['refined_distance'] = distance + distance_delta  # (B, N, 6)
        refined_angle = angle + angle_delta
        # refined_angle[refined_angle > np.pi] -= 2 * np.pi
        refined_angle = refined_angle % (2*np.pi)
        results['refined_angle'] = refined_angle  # (B, N)

        # 3. Merge from rpn proposals
        results.update(obj_scores=feats_dict['obj_scores'])  # (B, N, 2)
        results.update(ref_points=feats_dict['aggregated_points'])  # (B, N, 3)

        return results

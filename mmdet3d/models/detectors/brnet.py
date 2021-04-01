import torch
from mmdet.models import DETECTORS
from .two_stage import TwoStage3DDetector


@DETECTORS.register_module()
class BRNet(TwoStage3DDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
    ):
        super(BRNet, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained
        )

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_bboxes_ignore=None):
        points_cat = torch.stack(points)

        feats_dict = self.extract_feat(points_cat)

        losses = dict()
        if self.with_rpn:
            rpn_outs = self.rpn_head(feats_dict, self.train_cfg.rpn.sample_mod)
            feats_dict.update(rpn_outs)

            rpn_loss_inputs = (points, gt_bboxes_3d, gt_labels_3d,
                               pts_semantic_mask, pts_instance_mask, img_metas)
            rpn_losses = self.rpn_head.loss(
                rpn_outs,
                *rpn_loss_inputs,
                gt_bboxes_ignore=gt_bboxes_ignore,
                ret_target=True
            )
            feats_dict['targets'] = rpn_losses.pop('targets')
            losses.update(rpn_losses)

            # Generate rpn proposals
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            proposal_inputs = (points, rpn_outs, img_metas)
            proposal_list = self.rpn_head.get_bboxes(
                *proposal_inputs, use_nms=proposal_cfg.use_nms
            )
            feats_dict['proposal_list'] = proposal_list
        else:
            raise NotImplementedError

        roi_losses = self.roi_head.forward_train(
            feats_dict, img_metas, points,
            gt_bboxes_3d, gt_labels_3d,
            pts_semantic_mask,
            pts_instance_mask,
            gt_bboxes_ignore
        )
        losses.update(roi_losses)

        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=None):
        points_cat = torch.stack(points)

        feats_dict = self.extract_feat(points_cat)

        if self.with_rpn:
            proposal_cfg = self.test_cfg.rpn
            rpn_outs = self.rpn_head(feats_dict, proposal_cfg.sample_mod)
            feats_dict.update(rpn_outs)
            # Generate rpn proposals
            proposal_list = self.rpn_head.get_bboxes(
                points, rpn_outs, img_metas, use_nms=proposal_cfg.use_nms)
            feats_dict['proposal_list'] = proposal_list
        else:
            raise NotImplementedError

        return self.roi_head.simple_test(
            feats_dict, img_metas, points_cat)

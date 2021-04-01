import torch

from mmdet3d.core.bbox import bbox3d2result
from mmdet.models import HEADS
from ..builder import build_head, build_roi_extractor
from .base_3droi_head import Base3DRoIHead

@HEADS.register_module()
class BRRoIHead(Base3DRoIHead):
    def __init__(self,
                 roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None
    ):
        super(BRRoIHead, self).__init__(
            bbox_head=bbox_head, train_cfg=train_cfg, test_cfg=test_cfg
        )
        self.roi_extractor = build_roi_extractor(roi_extractor)

    def init_weights(self, pretrained):
        pass

    def init_mask_head(self):
        pass

    def init_bbox_head(self, bbox_head):
        bbox_head['train_cfg'] = self.train_cfg
        bbox_head['test_cfg'] = self.test_cfg
        self.bbox_head = build_head(bbox_head)

    def init_assigner_sampler(self):
        pass

    def forward_train(self,
                      feats_dict,
                      img_metas,
                      points,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_semantic_mask,
                      pts_instance_mask,
                      gt_bboxes_ignore=None):

        bbox_preds = self._bbox_forward(feats_dict)
        feats_dict.update(bbox_preds)
        losses = self.bbox_head.loss(feats_dict, bbox_preds)

        return losses

    def simple_test(self,
                    feats_dict,
                    img_metas,
                    points):
        bbox_preds = self._bbox_forward(feats_dict)
        feats_dict.update(bbox_preds)

        bbox_list = self.bbox_head.get_bboxes(
            points, bbox_preds, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def _bbox_forward(self, feats_dict):
        # (batch_size, 128, num_proposal)
        pooled_feats = self.roi_extractor(
            feats_dict['seed_points'],
            feats_dict['seed_features'],
            feats_dict['proposal_list'],
            feats_dict['aggregated_points'],
            feats_dict['distance']
        )

        # (batch_size, 128+128, num_proposal)
        fused_feats = torch.cat(
            (feats_dict['aggregated_features'], pooled_feats),
            dim=1
        )
        feats_dict.update(fused_feats=fused_feats)

        bbox_preds = self.bbox_head(feats_dict)

        return bbox_preds



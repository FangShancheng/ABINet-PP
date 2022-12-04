from typing import Dict
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import ROI_HEADS_REGISTRY
from adet.utils.comm import CrossEntropyLoss
from ..poolers import TopPooler
from .model import Model
from .attention import PositionAttention, PositionContentAttention
from .feat_seq_modeling import Transformer, ResNet


@ROI_HEADS_REGISTRY.register()
class PCATextHead(Model):
    """
    TextHead performs text region alignment and recognition.
    
    It is a simplified ROIHeads, only ground truth RoIs are
    used during training.
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        Args:
            in_channels (int): number of channels of the input feature
        """
        super(PCATextHead, self).__init__(cfg)
        # fmt: off
        pooler_resolution = cfg.MODEL.BATEXT.POOLER_RESOLUTION
        pooler_scales     = cfg.MODEL.BATEXT.POOLER_SCALES
        sampling_ratio    = cfg.MODEL.BATEXT.SAMPLING_RATIO
        canonical_size    = cfg.MODEL.BATEXT.CANONICAL_SIZE
        self.in_features  = cfg.MODEL.BATEXT.IN_FEATURES
        self.voc_size     = cfg.MODEL.BATEXT.VOC_SIZE
        seq_modeling      = cfg.MODEL.ABINET.VISION_SEQ_MODELING
        attention         = cfg.MODEL.ABINET.VISION_ATTENTION
        d_model           = cfg.MODEL.FPN.OUT_CHANNELS
        d_attn            = cfg.MODEL.ABINET.VISION_ATTN_DIM
        attn_iter_size    = cfg.MODEL.ABINET.VISION_ITER_SIZE
        num_modeling      = cfg.MODEL.ABINET.VISION_NUM_MODELING
        max_len           = cfg.MODEL.BATEXT.NUM_CHARS
        soft_ce           = cfg.MODEL.ABINET.SOFT_CE
        self.yield_proposal = cfg.MODEL.FCOS.YIELD_PROPOSAL
        # fmt: on

        self.pooler = TopPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type="BezierAlign",
            canonical_box_size=canonical_size,
            canonical_level=3,
            assign_crit="bezier")

        if seq_modeling == 'transformer':
            self.modeling = Transformer(d_model, num_layers=num_modeling)
        elif seq_modeling == 'resnet':
            self.modeling = ResNet(num_layers=num_modeling, coord=False)
        elif seq_modeling == 'none':
            self.modeling = nn.Identity()           
        else:
            raise Exception(f'Wrong type of vision sequence modeling [{seq_modeling}]')

        if attention == 'pa':
            self.attention = PositionAttention(max_len, d_model, num_channels=d_attn)
        elif attention == 'pca':
            self.attention = PositionContentAttention(max_len, d_model, num_channels=d_attn,
                             num_layers=num_modeling, iter_size=attn_iter_size)
        else:
            raise Exception(f'Wrong type of vision sequence modeling [{attention}]')

        self.cls = nn.Linear(d_model, self.voc_size + 1)
        self.ce = CrossEntropyLoss(soft_ce)

    def forward(self, images, features, proposals, targets=None):
        """
        see detectron2.modeling.ROIHeads
        """
        del images
        features = [features[f] for f in self.in_features]
        if self.training:
            beziers = [p.beziers for p in targets]
            targets = torch.cat([x.text for x in targets], dim=0)
        else:
            proposals = proposals['proposals'] if self.yield_proposal else proposals
            beziers = [p.top_feat for p in proposals]
        bezier_features = self.pooler(features, beziers)

        if bezier_features.size(0) <= 0:
            for box in proposals:
                box.beziers = box.top_feat
                box.recs = box.top_feat
            return proposals, {}, {}

        bezier_features = self.modeling(bezier_features)
        attn_vecs, attn_scores = self.attention(bezier_features)
        if isinstance(self.attention, PositionContentAttention):
            all_attn_vecs = attn_vecs
            attn_vecs = attn_vecs[-1]

        prediction = {}
        logits = self.cls(attn_vecs) # (N, T, C)
        prediction["logits"] = logits
        prediction["feature"] = attn_vecs
        prediction["pt_lengths"] = self._get_length(prediction["logits"])
        prediction["attn_scores"] = attn_scores

        if self.training:
            if isinstance(self.attention, PositionContentAttention):
                losses_list = []
                for vecs in all_attn_vecs:
                    logits_ = self.cls(vecs)
                    loss = self.ce(logits_, targets)
                    losses_list.append(loss)
                losses = {'loss_vision': sum(losses_list[1:]), 
                          'loss_query': losses_list[0]}
            else:
                loss_vision = self.ce(logits, targets)
                losses = {'loss_vision': loss_vision}
            return None, losses, prediction
        else:
            preds = logits.argmax(-1)
            start_ind = 0
            for proposals_per_im in proposals:
                end_ind = start_ind + len(proposals_per_im)
                proposals_per_im.recs = preds[start_ind:end_ind]
                proposals_per_im.beziers = proposals_per_im.top_feat
                start_ind = end_ind

            return proposals, {}, prediction

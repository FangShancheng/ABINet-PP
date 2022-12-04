import torch
import torch.nn as nn
from adet.utils.comm import CrossEntropyLoss

from .model import Model


class ABIAlignment(Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        l_d_model = cfg.MODEL.ABINET.LANGUAGE_DIM_MODEL
        v_d_model = cfg.MODEL.FPN.OUT_CHANNELS
        voc_size = cfg.MODEL.BATEXT.VOC_SIZE
        soft_ce = cfg.MODEL.ABINET.SOFT_CE
        self.loss_weight = cfg.MODEL.ABINET.ALIGNMENT_LOSS_WEIGHT
        self.max_length = cfg.MODEL.BATEXT.NUM_CHARS
        self.w_att = nn.Linear(2 * l_d_model, l_d_model)
        self.cls = nn.Linear(l_d_model, voc_size + 1)
        self.ce = CrossEntropyLoss(soft_ce)
        if l_d_model != v_d_model:
            self.proj = nn.Linear(v_d_model, l_d_model)
        else:
            self.proj = None

    def forward(self, l_feature, v_feature, gt_instances=None):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and d is dim of model
            v_feature: (N, T, E) shape the same as l_feature 
            l_lengths: (N,)
            v_lengths: (N,)
        """
        if self.proj is not None:
            v_feature = self.proj(v_feature)
        f = torch.cat((l_feature, v_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)
        prediction =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths}

        losses = None
        if self.training:
            targets = torch.cat([x.text for x in gt_instances], dim=0)
            loss_alignment = self.ce(logits, targets)
            losses = {'loss_alignment': self.loss_weight * loss_alignment}

        return losses, prediction

import torch
import torch.nn as nn

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from .model_vision import ABIVision
from .model_language import ABILanguage
from .model_alignment import ABIAlignment

@META_ARCH_REGISTRY.register()
class ABINetModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vision = ABIVision(cfg)
        self.language = ABILanguage(cfg)
        self.alignment = ABIAlignment(cfg)

    def forward(self, batched_inputs):
        results, v_res = self.vision(batched_inputs)
        if len(v_res) <= 0:  # None detection
            return results

        gt_instances = None
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.vision.device) for x in batched_inputs]

        v_tokens = torch.softmax(v_res['logits'], dim=-1)
        v_lengths = v_res['pt_lengths']
        language_inputs = {'tokens': v_tokens, 'lengths': v_lengths, 'gt_instances': gt_instances}
        l_losses, l_res = self.language(language_inputs)

        l_feature, v_feature = l_res['feature'], v_res['feature']
        a_losses, a_res = self.alignment(l_feature, v_feature, gt_instances)

        if self.training:
            results.update(l_losses)
            results.update(a_losses)
            return results
        else:
            preds = a_res['logits'].argmax(-1)
            start_ind = 0
            for proposals_per_im in results:
                end_ind = start_ind + len(proposals_per_im['instances'])
                proposals_per_im['instances'].recs = preds[start_ind:end_ind]
                start_ind = end_ind
            return results

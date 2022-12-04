import torch
import torch.nn as nn

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from .model_vision import ABIVision
from .model_language import ABILanguage
from .model_alignment import ABIAlignment

@META_ARCH_REGISTRY.register()
class ABINetIterModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.iter_size = cfg.MODEL.ABINET.ITER_SIZE
        self.model_eval = cfg.MODEL.ABINET.MODEL_EVAL
        assert self.model_eval in ['alignment', 'vision']
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

        a_res = v_res
        l_losses, a_losses = [], []
        for _ in range(self.iter_size):
            tokens = torch.softmax(a_res['logits'], dim=-1)
            lengths = a_res['pt_lengths']
            language_inputs = {'tokens': tokens, 'lengths': lengths, 'gt_instances': gt_instances}
            l_loss, l_res = self.language(language_inputs)
            l_losses.append(l_loss)

            l_feature, v_feature = l_res['feature'], v_res['feature']
            a_loss, a_res = self.alignment(l_feature, v_feature, gt_instances)
            a_losses.append(a_loss)

        if self.training:
            l_losses = sum([l['loss_language'] for l in l_losses]) / self.iter_size
            a_losses = sum([l['loss_alignment'] for l in a_losses]) / self.iter_size
            results.update({'loss_language': l_losses})
            results.update({'loss_alignment': a_losses})
            return results
        else:
            if self.model_eval == 'vision':
                preds = v_res['logits'].argmax(-1)
                logits = v_res['logits']
                pt_lengths = v_res['pt_lengths']                
            else:
                preds = a_res['logits'].argmax(-1)
                logits = a_res['logits']
                pt_lengths = a_res['pt_lengths']
            attn_scores = v_res['attn_scores'] 
            start_ind = 0
            for proposals_per_im in results:
                end_ind = start_ind + len(proposals_per_im['instances'])
                proposals_per_im['instances'].recs = preds[start_ind:end_ind]
                proposals_per_im['instances'].logits = logits[start_ind:end_ind]
                proposals_per_im['instances'].pt_lengths = pt_lengths[start_ind:end_ind]
                proposals_per_im['instances'].attn_scores = attn_scores[start_ind:end_ind]
                start_ind = end_ind
            return results

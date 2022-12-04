import logging
import torch
import torch.nn as nn

from .model import Model
from .transformer import (PositionalEncoding, 
                          TransformerDecoder,
                          TransformerDecoderLayer)

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from adet.data.dataset_language import build_train_loader_text
from adet.utils.comm import get_world_size
from adet.utils.comm import CrossEntropyLoss

@META_ARCH_REGISTRY.register()
class ABILanguage(Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.language_only = cfg.MODEL.ABINET.LANGUAGE_ONLY
        d_model = cfg.MODEL.ABINET.LANGUAGE_DIM_MODEL
        d_inner = cfg.MODEL.ABINET.LANGUAGE_DIM_INNER
        self.nhead = cfg.MODEL.ABINET.LANGUAGE_NUM_HEAD
        self.loss_weight = cfg.MODEL.ABINET.LANGUAGE_LOSS_WEIGHT
        self.aug_training = cfg.MODEL.ABINET.LANGUAGE_AUG_TRAINING
        self.max_num_instances = cfg.MODEL.ABINET.MAX_INS_PER_BATCH // get_world_size()
        num_layers = cfg.MODEL.ABINET.LANGUAGE_NUM_LAYER
        voc_size = cfg.MODEL.BATEXT.VOC_SIZE
        self.max_length = cfg.MODEL.BATEXT.NUM_CHARS
        soft_ce = cfg.MODEL.ABINET.SOFT_CE
        self.proj = nn.Linear(voc_size + 1, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerDecoderLayer(d_model, self.nhead, d_inner, dropout=0.1, 
                activation='relu', self_attn=False)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, voc_size + 1)
        self.ce = CrossEntropyLoss(soft_ce)
        if self.training and self.aug_training:  # TODO: still work during inference.
            self.data_loader = build_train_loader_text(cfg, 
                    bs=self.max_num_instances, one_hot_y=soft_ce, num_workers=2)

        if cfg.MODEL.ABINET.LANGUAGE_CHECKPOINT is not None:
            logger = logging.getLogger("adet.trainer")
            logger.info(f'Read language model from {cfg.MODEL.ABINET.LANGUAGE_CHECKPOINT}.')
            self.load(cfg.MODEL.ABINET.LANGUAGE_CHECKPOINT)

    def load(self, source, device=None, strict=True):
        state = torch.load(source, map_location=device)
        self.load_state_dict(state['model'], strict=strict)

    @property
    def device(self):
        return self.cls.weight.device

    def get_location_mask(self, sz, length, prob=0.1, device=None):
        bs = length.size(0)
        location_mask = self._get_location_mask(sz, device)
        disturbed_mask = self._get_random_mask(sz, bs, prob, device)
        visible_mask = self._get_visible_mask(sz, length, device)
        disturbed_mask = torch.where(visible_mask, torch.tensor(0., device=device), disturbed_mask)
        return location_mask.unsqueeze(0) + disturbed_mask

    def forward(self, language_inputs):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        tokens = language_inputs['tokens']
        lengths = language_inputs['lengths']
        gt_instances = language_inputs['gt_instances']
        bs = tokens.size(0)
        if self.language_only:
            tokens = tokens.to(self.device)
            lengths = lengths.to(self.device)
            gt_instances = gt_instances.to(self.device)

        if self.training and self.aug_training:
            data = next(self.data_loader)
            at_bs = max(self.max_num_instances - bs, 0)
            at_tokens = data['tokens'].to(self.device)[:at_bs]
            at_lengths = data['lengths'].to(self.device)[:at_bs]
            tokens = torch.cat([tokens, at_tokens])
            lengths = torch.cat([lengths, at_lengths])

        tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        lengths = lengths.clamp_(2, self.max_length)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)
        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)
        prediction =  {'feature': output[:bs], 
                       'logits': logits[:bs], 
                       'pt_lengths': pt_lengths[:bs]}

        if self.language_only:
            loss_language = self.ce(logits, gt_instances)
            losses = {'loss_language': self.loss_weight * loss_language}
            return losses

        losses = None
        if self.training:
            targets = torch.cat([x.text for x in gt_instances], dim=0)
            if self.aug_training:
                at_target = data['gt_instances'].to(self.device)[:at_bs]
                targets = torch.cat([targets, at_target])
            loss_language = self.ce(logits, targets)
            losses = {'loss_language': self.loss_weight * loss_language}

        return losses, prediction

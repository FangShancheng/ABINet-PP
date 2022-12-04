import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.max_length = cfg.MODEL.BATEXT.NUM_CHARS + 1
        self.null_label = cfg.MODEL.BATEXT.VOC_SIZE #  TODO: pay attention
    
    def _get_length(self, logit, dim=-1):
        """ Greed decoder to obtain length from logit"""
        out = (logit.argmax(dim=-1) == self.null_label)
        abn = out.any(dim)
        out = ((out.cumsum(dim) == 1) & out).max(dim)[1]
        out = out + 1  # additional end token
        out = torch.where(abn, out, out.new_tensor(logit.shape[1]))
        return out

    @staticmethod
    def _get_padding_mask(length, max_length):
        length = length.unsqueeze(-1)
        grid = torch.arange(0, max_length, device=length.device).unsqueeze(0)
        return grid >= length

    @staticmethod
    def _get_square_subsequent_mask(sz, device, diagonal=0, fw=True):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device), diagonal=diagonal) == 1)
        if fw: mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @staticmethod
    def _get_location_mask(sz, device=None):
        mask = torch.eye(sz, device=device)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask

    @staticmethod
    def _get_random_mask(sz, bs, prob=0.1, device=None):
        mask = torch.rand(bs, sz, sz, device=device) <= prob
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask

    @staticmethod
    def _get_visible_mask(sz, length, device=None):
        # generate a visible mask with text length to avoid all-unvisible locations
        bs = length.size(0)
        idx = torch.arange(0, sz, device=device).expand(bs, sz)
        offset = torch.stack([torch.randint(1, l.item(), (sz,), device=device) for l in length])
        idx = (idx + offset) % length[:, None]
        src = torch.ones(bs, sz, 1, device=device).bool()
        mask = torch.zeros(bs, sz, sz, device=device).bool().scatter(2, idx[:,:,None], src)
        return mask
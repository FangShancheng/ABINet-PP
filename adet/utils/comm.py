from re import S
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from detectron2.utils.comm import get_world_size
import cv2
import shutil
import numpy as np
import torchvision.utils as vutils
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import colors
from torchvision import transforms
from pathlib import Path
from adet.layers.bezier_align import BezierAlign
from detectron2.modeling.poolers import convert_boxes_to_pooler_format
from adet.structures import Beziers
from torch.optim import Optimizer
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping

def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def reduce_mean(tensor):
    num_gpus = get_world_size()
    total = reduce_sum(tensor)
    return total / num_gpus


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations

def onehot(label, depth, device=None):
    """ 
    Args:
        label: shape (n1, n2, ..., )
        depth: a scalar

    Returns:
        onehot: (n1, n2, ..., depth)
    """
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label, device=device)
    onehot = torch.zeros(label.size() + torch.Size([depth]), device=device)
    onehot = onehot.scatter_(-1, label.unsqueeze(-1), 1)

    return onehot

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, softmax=True):
        if softmax: log_prob = F.log_softmax(input, dim=-1)
        else: log_prob = torch.log(input)
        loss = -(target * log_prob).sum(dim=-1)
        if self.reduction == "mean": return loss.mean()
        elif self.reduction == "sum": return loss.sum()
        else: return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self, soft_ce=False):
        super().__init__()
        self.soft_ce = soft_ce
        self.ce = SoftCrossEntropyLoss() if self.soft_ce else nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        if self.soft_ce:
            loss = self.ce(inputs, targets)
        else:
            loss = self.ce(inputs.permute(0,2,1), targets.long())
        return loss 

def blend_mask(image, mask, alpha=0.5, cmap='jet', color='b', color_alpha=1.0):
    # normalize mask
    mask = (mask-mask.min()) / (mask.max() - mask.min() + np.finfo(float).eps)
    if mask.shape != image.shape:
        mask = cv2.resize(mask,(image.shape[1], image.shape[0]))
    # get color map
    color_map = plt.get_cmap(cmap)
    mask = color_map(mask)[:,:,:3]
    # convert float to uint8
    mask = (mask * 255).astype(dtype=np.uint8)

    # set the basic color
    basic_color = np.array(colors.to_rgb(color)) * 255 
    basic_color = np.tile(basic_color, [image.shape[0], image.shape[1], 1]) 
    basic_color = basic_color.astype(dtype=np.uint8)
    # blend with basic color
    blended_img = cv2.addWeighted(image, color_alpha, basic_color, 1-color_alpha, 0)
    # blend with mask
    blended_img = cv2.addWeighted(blended_img, alpha, mask, 1-alpha, 0)

    return blended_img


def build_optimizer(cfg, optim, base_lr, model):
    if optim == 'SGD':
        params = get_default_optimizer_params(
            model,
            base_lr=base_lr,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )
        return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
            params,
            lr=base_lr,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif optim == 'Adam':
        params = get_default_optimizer_params(
            model,
            base_lr=base_lr,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )
        return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
            params,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=0.,
        )
    else:
        raise Exception('Wrong type of Optimizer!')


# TODO: write different lr in tensorboard and commandline
class AutonomousOptimizer(Optimizer):
    
    def __init__(self, v_optim, l_optim, a_optim):
        self.v_optim = v_optim
        self.l_optim = l_optim
        self.a_optim = a_optim

    @property
    def param_groups(self):
        return self.v_optim.param_groups + self.l_optim.param_groups + self.a_optim.param_groups

    def state_dict(self):
        v_state = self.v_optim.state_dict()
        l_state = self.l_optim.state_dict()
        a_state = self.a_optim.state_dict()
        return {
            'v_state': v_state['state'], 'v_param_groups': v_state['param_groups'],
            'l_state': l_state['state'], 'l_param_groups': l_state['param_groups'],
            'a_state': a_state['state'], 'a_param_groups': a_state['param_groups'],
        }

    def load_state_dict(self, state_dict):
        v_state = {'state': state_dict['v_state'], 'param_groups': state_dict['v_param_groups']}
        l_state = {'state': state_dict['l_state'], 'param_groups': state_dict['l_param_groups']}
        a_state = {'state': state_dict['a_state'], 'param_groups': state_dict['a_param_groups']}
        self.v_optim.load_state_dict(v_state)
        self.l_optim.load_state_dict(l_state)
        self.a_optim.load_state_dict(a_state)

    def zero_grad(self):
        self.v_optim.zero_grad()
        self.l_optim.zero_grad()
        self.a_optim.zero_grad()
    
    def step(self):
        self.v_optim.step()
        self.l_optim.step()
        self.a_optim.step()

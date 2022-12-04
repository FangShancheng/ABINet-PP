import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import PositionalEncoding
from .feat_seq_modeling import Transformer


def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))

def decoder_layer(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))


class PositionAttention(nn.Module):
    def __init__(self, max_length, in_channels=512, num_channels=64, 
                 mode='nearest', s_start=(1, 2), **kwargs):
        super().__init__()
        self.max_length = max_length
        self.num_channels = num_channels
        self.mode = mode
        self.align_corners = None if mode=='nearest' else True

        if num_channels > 0:
            self.k_encoder = nn.Sequential(
                encoder_layer(in_channels, num_channels, s=s_start),
                encoder_layer(num_channels, num_channels, s=(2, 2)),
                encoder_layer(num_channels, num_channels, s=(2, 2)),
                encoder_layer(num_channels, num_channels, s=(2, 2))
            )
            self.k_decoder = nn.Sequential(
                decoder_layer(num_channels, num_channels),
                decoder_layer(num_channels, num_channels),
                decoder_layer(num_channels, num_channels),
                decoder_layer(num_channels, in_channels)
            )

        self.pos_encoder = PositionalEncoding(in_channels, dropout=0, max_len=max_length)
        self.project = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        N, E, H, W = x.size()
        k, v = x, x  # (N, E, H, W)

        if self.num_channels > 0:
            # calculate key vector
            features = []
            for i in range(0, len(self.k_encoder)):
                k = self.k_encoder[i](k)
                features.append(k)
            for i in range(0, len(self.k_decoder) - 1):
                pre = features[len(self.k_decoder) - 2 - i]
                k = F.interpolate(k, pre.shape[2:4], None, self.mode, self.align_corners)
                k = self.k_decoder[i](k)
                k = k + pre
            # TODO: Combine
            k = F.interpolate(k, x.shape[2:4], None, self.mode, self.align_corners)
            k = self.k_decoder[-1](k)

        # calculate query vector
        zeros = x.new_zeros((self.max_length, N, E))  # (T, N, E)
        q = self.pos_encoder(zeros)  # (T, N, E)
        q = q.permute(1, 0, 2)  # (N, T, E)
        q = self.project(q)  # (N, T, E)
        
        # calculate attention
        attn_scores = torch.bmm(q, k.flatten(2, 3))  # (N, T, (H*W))
        attn_scores = attn_scores / (E ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)

        v = v.permute(0, 2, 3, 1).view(N, -1, E)  # (N, (H*W), E)
        attn_vecs = torch.bmm(attn_scores, v)  # (N, T, E)

        return attn_vecs, attn_scores.view(N, -1, H, W)


class PositionContentAttention(nn.Module):
    def __init__(self, max_length, in_channels=512, num_channels=64, 
                 mode='nearest', num_layers=4, iter_size=1, **kwargs):
        super().__init__()
        self.max_length = max_length
        self.num_channels = num_channels
        self.iter_size = iter_size
        self.mode = mode
        self.align_corners = None if mode=='nearest' else True

        if num_channels > 0:
            self.k_encoder = nn.Sequential(
                encoder_layer(in_channels, num_channels, s=(2, 2)),
                encoder_layer(num_channels, num_channels, s=(2, 2)),
                encoder_layer(num_channels, num_channels, s=(2, 2)),
                encoder_layer(num_channels, num_channels, s=(1, 1))
            )
            self.k_decoder = nn.Sequential(
                decoder_layer(num_channels, num_channels),
                decoder_layer(num_channels, num_channels),
                decoder_layer(num_channels, num_channels),
                decoder_layer(num_channels, in_channels)
            )

        self.aggregator = Transformer(num_channels, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(in_channels, dropout=0, max_len=max_length)
        self.project = nn.Linear(in_channels, self.max_length * in_channels)

    def forward(self, x):
        N, E, H, W = x.size()
        k, v = x, x  # (N, E, H, W)

        if self.num_channels > 0:
            # calculate key vector
            features = []
            for i in range(0, len(self.k_encoder)):
                k = self.k_encoder[i](k)
                features.append(k)

            k = self.aggregator(k)
            for i in range(0, len(self.k_decoder) - 1):
                pre = features[len(self.k_decoder) - 2 - i]
                k = F.interpolate(k, pre.shape[2:4], None, self.mode, self.align_corners)
                k = self.k_decoder[i](k)
                k = k + pre
            # TODO: Combine
            k = F.interpolate(k, x.shape[2:4], None, self.mode, self.align_corners)
            k = self.k_decoder[-1](k)

        q0 = x.mean(dim=(2, 3))  # or use k as the initial vector
        q0 = self.project(q0)
        q0 = q0.view(N, self.max_length, -1)
        all_attn_vecs = [q0]

        v = v.permute(0, 2, 3, 1).view(N, -1, E)  # (N, (H*W), E)
        for _ in range(self.iter_size):
            q = all_attn_vecs[-1]
            q = q.permute(1, 0, 2)
            q = self.pos_encoder(q)
            q = q.permute(1, 0, 2)
            
            # calculate attention
            attn_scores = torch.bmm(q, k.flatten(2, 3))  # (N, T, (H*W))
            attn_scores = attn_scores / (E ** 0.5)
            attn_scores = torch.softmax(attn_scores, dim=-1)

            attn_vecs = torch.bmm(attn_scores, v)  # (N, T, E)
            all_attn_vecs.append(attn_vecs)

        return all_attn_vecs, attn_scores.view(N, -1, H, W)

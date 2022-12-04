import torch
import torch.nn as nn

from .transformer import (PositionalEncoding,
                          TransformerEncoder,
                          TransformerEncoderLayer)


def conv_layer(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))

class ResNet(nn.Module):
    def __init__(self, d_model=256, num_layers=3, coord=False):
        super().__init__()
        self.coord = coord
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0 and coord:
                layer = conv_layer(d_model + 2, d_model)
            else:
                layer = conv_layer(d_model, d_model)
            self.layers.append(layer)

    def encode_coord(self, feature):
        x_range = torch.linspace(-1, 1, feature.shape[-1], device=feature.device)
        y_range = torch.linspace(-1, 1, feature.shape[-2], device=feature.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([feature.shape[0], 1, -1, -1])
        x = x.expand([feature.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        feature = torch.cat([feature, coord_feat], 1)
        return feature

    def forward(self, x):
        if self.coord == True:
            x = self.encode_coord(x)
        for layer in self.layers:
            x = layer(x)
        return x


_default_tfmer_cfg = dict(d_model=512, nhead=8, d_inner=2048, # 1024
                          dropout=0.1, activation='relu')

class Transformer(nn.Module):
    def __init__(self, d_model=256, num_layers=3, d_inner=None):
        super().__init__()

        self.d_model = d_model
        nhead = _default_tfmer_cfg['nhead']
        d_inner = d_inner if d_inner is not None else _default_tfmer_cfg['d_inner']
        dropout = _default_tfmer_cfg['dropout']
        activation = _default_tfmer_cfg['activation']

        self.pos_encoder = PositionalEncoding(self.d_model)
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, 
                dim_feedforward=d_inner, dropout=dropout, activation=activation)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

    def forward(self, feature):
        if feature.dim() == 4:
            n, c, h, w = feature.shape
            feature = feature.view(n, c, -1).permute(2, 0, 1)
            feature = self.pos_encoder(feature)
            feature = self.transformer(feature)
            feature = feature.permute(1, 2, 0).view(n, c, h, w)
            return feature
        else:
            feature = feature.permute(2, 0, 1)
            feature = self.pos_encoder(feature)
            feature = self.transformer(feature)
            feature = feature.permute(1, 2, 0)
            return feature

import math
import torch
import einops
import numpy as np

from numpy import random as np_random
from abc import ABC, abstractmethod
from pero_pretraining.models.helpers import create_vgg_encoder, create_pero_vgg_layers


class TransformerEncoder(ABC, torch.nn.Module):
    def __init__(self, height=40, patch_size=(40, 8), in_channels=3, model_dim=512, num_heads=4, num_blocks=6,
                 feedforward_dim=2048, dropout=0.0, max_len=4096, *args, **kwargs):
        super(TransformerEncoder, self).__init__()

        self.height = height
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout
        self.max_len = max_len

        self.position_model = PositionalEncoding(self.model_dim, self.max_len)
        self.encoder_layers = self.create_layers()        
        self.intermediate_norm = torch.nn.LayerNorm(self.model_dim, eps=1e-05)
        # create mask pattern as random noise, but generated with the same seed
        np.random.seed(42)
        mask_tile = np.random.rand(1, self.in_channels, self.patch_size[0], self.patch_size[1])
        mask_tile = torch.tensor(mask_tile, dtype=torch.float32)

        self.mask_pattern = einops.repeat(mask_tile, 'n c h w -> n c h (w x)', x=1024).to("cuda")

    def create_layers(self):
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.model_dim,
                                                         nhead=self.num_heads,
                                                         dim_feedforward=self.feedforward_dim,
                                                         dropout=self.dropout)

        encoder_layers = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.num_blocks)
        return encoder_layers
    
    def forward(self, x, mask=None):
        if mask is not None:
            x = self.mask(x, mask)

        x = self.encode(x)

        return x
    
    def mask(self, x: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
        # x has shape (N, C, H, W)
        # mask has shape (N, W/8) and dtype int64
        # expand mask to shape (N, C, H, W/8)
        mask = torch.tensor(mask).to(x.device)
        mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.in_channels, self.height, -1)
        # stretch mask to shape (N, C, H, W)
        mask = einops.repeat(mask, 'n c h w -> n c h (s w)', s=self.patch_size[1])

        mask_pattern = self.mask_pattern.expand(x.shape[0], -1, -1, -1)

        x[mask == 1] = mask_pattern[:, :, :, :x.shape[3]][mask == 1]

        return x

    def encode(self, x):
        x = self._conv(x)
        #print(1, x.shape)
        x = self._transformer(x)
        #print(2, x.shape)

        return x

    @abstractmethod
    def _conv(self, x):
        pass

    def _transformer(self, x):
        x = einops.rearrange(x, 'n d s -> s n d')
        x = self.intermediate_norm(x)
        x = self.position_model(x)
        x = self.encoder_layers(x)
        x = einops.rearrange(x, 's n d -> n d s')

        return x


class VisionTransformerEncoder(TransformerEncoder):
    def __init__(self, height=40, patch_size=(40, 8), in_channels=3, model_dim=512, num_heads=4, num_blocks=6,
                 feedforward_dim=2048, dropout=0.0, *args, **kwargs):
        super(VisionTransformerEncoder, self).__init__(height=height, patch_size=patch_size, in_channels=in_channels, 
                                                       model_dim=model_dim, num_heads=num_heads, num_blocks=num_blocks,
                                                       feedforward_dim=feedforward_dim, dropout=dropout, *args, **kwargs)

        self.conv_layer = torch.nn.Conv2d(in_channels=self.in_channels,
                                          out_channels=self.model_dim,
                                          kernel_size=self.patch_size,
                                          stride=self.patch_size)


    def _conv(self, x):
        x = self.conv_layer(x)
        x = einops.rearrange(x, 'n c h w -> n (c h) w')

        return x


class VggTransformerEncoder(TransformerEncoder):
    def __init__(self, height=40, patch_size=(40, 8), in_channels=3, model_dim=512, num_heads=4, num_blocks=6,
                 feedforward_dim=2048, dropout=0.0, base_channels=64, num_conv_blocks=4, pretrained_vgg_layers=17,
                 use_pero_vgg=True, *args, **kwargs):
        super(VggTransformerEncoder, self).__init__(height=height, patch_size=patch_size, in_channels=in_channels,
                                                    model_dim=model_dim, num_heads=num_heads, num_blocks=num_blocks,
                                                    feedforward_dim=feedforward_dim, dropout=dropout, *args, **kwargs)

        self.base_channels = base_channels
        self.num_conv_blocks = num_conv_blocks
        self.pretrained_vgg_layers = pretrained_vgg_layers

        if use_pero_vgg:
            self.conv_layers = create_pero_vgg_layers(dropout=self.dropout)
        else:
            self.conv_layers = create_vgg_encoder(in_channels=self.in_channels,
                                                  num_conv_blocks=self.num_conv_blocks,
                                                  base_channels=self.base_channels,
                                                  patch_size=self.patch_size,
                                                  pretrained_vgg_layers=self.pretrained_vgg_layers,
                                                  dropout=self.dropout,
                                                  num_conv_layers=[2, 2, 3, 2])

        conv_layers_vertical_subsampling = 2 ** self.num_conv_blocks
        aggregation_height = self.height // conv_layers_vertical_subsampling

        conv_layers_out_channels = self.base_channels * (2 ** (self.num_conv_blocks - 1))

        self.aggregation = torch.nn.Sequential(
            torch.nn.Conv2d(conv_layers_out_channels, self.model_dim, kernel_size=(aggregation_height, 1), stride=1,
                            padding=0),
            torch.nn.LeakyReLU()
        )

    def _conv(self, x):
        x = self.conv_layers(x)
        x = self.aggregation(x)
        x = einops.rearrange(x, 'n c h w -> n (c h) w')

        return x


class PositionalEncoding(torch.nn.Module):
    """
    Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, max_len=1024, random_shift=True, **kwargs):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.random_shift = random_shift

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe, persistent=False)
        # self.pe.shape is [max_len, 1, d_model]

    def forward(self, x):
        if self.random_shift and self.training:
            seq_len, batch_size, d_model = x.size()
            max_shift = self.pe.size(0) - seq_len

            if max_shift == 0:
                positional_encoding = self.pe[:x.size(0), :, :]
            else:
                offsets = torch.randint(0, max_shift, (batch_size,), device=x.device)
                positional_encoding = torch.zeros(seq_len, batch_size, d_model, device=x.device)
                for i, offset in enumerate(offsets):
                    positional_encoding[:, i, :] = self.pe[offset:offset+seq_len, 0, :]
        else:
            positional_encoding = self.pe[:x.size(0), :, :]

        return x + positional_encoding

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, max_len={self.max_len}, random_shift={self.random_shift}"


def main():
    N = 16
    C = 3
    H = 40
    W = 1024
    S = 8
    P = 0.2

    x = torch.rand((N, C, H, W))
    vit = VisionTransformerEncoder(height=H, patch_size=(H, S), in_channels=C)
    vggt = VggTransformerEncoder(height=H, patch_size=(H, S), in_channels=C)

    mask = torch.zeros((N, W // S))
    mask[torch.rand_like(mask) < P] = 1
    mask = mask.long()

    vit_y = vit(x)
    vit_y_mask = vit(x, mask)

    vggt_y = vggt(x)
    vggt_y_mask = vggt(x, mask)

    print(vit)
    print()
    print(vggt)
    print()

    print(f"Input: {x.shape}")
    print(f"Vision Transformer: {vit_y.shape}")
    print(f"Vision Transformer with mask: {vit_y_mask.shape}")
    print(f"VGG Transformer: {vggt_y.shape}")
    print(f"VGG Transformer with mask: {vggt_y_mask.shape}")


if __name__ == '__main__':
    main()

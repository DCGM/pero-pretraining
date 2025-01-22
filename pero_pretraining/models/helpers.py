import torch


def create_vgg_encoder(in_channels, num_conv_blocks, base_channels, patch_size, pretrained_vgg_layers, dropout, num_conv_layers):
    layers = []
    in_channels = in_channels
    current_subsampling = [1, 1]

    for i in range(num_conv_blocks):
        out_channels = base_channels * (2 ** i)

        block_subsampling = [1, 1]
        if current_subsampling[0] < patch_size[0]:
            block_subsampling[0] = 2
            current_subsampling[0] *= 2

        if current_subsampling[1] < patch_size[1]:
            block_subsampling[1] = 2
            current_subsampling[1] *= 2

        num_conv_layers_in_block = num_conv_layers[i]
        batch_norm = i == num_conv_blocks - 1

        layers += create_encoder_block(in_channels, out_channels, num_conv_layers_in_block, block_subsampling, dropout, batch_norm)
        in_channels = out_channels

    layers = torch.nn.Sequential(*layers)

    if pretrained_vgg_layers > 0:
        import torchvision
        pretrained_vgg = torchvision.models.vgg16(pretrained=True)
        pretrained_vgg = pretrained_vgg.features[:pretrained_vgg_layers]

        layers.load_state_dict(pretrained_vgg.state_dict(), strict=False)

    return layers


def create_encoder_block(in_channels, out_channels, num_conv_layers, subsampling, dropout, batch_norm):
    block_layers = []

    for _ in range(num_conv_layers):
        block_layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        block_layers.append(torch.nn.ReLU())
        in_channels = out_channels

    additional_layers = [torch.nn.MaxPool2d(kernel_size=subsampling, stride=subsampling)]

    if batch_norm:
        additional_layers.append(torch.nn.BatchNorm2d(out_channels))

    additional_layers.append(torch.nn.Dropout(dropout))

    layers = torch.nn.Sequential(*block_layers, torch.nn.Sequential(*additional_layers))

    return layers


def create_vgg_decoder(out_channels, num_conv_blocks, base_channels, patch_size, dropout, num_conv_layers, upsampling):
    layers = []

    current_in_channels = base_channels
    for i in range(num_conv_blocks):
        current_out_channels = current_in_channels // 2
        upsampling_factor = (2.0, 2.0)
        num_conv_layers_in_block = num_conv_layers[i]
        layers += create_decoder_block(current_in_channels, current_out_channels, num_conv_layers_in_block,
                                       dropout=dropout, upsampling_mode=upsampling, upsampling_factor=upsampling_factor)

        current_in_channels = current_out_channels

    layers.append(torch.nn.Conv2d(in_channels=current_in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                  stride=(1, 1), padding=(1, 1)))

    layers = torch.nn.Sequential(*layers)

    return layers


def create_decoder_block(in_channels, out_channels, num_conv_layers, dropout, upsampling_factor, upsampling_mode='bilinear'):
    block_layers = []

    for i in range(num_conv_layers - 1):
        block_layers.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        block_layers.append(torch.nn.ReLU(inplace=True))

    block_layers.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
    block_layers.append(torch.nn.ReLU(inplace=True))
    block_layers.append(torch.nn.Dropout(p=dropout))
    block_layers.append(torch.nn.Upsample(scale_factor=upsampling_factor, mode=upsampling_mode))

    layers = torch.nn.Sequential(*block_layers)

    return layers


def create_pero_vgg_layers():
    from torch.nn import Conv2d, ReLU, MaxPool2d, Dropout, LeakyReLU, BatchNorm2d

    layers = torch.nn.Sequential(*[
        Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False),
        Dropout(p=0.0, inplace=True),
        Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False),
        Dropout(p=0.0, inplace=True),
        Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False),
        Dropout(p=0.0, inplace=True),
        torch.nn.Sequential(*[
            Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            LeakyReLU(negative_slope=0.01),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            LeakyReLU(negative_slope=0.01),
            MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)]),
        BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        Dropout(p=0.0, inplace=True)])

    return layers


def create_pero_vgg_encoder(out_channels=768, aggregation_height=3):
    from torch.nn import Conv2d, LeakyReLU

    class EncoderLayers(torch.nn.Module):
        def __init__(self, layers):
            super(EncoderLayers, self).__init__()
            self.blocks_2d = layers

        def forward(self, x):
            return self.blocks_2d(x)

    class EncoderFrontend(torch.nn.Module):
        def __init__(self, blocks_2d, aggregation_conv):
            super(EncoderFrontend, self).__init__()
            self.blocks_2d = blocks_2d
            self.aggregation_conv = aggregation_conv

        def forward(self, x):
            x = self.blocks_2d(x)
            x = self.aggregation_conv(x)
            return x

    class Encoder(torch.nn.Module):
        def __init__(self, encoder_frontend):
            super(Encoder, self).__init__()
            self.encoder_frontend = encoder_frontend

        def forward(self, x):
            return self.encoder_frontend(x)

    layers = create_pero_vgg_layers()

    aggregation_conv = torch.nn.Sequential(*[
        Conv2d(512, out_channels, kernel_size=(aggregation_height, 1), stride=(1, 1)),
        LeakyReLU(negative_slope=0.01)])

    blocks_2d = EncoderLayers(layers)
    encoder_frontend = EncoderFrontend(blocks_2d, aggregation_conv)
    encoder = Encoder(encoder_frontend)

    return encoder


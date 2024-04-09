import torch

from pero_pretraining.models.helpers import create_vgg_encoder, create_vgg_decoder


class VGGEncoder(torch.nn.Module):
    def __init__(self, height=40, patch_size=(40, 8), in_channels=3, dropout=0.0, base_channels=64, num_conv_blocks=3,
                 num_conv_layers=(2, 2, 3), pretrained_vgg_layers=17, aggregation="conv"):
        super().__init__()
        self.height = height
        self.aggregation = aggregation

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_conv_blocks = num_conv_blocks
        self.num_conv_layers = num_conv_layers
        self.patch_size = patch_size
        self.dropout = dropout
        self.pretrained_vgg_layers = pretrained_vgg_layers

        self.encoder = create_vgg_encoder(in_channels=self.in_channels,
                                          num_conv_blocks=self.num_conv_blocks,
                                          base_channels=self.base_channels,
                                          patch_size=self.patch_size,
                                          pretrained_vgg_layers=self.pretrained_vgg_layers,
                                          dropout=self.dropout,
                                          num_conv_layers=self.num_conv_layers)

        conv_layers_subsampling = 2 ** self.num_conv_blocks
        aggregation_height = self.height // conv_layers_subsampling

        self.out_channels = self.base_channels * (2 ** (self.num_conv_blocks - 1))

        if self.aggregation == "conv":
            self.aggregation_layer = torch.nn.Conv2d(in_channels=self.out_channels,
                                                     out_channels=self.out_channels,
                                                     kernel_size=(aggregation_height, 1),
                                                     stride=1,
                                                     padding=0)
        else:
            raise NotImplementedError()

    def forward(self, x):
        x = self.encoder(x)
        x = self.aggregation_layer(x)

        return x


class VGGDecoder(torch.nn.Module):
    def __init__(self, height=40, patch_size=(40, 8), out_channels=3, dropout=0.0, base_channels=256, num_conv_blocks=3,
                 num_conv_layers=(3, 2, 2), upsampling="bilinear"):
        super().__init__()
        self.height = height
        self.upsampling = upsampling

        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_conv_blocks = num_conv_blocks
        self.num_conv_layers = num_conv_layers
        self.patch_size = patch_size
        self.dropout = dropout

        self.decoder = create_vgg_decoder(out_channels=out_channels,
                                          num_conv_blocks=num_conv_blocks,
                                          base_channels=base_channels,
                                          patch_size=patch_size,
                                          dropout=dropout,
                                          num_conv_layers=num_conv_layers,
                                          upsampling=upsampling)

        conv_layers_upsampling = 2 ** self.num_conv_blocks
        upsampling_height = self.height // conv_layers_upsampling

        self.upsampling_layer = torch.nn.Upsample(scale_factor=(upsampling_height, 1), mode=upsampling)

    def forward(self, x):
        x = self.upsampling_layer(x)
        x = self.decoder(x)
        return x


class AE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        result = {
            'tokens': None,
            'loss': None,
            'reconstructions': None
        }

        tokens = self.encoder(x)
        output = self.decoder(tokens)

        loss = torch.nn.functional.mse_loss(x, output)

        result['tokens'] = tokens
        result['loss'] = loss
        result['reconstructions'] = output

        return result


class VQVAE(torch.nn.Module):
    def __init__(self, encoder, decoder, num_tokens, codebook_dim, commitment_cost=0.25, decay=0.99, reconstruction_loss='mse'):
        super(VQVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.encoder_projection_layer = torch.nn.Conv2d(encoder.out_channels, codebook_dim, 1)
        self.decoder_projection_layer = torch.nn.Conv2d(codebook_dim, decoder.base_channels, 1)

        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.reconstruction_loss = reconstruction_loss

        self.vq = VectorQuantizer(self.num_tokens, self.codebook_dim, commitment_cost, decay)

    def calculate_loss(self, images: torch.Tensor, reconstructions: torch.Tensor, features: torch.Tensor, tokens: torch.Tensor):
        if self.reconstruction_loss.lower() in ('l2', 'mse'):
            recon_loss = torch.nn.functional.mse_loss(images, reconstructions)
        elif self.reconstruction_loss.lower() in ('l1', 'mae'):
            recon_loss = torch.nn.functional.l1_loss(images, reconstructions)
        else:
            raise ValueError(f'Unknown reconstruction loss: {self.reconstruction_loss}')

        vq_loss = self.vq.calculate_loss(tokens, features)
        loss = vq_loss + recon_loss

        return loss

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def quantize(self, x: torch.Tensor):
        x = self.encoder_projection_layer(x)
        tokens, labels = self.vq(x)
        projected_tokens = self.decoder_projection_layer(tokens)
        return projected_tokens, labels

    def forward(self, images):
        result = {
            'tokens': None,
            'loss': None,
            'reconstructions': None
        }

        features = self.encode(images)
        tokens, labels = self.quantize(features)
        reconstructions = self.decode(tokens)

        loss = self.calculate_loss(images, reconstructions, features, tokens)

        result['tokens'] = tokens
        result['labels'] = labels
        result['loss'] = loss
        result['reconstructions'] = reconstructions
        result['counts'] = torch.bincount(labels, minlength=self.num_tokens)

        return result


class VectorQuantizer(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)

        if decay > 0.0:
            self.embedding.weight.data.normal_()

            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            self.ema_w = torch.nn.Parameter(torch.Tensor(num_embeddings, self.embedding_dim))
            self.ema_w.data.normal_()

        else:
            self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

    def calculate_loss(self, tokens, features):
        if self.decay > 0.0:
            q_latent_loss = 0
        else:
            q_latent_loss = torch.nn.functional.mse_loss(tokens, features.detach())

        e_latent_loss = torch.nn.functional.mse_loss(tokens.detach(), features)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        return loss

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.decay > 0.0 and self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = torch.nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)

            self.embedding.weight = torch.nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))

        quantized = inputs + (quantized - inputs).detach()

        return quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices.detach().reshape(-1)


def main():
    N = 16
    C = 3
    H = 40
    W = 1024

    x = torch.rand((N, C, H, W))
    encoder = VGGEncoder()
    decoder = VGGDecoder()

    ae = AE(encoder, decoder)
    vqvae = VQVAE(encoder, decoder, num_tokens=1024, codebook_dim=512)

    y_ae = ae(x)
    y_vqvae = vqvae(x)

    print(ae)
    print()
    print(vqvae)
    print()

    print(f"Input: {x.shape}")
    print(f"AE reconstructions: {y_ae['reconstructions'].shape}")
    print(f"VQVAE reconstructions: {y_vqvae['reconstructions'].shape}")
    print(f"VQVAE tokens: {y_vqvae['tokens'].shape}")
    print(f"VQVAE labels: {y_vqvae['labels'].shape}")


if __name__ == '__main__':
    main()


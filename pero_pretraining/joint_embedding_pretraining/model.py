import torch
import einops

from pero_pretraining.models.transformers import VisionTransformerEncoder, VggTransformerEncoder


def init_backbone(backbone_definition):
    backbone_type = backbone_definition.get("type", "vit")

    if backbone_type == "vit":
        backbone = VisionTransformerEncoder()
    elif backbone_type == "vggt":
        backbone = VggTransformerEncoder()
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")

    return backbone


def init_head(head_definition):
    head_type = head_definition.get("type", "linear")

    if head_type == "linear":
        head = LinearHead()
    elif head_type == "mlp":
        head = MLPHead()
    else:
        raise ValueError(f"Unknown head type: {head_type}")

    return head


class JointEmbeddingTransformerEncoder(torch.nn.Module):
    def __init__(self, backbone, head, loss):
        super(JointEmbeddingTransformerEncoder, self).__init__()

        self.backbone = backbone
        self.head = head
        self.loss = loss

    def forward(self, images1, images2, image_masks1, image_masks2, shift_masks1, shift_masks2):
        output1 = self.encode(images1)
        output2 = self.encode(images2)

        loss = self.loss(output1, output2, image_masks1, image_masks2, shift_masks1, shift_masks2)

        result = {
            'output1': output1.detach(),
            'output2': output2.detach(),
            **loss
        }

        return result

    def encode(self, images):
        x = self.backbone(images, mask=None)
        x = einops.rearrange(x, 'n c w -> n w c')
        output = self.head(x)

        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class LinearHead(torch.nn.Module):
    def __init__(self, in_features=512, out_features=4096):
        super(LinearHead, self).__init__()

        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class MLPHead(torch.nn.Module):
    def __init__(self, in_dim=512, hidden_dim=8192, num_layers=3, use_bn=False):
        super(MLPHead, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_bn = use_bn

        self.layers = self._create_layers()

    def _create_layers(self):
        layers = []

        input_dim = self.in_dim

        for i in range(self.num_layers-1):
            layers.append(torch.nn.Linear(input_dim, self.hidden_dim))
            input_dim = self.hidden_dim

            if self.use_bn:
                layers.append(torch.nn.BatchNorm1d(self.hidden_dim))

            layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Linear(input_dim, self.hidden_dim))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        N, S, D = x.shape
        x_reshaped = x.reshape(N*S, D)

        y = self.layers(x_reshaped)

        out = y.reshape(N, S, -1)
        return out

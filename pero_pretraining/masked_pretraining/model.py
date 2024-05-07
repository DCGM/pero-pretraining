import torch

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
    else:
        raise ValueError(f"Unknown head type: {head_type}")

    return head


class MaskedTransformerEncoder(torch.nn.Module):
    def __init__(self, net, loss=None):
        super(MaskedTransformerEncoder, self).__init__()

        self.net = net
        self.loss = MaskedCrossEntropyLoss() if loss is None else loss

    def forward(self, x, labels=None, mask=None):
        output = self.net(x, mask)

        loss = None
        if mask is not None and labels is not None:
            loss = self.loss(output, labels, mask)

        result = {
            'output': output,
            'loss': loss
        }

        return result

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class MaskedCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, output, labels, mask):
        masked_output = output[mask == 1]
        masked_labels = labels[mask == 1]

        loss = torch.nn.functional.cross_entropy(masked_output, masked_labels)
        
        return loss
    

class LinearHead(torch.nn.Module):
    def __init__(self, in_features=512, out_features=4096):
        super(LinearHead, self).__init__()

        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

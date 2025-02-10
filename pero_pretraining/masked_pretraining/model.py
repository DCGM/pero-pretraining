import torch
import einops

from pero_pretraining.models.transformers import VisionTransformerEncoder, VggTransformerEncoder


def init_backbone(backbone_definition):
    backbone_type = backbone_definition.get("type", "vit")
    
    if backbone_type == "vit":
        backbone = VisionTransformerEncoder(**backbone_definition)
    elif backbone_type == "vggt":
        backbone = VggTransformerEncoder(**backbone_definition)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
    
    return backbone


def init_head(head_definition):
    head_type = head_definition.get("type", "linear")
    if "type" in head_definition:
        del head_definition["type"]

    if head_type == "linear":
        head = LinearHead(**head_definition)
    else:
        raise ValueError(f"Unknown head type: {head_type}")

    return head


class MaskedTransformerEncoder(torch.nn.Module):
    def __init__(self, backbone, head, loss=None):
        super(MaskedTransformerEncoder, self).__init__()

        self.backbone = backbone
        self.head = head
        self.loss = MaskedCrossEntropyLoss() if loss is None else loss

    def forward(self, x, labels=None, mask=None):
        output = self.encode(x, mask)

        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).to(output.device)

        loss = None
        if mask is not None and labels is not None:
            loss = self.loss(output, labels, mask)

        result = {
            'output': output,
            'loss': loss
        }

        return result

    def encode(self, images, mask=None):
        x = self.backbone(images, mask=mask)
        x = einops.rearrange(x, 'n c w -> n w c')
        output = self.head(x)

        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class MaskedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, unmasked_weight=None):
        super(MaskedCrossEntropyLoss, self).__init__()

        self.unmasked_weight = unmasked_weight

    def forward(self, output, labels, mask):
        masked_output = output[mask == 1]
        masked_labels = labels[mask == 1]

        loss = torch.nn.functional.cross_entropy(masked_output, masked_labels)

        if self.unmasked_weight is not None:
            unmasked_output = output[mask == 0]
            unmasked_labels = labels[mask == 0]

            image_mask = unmasked_labels >= 0
            unmasked_output = unmasked_output[image_mask]
            unmasked_labels = unmasked_labels[image_mask]

            unmasked_loss = torch.nn.functional.cross_entropy(unmasked_output, unmasked_labels)
            loss = loss + self.unmasked_weight * unmasked_loss
        
        return loss
    

class LinearHead(torch.nn.Module):
    def __init__(self, in_features=512, out_features=4096):
        super(LinearHead, self).__init__()

        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


def main():
    backbone_definition = {
        "type": "vit"
    }

    head_definition = {
        "type": "linear"
    }

    backbone = init_backbone(backbone_definition)
    head = init_head(head_definition)

    import einops

    class Net(torch.nn.Module):
        def __init__(self, backbone, head):
            super(Net, self).__init__()

            self.backbone = backbone
            self.head = head

        def forward(self, x, mask=None):
            x = self.backbone(x, mask)

            x = einops.rearrange(x, 'n c w -> n w c')
            x = self.head(x)
            return x

    net = Net(backbone, head)
    model = MaskedTransformerEncoder(net)
    print(model)

    input_data = torch.rand(1, 3, 40, 1024)
    output = model(input_data)

    print(input_data.shape)
    print(output["output"].shape)

    return 0


if __name__ == "__main__":
    main()

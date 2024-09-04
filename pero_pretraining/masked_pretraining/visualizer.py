import torch

from pero_pretraining.common.visualizer import Visualizer
from pero_pretraining.masked_pretraining.batch_operator import BatchOperator


class MaskedVisualizer(BatchOperator):
    def __init__(self, model, dataloader, device,  masking_prob=0.2):
        super(MaskedVisualizer, self).__init__(device, masking_prob)

        self.model = model
        self.dataloader = dataloader

        self._num_labels = self.model.net.head.linear.out_features
        self._visualizer = Visualizer()

    def visualize(self):
        batch = next(iter(self.dataloader))
        # predictions = self._inference_step(batch)

        with torch.no_grad():
            images, labels, mask = self.prepare_batch(batch)
            output = self.model.forward(images, labels, mask)

        predictions = torch.argmax(output['output'], dim=-1).cpu().numpy()

        output = self._visualizer.visualize(images=batch['images'],
                                            image_masks=batch['image_masks'],
                                            labels=batch['labels'],
                                            predicted_labels=predictions,
                                            mask=mask,
                                            num_labels=self._num_labels)

        return output

    def _inference_step(self, batch):
        with torch.no_grad():
            images, labels, mask = self.prepare_batch(batch)
            output = self.model.forward(images, labels, mask)

        return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb-path", type=str, help="Path to LMDB", required=True)
    parser.add_argument("--lines-path", type=str, help="Path to lines file", required=True)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")

    parser.add_argument("--backbone", type=str, help="Backbone definition", required=True)
    parser.add_argument("--head", type=str, help="Head definition", required=True)
    parser.add_argument("--model-path", type=str, help="Path to model", required=True)

    parser.add_argument("--output", type=str, help="Output image path", required=True)

    args = parser.parse_args()
    return args


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


def init_model(device, backbone_definition, head_definition, path=None):
    backbone = init_backbone(backbone_definition)
    head = init_head(head_definition)
    net = Net(backbone, head)

    model = MaskedTransformerEncoder(net)
    model.to(device)

    if path is not None:
        model.load(path)

    return model


def main():
    args = parse_args()

    np.random.seed(37)
    torch.random.manual_seed(37)

    dataset = Dataset(args.lmdb_path, args.lines_path, augmentations=None, pair_images=False)
    batch_creator = BatchCreator()
    dataloader = create_dataloader(dataset, batch_creator, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, persistent_workers=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    backbone_definition = json.loads(args.backbone)
    head_definition = json.loads(args.head)

    model = init_model(device, backbone_definition, head_definition, args.model_path)

    visualizer = MaskedVisualizer(model, dataloader, device)
    image = visualizer.visualize()

    cv2.imwrite(args.output, image)

    return 0


if __name__ == "__main__":
    import cv2
    import json
    import torch
    import einops
    import argparse
    import numpy as np
    from pero_pretraining.common.dataset import Dataset
    from pero_pretraining.common.dataloader import create_dataloader, BatchCreator
    from pero_pretraining.masked_pretraining.model import init_backbone, init_head, MaskedTransformerEncoder
    exit(main())

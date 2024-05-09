import torch


class BatchOperator:
    def __init__(self, device):
        self.device = device

    def prepare_batch(self, batch):
        images = self._prepare_batch_images(batch, key="images")

        return images

    def _prepare_batch_images(self, batch, key="images"):
        images = batch[key] / 255.
        images = torch.from_numpy(images).float().to(self.device)

        return images

    @staticmethod
    def batch_size(batch):
        return batch['images'].shape[0]

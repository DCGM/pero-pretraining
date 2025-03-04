import torch
import numpy as np


class BatchOperator:
    def __init__(self, device):
        self.device = device

    def prepare_batch(self, batch):
        images1 = self._prepare_batch_images(batch, key="images")
        images2 = self._prepare_batch_images(batch, key="images2")

        image_masks1 = self._prepare_batch_masks(batch, key="image_masks")
        image_masks2 = self._prepare_batch_masks(batch, key="image_masks2")

        shift_masks1 = self._prepare_batch_masks(batch, key="shift_masks")
        shift_masks2 = self._prepare_batch_masks(batch, key="shift_masks2")

        return images1, images2, image_masks1, image_masks2, shift_masks1, shift_masks2

    def _prepare_batch_images(self, batch, key="images"):
        images = torch.from_numpy(batch[key]).to(self.device).float().permute(0, 3, 1, 2) / 255.0

        return images

    def _prepare_batch_masks(self, batch, key="image_masks"):
        masks = torch.from_numpy(batch[key]).to(self.device)

        return masks

    @staticmethod
    def batch_size(batch):
        return batch['images'].shape[0]

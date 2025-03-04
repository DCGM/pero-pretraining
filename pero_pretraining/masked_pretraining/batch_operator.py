import torch
import numpy as np


class BatchOperator:
    def __init__(self, device, masking_prob):
        self.device = device
        self.masking_prob = masking_prob

    def prepare_batch(self, batch):
        images = self._prepare_batch_images(batch)
        labels = self._prepare_batch_labels(batch)
        mask = self._create_mask(batch)

        return images, labels, mask

    def _prepare_batch_images(self, batch):
        images = torch.from_numpy(batch['images']).to(self.device).float().permute(0, 3, 1, 2) / 255.0

        return images
    
    def _prepare_batch_labels(self, batch):
        labels = torch.from_numpy(batch['labels']).to(self.device).long()

        return labels

    def _create_mask(self, batch):
        labels = batch['labels']
        active_labels = (labels >= 0).astype(int)
        mask = (np.random.rand(*labels.shape) < self.masking_prob).astype(int) * active_labels

        return mask

    @staticmethod
    def batch_size(batch):
        return batch['images'].shape[0]

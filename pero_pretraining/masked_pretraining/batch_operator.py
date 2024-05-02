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
        images = batch['images'] / 255.
        images = torch.from_numpy(images).float().to(self.device)

        return images
    
    def _prepare_batch_labels(self, batch):
        labels = batch['labels']
        labels = torch.from_numpy(labels).long().to(self.device)

        return labels

    def _create_mask(self, batch):
        labels = batch['labels']
        active_labels = (labels >= 0).astype(int)
        mask = (np.random.rand(*labels.shape) < self.masking_prob).astype(int) * active_labels

        return mask

    @staticmethod
    def batch_size(batch):
        return batch['images'].shape[0]

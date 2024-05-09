import torch

from pero_pretraining.common.visualizer import Visualizer
from pero_pretraining.masked_pretraining.batch_operator import BatchOperator


class MaskedVisualizer(BatchOperator):
    def __init__(self, model, dataloader, masking_prob=0.2):
        super(MaskedVisualizer, self).__init__(model.device, masking_prob)

        self.model = model
        self.dataloader = dataloader

        self._num_labels = self.model.net[-1].linear.out_features
        self._visualizer = Visualizer()

    def visualize(self):
        batch = next(iter(self.dataloader))
        predictions = self._inference_step(batch)

        output = self._visualizer.visualize(images=batch['images'],
                                            image_masks=batch['image_masks'],
                                            labels=batch['labels'],
                                            predicted_labels=predictions['output'],
                                            num_labels=self._num_labels)

        return output

    def _inference_step(self, batch):
        with torch.no_grad():
            images, labels, mask = self.prepare_batch(batch)
            output = self.model.forward(images, labels, mask)

        return output

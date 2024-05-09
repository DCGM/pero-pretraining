import torch
import numpy as np

from pero_pretraining.common.visualizer import Visualizer
from pero_pretraining.autoencoders.batch_operator import BatchOperator


class AutoEncodersVisualizer(BatchOperator):
    def __init__(self, model, dataloader):
        super(AutoEncodersVisualizer, self).__init__(model.device)

        self.model = model
        self.dataloader = dataloader

        self._visualizer = Visualizer()

    def visualize(self):
        batch = next(iter(self.dataloader))
        predictions = self._inference_step(batch)

        image = self._visualizer.visualize(images=batch['images'],
                                           images2=predictions['reconstructions'])

        return image

    def _inference_step(self, batch):
        with torch.no_grad():
            images = self.prepare_batch(batch)
            output = self.model.forward(images)

        return output

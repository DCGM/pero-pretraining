import torch

from pero_pretraining.common.visualizer import Visualizer


class AutoEncodersVisualizer:
    def __init__(self, batch_operator, model, dataloader):
        self.batch_operator = batch_operator

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
            images = self.batch_operator.prepare_batch(batch)
            output = self.model.forward(images)

        return output

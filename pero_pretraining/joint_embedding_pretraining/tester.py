import torch
import numpy as np
from collections import defaultdict

from pero_pretraining.joint_embedding_pretraining.batch_operator import BatchOperator


class Tester(BatchOperator):
    def __init__(self, model, dataloader, max_lines=None):
        super(Tester, self).__init__(model.device)

        self.model = model
        self.dataloader = dataloader
        self.max_lines = max_lines

    def test(self):
        total_loss = 0
        num_lines = 0
        num_batches = 0

        self.model.eval()

        with torch.no_grad():
            dataloader_iterator = iter(self.dataloader)

            while True:
                try:
                    batch = next(dataloader_iterator)
                except StopIteration:
                    break

                result = self.test_step(batch)
                total_loss += result['loss']

                num_lines += self.batch_size(batch)
                num_batches += 1

                if self.max_lines is not None and num_lines > self.max_lines:
                    break

        self.model.train()

        average_loss = total_loss / num_batches

        output = {
            'loss': average_loss,
        }

        return output

    def test_step(self, batch):
        images1, images2, image_masks1, image_masks2, shift_masks1, shift_masks2  = self.prepare_batch(batch)
        output = self.model.forward(images1, images2, image_masks1, image_masks2, shift_masks1, shift_masks2)

        return output

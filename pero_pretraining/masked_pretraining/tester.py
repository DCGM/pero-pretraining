import torch
import numpy as np
from collections import defaultdict


class Tester:
    def __init__(self, batch_operator, model, dataloader, max_lines=None, measured_errors=(1, 3, 10)):
        self.batch_operator = batch_operator

        self.model = model
        self.dataloader = dataloader
        self.max_lines = max_lines
        self.measured_errors = measured_errors

    def test(self):
        total_loss = 0
        num_lines = 0
        num_batches = 0
        errors = defaultdict(int)

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

                self._update_errors(errors, result, batch)

                num_lines += self.batch_operator.batch_size(batch)
                num_batches += 1

                if self.max_lines is not None and num_lines > self.max_lines:
                    break

        self.model.train()

        average_loss = total_loss / num_batches
        errors = {k: v / errors['length'] for k, v in errors.items() if 'errors' in k}

        output = {
            'loss': average_loss,
            **errors
        }

        return output

    def test_step(self, batch):
        images, labels, mask = self.batch_operator.prepare_batch(batch)
        output = self.model.forward(images, labels, mask)

        batch['mask'] = mask

        return output

    def _update_errors(self, errors, result, batch):
        output = result['output'].cpu().numpy()
        mask = batch['mask']
        labels = batch['labels']

        if type(mask) == torch.Tensor:
            mask = mask.cpu().numpy()

        if type(labels) == torch.Tensor:
            labels = labels.cpu().numpy()

        masked_output = output[mask == 1]  # .cpu().numpy()
        masked_labels = labels[mask == 1]  # .cpu().numpy()

        for i, measured_error in enumerate(self.measured_errors):
            masked_predictions = np.argmax(masked_output, axis=1) if measured_error == 1 else self._topk(masked_output, measured_error)

            prediction_errors, length = self._calculate_errors(masked_predictions, masked_labels)
            errors[f"errors_{measured_error}"] += prediction_errors

            if i == 0:
                errors['length'] += length

        return errors

    @staticmethod
    def _topk(output, k):
        top = np.argsort(output, axis=1)[:, -k:]

        return top

    @staticmethod
    def _calculate_errors(hypothesis, reference):
        errors = 0

        for h, r in zip(hypothesis, reference):
            if type(h) in {list, np.ndarray, torch.Tensor}:
                if r not in h:
                    errors += 1
            else:
                if h != r:
                    errors += 1

        return errors, len(reference)

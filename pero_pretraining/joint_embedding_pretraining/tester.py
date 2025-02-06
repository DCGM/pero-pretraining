import torch


class Tester:
    def __init__(self, batch_operator, model, dataloader, max_lines=None):
        self.batch_operator = batch_operator

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

                num_lines += self.batch_operator.batch_size(batch)
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
        images1, images2, image_masks1, image_masks2, shift_masks1, shift_masks2  = self.batch_operator.prepare_batch(batch)

        if self.bfloat16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.model.forward(images1, images2, image_masks1, image_masks2, shift_masks1, shift_masks2)
            output['output1'] = output['output1'].float()
            output['output2'] = output['output2'].float()

        else:
            output = self.model.forward(images1, images2, image_masks1, image_masks2, shift_masks1, shift_masks2)


        return output

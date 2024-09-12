import torch
from pero_pretraining.masked_pretraining.batch_operator import BatchOperator
import time

class Trainer(BatchOperator):
    def __init__(self, model, dataloader, optimizer, scheduler, device, masking_prob=0.2):
        super(Trainer, self).__init__(device, masking_prob)

        self.model = model
        self.dataloader = dataloader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.on_view_step = None

    def train(self, end_iteration, start_iteration=0, view_step=1000):
        dataloader_iterator = iter(self.dataloader)

        start_time = time.time()
        iteration_count = 0
        last_batch_size = None

        for iteration in range(start_iteration, end_iteration + 1):
            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(self.dataloader)
                batch = next(dataloader_iterator)

            if last_batch_size and self.device.type == "cuda" and last_batch_size != batch['images'].shape:
                torch.cuda.empty_cache()
                print("Cleared cache.")
            last_batch_size = batch['images'].shape

            self.scheduler.update_learning_rate(iteration)
            self.train_step(batch)

            iteration_count += 1

            if self.on_view_step is not None and iteration > 0 and iteration % view_step == 0:
                print(f"Iteration {iteration}. Time: {time.time() - start_time:.2f}s. Speed: {iteration_count / (time.time() - start_time):.2f} it/s.")
                self.on_view_step(iteration, self.model)
                iteration_count = 0
                start_time = time.time()


    def train_step(self, batch):

        images, labels, mask = self.prepare_batch(batch)
        self.optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model.forward(images, labels, mask)
            loss = output['loss']
        loss.backward()

        self.optimizer.step()

        return loss

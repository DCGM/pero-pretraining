import time
import torch


class Trainer:
    def __init__(self, batch_operator, model, dataloader, optimizer, scheduler, bfloat16=False):
        self.batch_operator = batch_operator

        self.model = model
        self.dataloader = dataloader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.bfloat16 = bfloat16

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

            # if last_batch_size and self.device.type == "cuda" and last_batch_size != batch['images'].shape:
            #     torch.cuda.empty_cache()
            #     print("Cleared cache.")
            # last_batch_size = batch['images'].shape

            self.scheduler.update_learning_rate(iteration)
            self.train_step(batch)

            if self.batch_operator.device.type == "cuda":
                torch.cuda.empty_cache()

            iteration_count += 1

            if self.on_view_step is not None and iteration > 0 and iteration % view_step == 0:
                elapsed_time = time.time() - start_time
                self.on_view_step(iteration, self.model, elapsed_time, iteration_count)
                iteration_count = 0
                start_time = time.time()


    def train_step(self, batch):
        images, labels, mask = self.batch_operator.prepare_batch(batch)
        self.optimizer.zero_grad()

        if self.bfloat16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.model.forward(images, labels, mask)
        else:
            output = self.model.forward(images, labels, mask)

        loss = output['loss']
        loss.backward()

        self.optimizer.step()

        return loss

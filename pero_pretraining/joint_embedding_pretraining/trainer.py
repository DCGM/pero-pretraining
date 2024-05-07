import torch
from pero_pretraining.joint_embedding_pretraining.batch_operator import BatchOperator


class Trainer(BatchOperator):
    def __init__(self, model, dataloader, optimizer, scheduler):
        super(Trainer, self).__init__(model.device)

        self.model = model
        self.dataloader = dataloader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.on_view_step = None

    def train(self, end_iteration, start_iteration=0, view_step=1000):
        dataloader_iterator = iter(self.dataloader)

        for iteration in range(start_iteration, end_iteration + 1):
            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(self.dataloader)
                batch = next(dataloader_iterator)

            self.scheduler.update_learning_rate(iteration)

            self.train_step(batch)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            if self.on_view_step is not None and iteration > 0 and iteration % view_step == 0:
                self.on_view_step(iteration)

    def train_step(self, batch):
        self.optimizer.zero_grad()

        images1, images2, image_masks1, image_masks2, shift_masks1, shift_masks2  = self.prepare_batch(batch)
        output = self.model.forward(images1, images2, image_masks1, image_masks2, shift_masks1, shift_masks2)

        loss = output['loss']
        loss.backward()

        self.optimizer.step()

        return loss

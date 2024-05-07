import torch
import numpy as np

from pero_pretraining.common.visualizer import Visualizer
from pero_pretraining.masked_pretraining.batch_operator import BatchOperator


class JointEmbeddingVisualizer(BatchOperator):
    def __init__(self, model, dataloader, num_labels=4096):
        super(JointEmbeddingVisualizer, self).__init__(model.device)

        self._visualizer = Visualizer()

        self.model = model
        self.dataloader = dataloader
        self.num_labels = num_labels

    def visualize(self):
        batch = next(iter(self.dataloader))
        predictions = self._inference_step(batch)

        image = self._visualizer.visualize(images=batch['images'],
                                           images2=batch['images2'],
                                           image_masks=batch['image_masks'],
                                           image_masks2=batch['image_masks2'],
                                           shift_masks=batch['shift_masks'],
                                           shift_masks2=batch['shift_masks2'])

        bottom_padding = image.shape[0] // batch['images'].shape[0] - batch['images'].shape[2]
        similarity_image = self._visualize_similarity(batch['images'], batch['images2'], predictions['output1'], predictions['output2'], bottom_padding=bottom_padding)

        image = np.concatenate([image, similarity_image], axis=1)

        return image

    def _inference_step(self, batch):
        with torch.no_grad():
            images1, images2, image_masks1, image_masks2, shift_masks1, shift_masks2 = self.prepare_batch(batch)
            output = self.model.forward(images1, images2, image_masks1, image_masks2, shift_masks1, shift_masks2)

        return output

    def _visualize_similarity(self, x, y, x_output, y_output, k=10, bottom_padding=0):
        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)

        x_exp = x_output / torch.norm(x_output, p=2, dim=1, keepdim=True)
        y_exp = y_output / torch.norm(y_output, p=2, dim=1, keepdim=True)

        # for each sample select random frame id
        query_ids = torch.randint(0, x_exp.shape[2], (x.shape[0],))
        query = x_exp[torch.arange(x.shape[0]), :, query_ids]

        # concatenate all the sequences from y_exp
        keys = y_exp.permute(0, 2, 1).reshape(-1, y_exp.shape[1])

        # compute similarity between queries and keys
        sim = query @ keys.T

        # select top 'k' values
        _, topk = torch.topk(sim, k, dim=1, largest=False)

        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        y = np.concatenate([line for line in y], axis=1)

        # create a collage of top k retrieved patches
        collage = self._create_collage(x, y, query_ids, k, topk, bottom_padding)
        return collage

    def _create_collage(self, x, y, query_ids, k, topk, bottom_padding=0, crop_width=64, separator_width=5):
        separator = np.zeros((x.shape[1], separator_width,  3), dtype=np.uint8)
        collage = np.zeros(((x.shape[1]+2) * x.shape[0], (k+1) * crop_width + k * separator_width, 3), dtype=np.uint8)

        for i in range(x.shape[0]):
            row_images = [self._get_line_crop(x[i], query_ids[i] * self._visualizer.subsampling_factor, crop_width)]

            for j in range(k):
                row_images.append(separator)
                row_images.append(self._get_line_crop(y, topk[i, j] * self._visualizer.subsampling_factor, crop_width))

            row_images = np.concatenate(row_images, axis=1)
            row_images = np.pad(row_images, ((0, bottom_padding), (0, 0), (0, 0)), mode='constant', constant_values=0)

            collage[i * (row_images.shape[0] + 2):i * (row_images.shape[0] + 2) + row_images.shape[0], :row_images.shape[1], :] = row_images

        return collage

    @staticmethod
    def _get_line_crop(x, pos, width=32):
        start = max(pos - width // 2, 0)
        end = min(pos + width // 2, x.shape[1]-1)
        return x[:, start:end, :]
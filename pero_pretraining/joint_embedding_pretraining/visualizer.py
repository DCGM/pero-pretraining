import torch
import numpy as np

from pero_pretraining.common.visualizer import Visualizer
from pero_pretraining.joint_embedding_pretraining.batch_operator import BatchOperator


class JointEmbeddingVisualizer:
    def __init__(self, batch_operator, model, dataloader, bfloat16=False):
        self.batch_operator = batch_operator

        self.model = model
        self.dataloader = dataloader

        self._visualizer = Visualizer()
        self.bfloat16 = bfloat16

    def visualize(self):
        batch = next(iter(self.dataloader))
        predictions = self._inference_step(batch)

        image = self._visualizer.visualize(images=batch['images'],
                                           images2=batch['images2'],
                                           image_masks=batch['image_masks'],
                                           image_masks2=batch['image_masks2'],
                                           shift_masks=batch['shift_masks'],
                                           shift_masks2=batch['shift_masks2'])

        bottom_padding = image.shape[0] // batch['images'].shape[0] - batch['images'].shape[1]
        similarity_image = self._visualize_similarity(batch['images'],
                                                      batch['images2'],
                                                      batch['image_masks'],
                                                      predictions['output1'],
                                                      predictions['output2'],
                                                      bottom_padding=bottom_padding)

        image = np.concatenate([image, similarity_image], axis=1)

        return image

    def _inference_step(self, batch):
        with torch.no_grad():
            images1, images2, image_masks1, image_masks2, shift_masks1, shift_masks2 = self.batch_operator.prepare_batch(batch)

            if self.bfloat16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = self.model.forward(images1, images2, image_masks1, image_masks2, shift_masks1, shift_masks2)
                output['output1'] = output['output1'].float()
                output['output2'] = output['output2'].float()

            else:
                output = self.model.forward(images1, images2, image_masks1, image_masks2, shift_masks1, shift_masks2)

            batch['images'] = images1.permute(0, 2, 3, 1).cpu().numpy()
            batch['images2'] = images2.permute(0, 2, 3, 1).cpu().numpy()
            batch['image_masks'] = image_masks1.cpu().numpy()
            batch['image_masks2'] = image_masks2.cpu().numpy()
            batch['shift_masks'] = shift_masks1.cpu().numpy()
            batch['shift_masks2'] = shift_masks2.cpu().numpy()

        return output

    def _visualize_similarity(self, x, y, x_mask, x_output, y_output, k=10, bottom_padding=0):
        if x.dtype == np.float32:
            x = (x * 255).astype(np.uint8)

        if y.dtype == np.float32:
            y = (y * 255).astype(np.uint8)

        x_exp = x_output / torch.norm(x_output, p=2, dim=1, keepdim=True)
        y_exp = y_output / torch.norm(y_output, p=2, dim=1, keepdim=True)

        starts = []
        ends = []
        for i in range(x_exp.shape[0]):
            x_mask_image = np.where(x_mask[i] == 1)[0]
            starts.append(x_mask_image[0])
            ends.append(x_mask_image[-1])

        # for each sample select random frame id
        query_ids = np.random.randint(starts, ends)
        query = x_exp[torch.arange(x.shape[0]), query_ids]

        # concatenate all the sequences from y_exp
        keys = y_exp.reshape(-1, y_exp.shape[2])

        # compute similarity between queries and keys
        sim = query @ keys.T

        # select top 'k' values
        _, topk = torch.topk(sim, k, dim=1, largest=False)

        y = np.concatenate([line for line in y], axis=1)

        # create a collage of top k retrieved patches
        collage = self._create_collage(x, y, query_ids, k, topk, bottom_padding)
        return collage

    def _create_collage(self, x, y, query_ids, k, topk, bottom_padding=0, crop_width=64, separator_width=5):
        separator = np.zeros((x.shape[1], separator_width,  3), dtype=np.uint8)
        collage = np.zeros(((x.shape[1] + bottom_padding) * x.shape[0], (k+1) * crop_width + k * separator_width, 3), dtype=np.uint8)

        for i in range(x.shape[0]):
            row_images = [self._get_line_crop(x[i], query_ids[i] * self._visualizer.subsampling_factor, crop_width)]

            for j in range(k):
                row_images.append(separator)
                row_images.append(self._get_line_crop(y, topk[i, j] * self._visualizer.subsampling_factor, crop_width))

            row_images = np.concatenate(row_images, axis=1)
            row_images = np.pad(row_images, ((0, bottom_padding), (0, 0), (0, 0)), mode='constant', constant_values=0)

            collage[i * row_images.shape[0]:(i + 1) * row_images.shape[0], :row_images.shape[1], :] = row_images

        return collage

    @staticmethod
    def _get_line_crop(x, pos, width=32):
        start = max(pos - width // 2, 0)
        end = min(pos + width // 2, x.shape[1]-1)
        return x[:, start:end, :]

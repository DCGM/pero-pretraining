import cv2
import numpy as np


class Visualizer:
    def __init__(self, subsampling_factor=8):
        self.subsampling_factor = subsampling_factor

    def visualize(self, images, images2=None, image_masks=None, image_masks2=None, shift_masks=None, shift_masks2=None,
                  labels=None, predicted_labels=None, num_labels=None, original_images=None, original_images2=None):
        image = self.visualize_column(images=images,
                                      image_masks=image_masks,
                                      shift_masks=shift_masks,
                                      labels=labels,
                                      predictions=predicted_labels,
                                      num_labels=num_labels,)

        separator = np.ones((image.shape[0], 10, 3), dtype=np.uint8) * 255

        if images2 is not None:
            image2 = self.visualize_column(images=images2,
                                           image_masks=image_masks2,
                                           shift_masks=shift_masks2,
                                           labels=labels,
                                           predictions=predicted_labels,
                                           num_labels=num_labels)

            image = np.concatenate((image, separator, image2), axis=1)

        if original_images is not None:
            original_line_image_height = original_images.shape[1]
            visualized_line_image_height = image.shape[0] / len(images)
            line_padding = int(visualized_line_image_height - original_line_image_height)

            original_images = self.visualize_column(images=original_images, line_padding=line_padding)

            image = np.concatenate((image, separator, original_images), axis=1)

        if original_images2 is not None:
            original_line_image_height = original_images2.shape[1]
            visualized_line_image_height = image.shape[0] / len(images2)
            line_padding = int(visualized_line_image_height - original_line_image_height)

            original_images2 = self.visualize_column(images=original_images2, line_padding=line_padding)

            image = np.concatenate((image, separator, original_images2), axis=1)

        return image

    def visualize_column(self, images, predictions=None, labels=None, num_labels=None, image_masks=None, shift_masks=None,
                         line_padding=0):
        lines = []
        for i, line_image in enumerate(images):
            line = [line_image]

            if line_padding > 0:
                line.append(np.zeros((line_padding, line_image.shape[1], 3), dtype=np.uint8))

            if image_masks is not None:
                colors = {
                    0: [64, 64, 255],
                    1: [64, 255, 64]
                }

                line.append(self.visualize_annotation(line_image, image_masks[i], colors_dict=colors))

            if shift_masks is not None:
                colors = {
                    0: [64, 64, 255],  # not-shared, red
                    1: [64, 255, 64],  # shared, green
                    2: [0, 192, 255],  # shared padding, orange
                }

                line.append(self.visualize_annotation(line_image, shift_masks[i], colors_dict=colors))

            if labels is not None:
                line.append(self.visualize_annotation(line_image, labels[i], num_labels))

                if predictions is not None:
                    line.append(self.visualize_annotation(line_image, predictions[i], num_labels))

                    correct_labels = np.equal(labels[i], predictions[i]).astype(np.uint8)
                    line.append(self.visualize_annotation(line_image, correct_labels, 2))

            lines.append(np.concatenate(line, axis=0))

        image = np.concatenate(lines, axis=0)

        return image

    def visualize_annotation(self, image, annotation, n=2, colors_dict=None):
        annotation_image = np.zeros((self.subsampling_factor, image.shape[1], 3), dtype=np.uint8)

        for i, label in enumerate(annotation):
            color = colors_dict[label] if colors_dict is not None and label in colors_dict else self.label_to_color(label, n)
            annotation_image[:, i * self.subsampling_factor:(i + 1) * self.subsampling_factor] = color

        return annotation_image

    def label_to_color(self, label, num_labels):
        label_color_number = int((256**3 - 1) * label / (num_labels - 1))
        label_color_number = max(0, min(label_color_number, 256**3 - 1))

        binary = bin(label_color_number)[2:]

        if len(binary) < 24:
            binary = '0' * (24 - len(binary)) + binary

        r = int(binary[:8], 2)
        g = int(binary[8:16], 2)
        b = int(binary[16:], 2)

        return [b, g, r]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb-path", type=str, help="Path to LMDB", required=True)
    parser.add_argument("--lines-path", type=str, help="Path to lines file", required=True)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    np.random.seed(37)
    torch.random.manual_seed(37)

    dataset = Dataset(args.lmdb_path, args.lines_path, augmentations=None, pair_images=False)
    batch_creator = BatchCreator()
    # batch_creator = BatchCreator(crop_width=512, crop_step=8)
    dataloader = create_dataloader(dataset, batch_creator, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, persistent_workers=False)
    batch = next(iter(dataloader))

    visualizer = Visualizer()
    image = visualizer.visualize(images=batch['images'],
                                 images2=batch['images2'],
                                 image_masks=batch['image_masks'],
                                 image_masks2=batch['image_masks2'],
                                 shift_masks=batch['shift_masks'],
                                 shift_masks2=batch['shift_masks2'],
                                 labels=batch['labels'],
                                 num_labels=4096,
                                 original_images=batch['original_images'],
                                 original_images2=batch['original_images2'])

    cv2.imshow("Image", image)
    cv2.waitKey(0)

    return 0


if __name__ == "__main__":
    import torch
    import argparse
    from pero_pretraining.common.dataset import Dataset
    from pero_pretraining.common.dataloader import create_dataloader, BatchCreator
    exit(main())

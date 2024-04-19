import numpy as np
from typing import List, Dict
from torch.utils.data import DataLoader


def create_dataloader(dataset, batch_creator=None, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True):
    if batch_creator is None:
        batch_creator = BatchCreator()

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             persistent_workers=persistent_workers if num_workers > 0 else False,
                             collate_fn=batch_creator.create_batch)
    return data_loader


class BatchCreator:
    def __init__(self, subsampling_factor=8, padding_coefficient=32, crop_width=None, crop_step=1, same_left_paddings=False):
        self.subsampling_factor = subsampling_factor
        self.padding_coefficient = padding_coefficient
        self.crop_width = crop_width
        self.crop_step = crop_step
        self.same_left_paddings = same_left_paddings

    def create_batch(self, data: List[Dict]) -> Dict:
        stack_images_result = self.stack_images(data)

        (images, images2, image_masks, image_masks2, left_paddings, left_paddings2, original_images, original_images2,
         shifts, shift_masks1, shift_masks2) = stack_images_result

        target_labels_length = images.shape[2] // self.subsampling_factor

        labels, ids = self.stack_annotations(data, target_labels_length, left_paddings)

        batch = {
            'images': images,
            'images2': images2,
            'image_masks': image_masks,
            'image_masks2': image_masks2,
            'shifts': shifts,
            'shift_masks': shift_masks1,
            'shift_masks2': shift_masks2,
            'labels': labels,
            'ids': ids,
            'original_images': original_images,
            'original_images2': original_images2,
        }

        return batch

    def stack_annotations(self, data: List[Dict], target_labels_length, left_paddings):
        ids = [d['image_id'] for d in data]
        labels = None

        if any([d['labels'] is not None for d in data]):
            labels = np.full((len(data), target_labels_length), fill_value=-1)
            for i, (d, lp) in enumerate(zip(data, left_paddings)):
                if d['labels'] is not None:
                    labels[i, lp:lp + len(d['labels'])] = d['labels']

        return labels, ids

    def stack_images(self, data: List[Dict]):
        if self.crop_width is not None:
            crop_shifts = self.crop_images(data)
            target_width = self.crop_width
        else:
            crop_shifts = [0] * len(data)
            all_widths = [d['image'].shape[1] for d in data] + [d['image2'].shape[1] for d in data if 'image2' in d and d['image2'] is not None]
            target_width = self.calculate_padded_image_width(max(all_widths))

        image_height = data[0]['image'].shape[0]
        image_channels = data[0]['image'].shape[2]

        batch_images1 = np.zeros([len(data), image_height, target_width, image_channels], dtype=np.uint8)
        batch_image_masks1 = np.ones([len(data), target_width // self.subsampling_factor], dtype=np.uint8)

        left_paddings1 = []

        for batch_image, batch_image_mask, line_image in zip(batch_images1, batch_image_masks1, [d['image'] for d in data]):
            if line_image.shape[1] == target_width:
                left_padding = 0
            else:
                left_padding = np.random.randint(0, target_width - line_image.shape[1]) // self.subsampling_factor

            left_padding_pixels = left_padding * self.subsampling_factor

            batch_image[:, left_padding_pixels:left_padding_pixels + line_image.shape[1]] = line_image
            batch_image_mask[:left_padding] = 0
            batch_image_mask[left_padding + int(np.ceil(line_image.shape[1] / self.subsampling_factor)):] = 0

            left_paddings1.append(left_padding)

        batch_images2 = None
        batch_image_masks2 = None
        shifts = None
        left_paddings2 = None
        shift_masks1 = None
        shift_masks2 = None

        if any([d['image2'] is not None for d in data]):
            batch_images2 = np.zeros_like(batch_images1)
            batch_image_masks2 = np.ones_like(batch_image_masks1)
            left_paddings2 = []

            for batch_image, batch_image_mask, line_image, left_padding in zip(batch_images2, batch_image_masks2, [d['image2'] for d in data], left_paddings1):
                if not self.same_left_paddings:
                    if line_image.shape[1] == target_width:
                        left_padding = 0
                    else:
                        left_padding = np.random.randint(0, target_width - line_image.shape[1]) // self.subsampling_factor

                left_padding_pixels = left_padding * self.subsampling_factor

                batch_image[:, left_padding_pixels:left_padding_pixels + line_image.shape[1]] = line_image
                batch_image_mask[:left_padding] = 0
                batch_image_mask[left_padding + int(np.ceil(line_image.shape[1] / self.subsampling_factor)):] = 0

                left_paddings2.append(left_padding)

            shifts = [cs + (lp1 - lp2) for cs, lp1, lp2 in zip(crop_shifts, left_paddings1, left_paddings2)]

            shift_masks1 = np.zeros([len(data), target_width // self.subsampling_factor], dtype=np.uint8)
            for shift_mask1, shift in zip(shift_masks1, shifts):
                if shift < 0:
                    shift_mask1[:shift] = 1
                else:
                    shift_mask1[shift:] = 1

            shift_masks2 = np.copy(shift_masks1[:, ::-1])

            shift_masks1[np.bitwise_and(shift_masks1 == 1, batch_image_masks1 == 0)] = 2
            shift_masks2[np.bitwise_and(shift_masks2 == 1, batch_image_masks2 == 0)] = 2

        original_images1 = None
        if any(['image_original' in d and d['image_original'] is not None for d in data]):
            max_width = max([d['image_original'].shape[1] for d in data])
            original_images1 = np.zeros([len(data), image_height, max_width, image_channels], dtype=np.uint8)
            for batch_image, line_image in zip(original_images1, [d['image_original'] for d in data]):
                batch_image[:, :line_image.shape[1]] = line_image

        original_images2 = None
        if any(['image2_original' in d and d['image2_original'] is not None for d in data]):
            max_width = max([d['image2_original'].shape[1] for d in data])
            original_images2 = np.zeros([len(data), image_height, max_width, image_channels], dtype=np.uint8)
            for batch_image, line_image in zip(original_images2, [d['image2_original'] for d in data]):
                batch_image[:, :line_image.shape[1]] = line_image

        return (batch_images1, batch_images2, batch_image_masks1, batch_image_masks2, left_paddings1, left_paddings2,
                original_images1, original_images2, shifts, shift_masks1, shift_masks2)

    def crop_images(self, data: List[Dict]):
        shifts = []

        for d in data:
            d['image_original'] = d['image']
            d['image2_original'] = d['image2']

            d['image'], start = self.crop_image(d['image'])

            # TODO: Make the shift independent of the subsampling factor?
            min_shift = -min(start // self.subsampling_factor, self.crop_width // self.subsampling_factor - 1)
            max_shift = max(0, min((d['image_original'].shape[1] - start - self.crop_width) // self.subsampling_factor, self.crop_width // self.subsampling_factor - 1))

            if min_shift == max_shift:
                shift = min_shift
            else:
                shift = np.random.randint(min_shift, max_shift)

            start += (shift * self.subsampling_factor)

            d['image2'], _ = self.crop_image(d['image2'], start=start)

            shifts.append(shift)

        return shifts

    def crop_image(self, image, start=None):
        if image.shape[1] <= self.crop_width:
            return image, 0

        if start is None:
            diff = image.shape[1] - self.crop_width
            start = np.random.randint(0, diff) // self.crop_step
            start *= self.crop_step

        crop = image[:, start:start + self.crop_width, :]
        return crop, start

    def calculate_padded_image_width(self, image_width: int):
        return int(np.ceil(image_width / self.padding_coefficient) * self.padding_coefficient) + self.padding_coefficient


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

    dataset = Dataset(args.lmdb_path, args.lines_path, augmentations=None, pair_images=True)
    batch_creator = BatchCreator(crop_width=512, crop_step=8)
    dataloader = create_dataloader(dataset, batch_creator, batch_size=args.batch_size, num_workers=args.num_workers)

    first_batch = None
    lines_counter = 0
    for i, batch in enumerate(dataloader):
        if i == 0:
            first_batch = batch

        batch_size = batch['images'].shape[0]
        lines_counter += batch_size

        print(f"Batch #{i}, batch size: {batch_size}")

    print(f"Total number of images: {lines_counter}")
    print()

    print("First batch")
    print(f"Batch size: {args.batch_size}")
    print(f"Images shape: {first_batch['images'].shape}")
    print(f"Images2 shape: {first_batch['images2'].shape}" if first_batch['images2'] is not None else "Images2: None")
    print(f"Image masks shape: {first_batch['image_masks'].shape}")
    print(f"Image masks2 shape: {first_batch['image_masks2'].shape}" if first_batch['image_masks2'] is not None else "Image masks2: None")
    print(f"Shifts: {first_batch['shifts']}" if first_batch['shifts'] is not None else "Shifts: None")
    print(f"Shift masks shape: {first_batch['shift_masks'].shape}" if first_batch['shift_masks'] is not None else "Shift masks: None")
    print(f"Shift masks2 shape: {first_batch['shift_masks2'].shape}" if first_batch['shift_masks2'] is not None else "Shift masks2: None")
    print(f"Labels shape: {first_batch['labels'].shape}" if first_batch['labels'] is not None else "Labels: None")
    print(f"IDs: {first_batch['ids']}")
    print(f"Original images shape: {first_batch['original_images'].shape}" if first_batch['original_images'] is not None else "Original images: None")
    print(f"Original images2 shape: {first_batch['original_images2'].shape}" if first_batch['original_images2'] is not None else "Original images2: None")

    return 0


if __name__ == "__main__":
    import argparse
    from pero_pretraining.common.dataset import Dataset
    exit(main())

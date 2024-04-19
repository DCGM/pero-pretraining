import cv2
import lmdb
import logging
import numpy as np


class Dataset:
    def __init__(self, lmdb_path, lines_path, augmentations=None, pair_images=False):
        self.lmdb_path = lmdb_path
        self.lines_path = lines_path
        self.augmentations = augmentations
        self.pair_images = pair_images

        self._logger = logging.getLogger(__name__)

        self._image_ids = []
        self._labels = {}
        self._has_labels = False

        self._load_data()
        self._txn = lmdb.open(self.lmdb_path, readonly=True).begin()

    def _load_data(self):
        images_counter = 0
        labels_counter = 0

        with open(self.lines_path, "r") as file:
            for line in file:
                image_id, labels = self._parse_line(line)
                self._image_ids.append(image_id)

                images_counter += 1

                if labels is not None:
                    self._labels[image_id] = labels
                    self._has_labels = True
                    labels_counter += 1

        self._logger.info(f"Dataset '{self.lines_path}' loaded: {images_counter} image{'s' if images_counter > 0 else ''} and {labels_counter} label{'s' if labels_counter > 0 else ''}.")

    def _load_image(self, image_id):
        data = self._txn.get(image_id.encode())
        if data is None:
            self._logger.warning(f"Unable to load image '{image_id}' specified in '{self.lines_path}' from LMDB '{self.lmdb_path}'.")
            return None

        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self._logger.warning(f"Unable to decode image '{image_id}'.")
            return None

        return img

    @staticmethod
    def _parse_line(line):
        if " " in line:
            image_id, *labels = line.strip().split()
        else:
            image_id = line.strip()
            labels = None

        return image_id, labels

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, idx):
        image_id = self._image_ids[idx]
        image = self._load_image(image_id)
        labels = None
        image2 = None

        if self.augmentations is not None:
            image = self.augmentations(image=image)

        if self._has_labels:
            if image_id in self._labels:
                labels = self._labels[image_id]
            else:
                self._logger.warning(f"Labels for image {image_id} not found.")

        if self.pair_images:
            image2 = np.copy(image)
            if self.augmentations is not None:
                image2 = self.augmentations(image=image2)

        item = {
            "image": image,
            "image2": image2,
            "labels": labels,
            "image_id": image_id
        }

        return item


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb", help="Path to the LMDB.")
    parser.add_argument("--trn-lines", help="Path to the file with training lines.")
    parser.add_argument("--tst-lines", help="Path to the file with testing lines.")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    training_dataset = Dataset(args.lmdb, args.trn_lines)
    testing_dataset = Dataset(args.lmdb, args.tst_lines)

    training_sample = training_dataset[0]
    testing_sample = testing_dataset[0]

    print("Training sample")
    print(f"Image shape: {training_sample['image'].shape}")
    print(f"Labels: {training_sample['labels']}")
    print()

    print(f"Testing sample")
    print(f"Image shape: {testing_sample['image'].shape}")
    print(f"Labels: {testing_sample['labels']}")

    return 0


if __name__ == "__main__":
    import argparse
    exit(main())

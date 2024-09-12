import cv2
import lmdb
import logging
import numpy as np
import json

class Dataset:
    def __init__(self, lmdb_path, lines_path, augmentations=None, pair_images=False, max_width=2048, label_step=8, skip=0):
        self.lmdb_path = lmdb_path
        self.lines_path = lines_path
        self.augmentations = augmentations
        self.pair_images = pair_images
        self.max_width = max_width
        self.label_step = label_step

        self._logger = logging.getLogger(__name__)

        self._image_ids = []
        self._labels = {}
        self._has_labels = False

        self.skip = skip
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
        return len(self._image_ids) - self.skip 

    def __getitem__(self, idx):
        idx = idx + self.skip
        image_id = self._image_ids[idx]
        image = self._load_image(image_id)[:, :self.max_width]
        labels = None
        image2 = None

        if self._has_labels:
            if image_id in self._labels:
                labels = self._labels[image_id][:(self.max_width // self.label_step)]
            else:
                self._logger.warning(f"Labels for image {image_id} not found.")


        if self.augmentations is not None:
            image = self.augmentations(image=image)

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


class DatasetLMDB:
    def __init__(self, lmdb_path, lines_path, augmentations=None, pair_images=False, max_width=2048, label_step=8, fill_width=False, exact_width=False):
        self.lmdb_path = lmdb_path
        self.lines_path = lines_path
        self.augmentations = augmentations
        self.pair_images = pair_images
        self.max_width = max_width
        self.label_step = label_step
        self.fill_width = fill_width
        self.exact_width = exact_width

        self._logger = logging.getLogger(__name__)

        self._has_labels = False

        self._txn_labels = lmdb.open(self.lines_path, readonly=True).begin()
        self._txn = lmdb.open(self.lmdb_path, readonly=True).begin()

        self.image_count = self._txn_labels.stat()['entries']

        print("DATASET", lines_path, self.image_count)

        self._eol_patch = None

    def _load_image_and_labels(self, image_id):
        lmdb_id = f"{image_id:10d}"
        image_info = self._txn_labels.get(lmdb_id.encode())
        image_info = json.loads(image_info)
        labels = image_info["labels"]

        if "image" in image_info:
            image_id = image_info["image"]
            data = self._txn.get(image_id.encode())
            if data is None:
                self._logger.warning(f"Unable to load image '{image_id}' specified in '{self.lines_path}' from LMDB '{self.lmdb_path}'.")
                return None

            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                self._logger.warning(f"Unable to decode image '{image_id}'.")
                return None
        elif "images" in image_info:
            images = image_info["images"]
            img = []
            for image_id in images:
                data = self._txn.get(image_id.encode())
                if data is None:
                    self._logger.warning(f"Unable to load image '{image_id}' specified in '{self.lines_path}' from LMDB '{self.lmdb_path}'.")
                    return None
                image_data = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                pad = self.label_step - image_data.shape[1] % self.label_step
                pad += self.label_step
                image_data = np.concatenate([image_data, np.zeros((image_data.shape[0], pad, 3), dtype=np.uint8)], axis=1)
                img.append(image_data)
            img = np.concatenate(img, axis=1)
        else:
            self._logger.warning(f"Image/images not found in the line {image_id}.")
            return None

        return img, labels

    @staticmethod
    def _parse_line(line):
        if " " in line:
            image_id, *labels = line.strip().split()
        else:
            image_id = line.strip()
            labels = None

        return image_id, labels

    def __len__(self):
        return self.image_count

    def _get_fixed_width_image(self, image_id):
        all_images = []
        all_labels = []
        width = 0
        while True:
            image, labels = self._load_image_and_labels(image_id)
            width += image.shape[1]
            if width >= self.max_width and not self.exact_width:
                break
            if self._eol_patch is None:
                self._eol_patch = np.zeros((image.shape[0], self.label_step, 3), dtype=np.uint8)
                self._eol_patch[:, 0::3, 0] = 255
                self._eol_patch[:, 1::3, 0] = 255
                self._eol_patch[:, 2::3, 0] = 255

            # Image width must be divisible by 8 - pad it
            if image.shape[1] % self.label_step != 0:
                pad = self.label_step - image.shape[1] % self.label_step
                image = np.concatenate([image, np.zeros((image.shape[0], pad, 3), dtype=np.uint8)], axis=1)
            labels += [0]
            all_images.append(image)
            all_images.append(self._eol_patch)
            all_labels.append(labels)
            image_id = (image_id + 1) % self.image_count
            if width >= self.max_width:
                break

        image = np.concatenate(all_images, axis=1)
        labels = np.concatenate(all_labels)

        return image, labels

    def __getitem__(self, idx):
        if self.fill_width:
            image, labels = self._get_fixed_width_image(idx)
        else:
            image, labels = self._load_image_and_labels(idx)
        image = image[:, :self.max_width]
        labels = labels[:(self.max_width // self.label_step)]
        image2 = None

        if self.augmentations is not None:
            image = self.augmentations(image=image)

        if self.pair_images:
            image2 = np.copy(image)
            if self.augmentations is not None:
                image2 = self.augmentations(image=image2)

        item = {
            "image": image,
            "image2": image2,
            "labels": labels,
            "image_id": idx
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

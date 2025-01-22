import cv2
import lmdb
import argparse
import numpy as np
from random import shuffle
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, help="Input labels file")
    parser.add_argument("--ngrams", type=str, help="Input ngrams file")
    parser.add_argument("--lmdb", type=str, help="LMDB path")
    parser.add_argument("--subsampling", type=int, default=8, help="Subsampling factor")
    parser.add_argument("--crops-per-line", type=int, default=16, help="Number of crops per line")
    parser.add_argument("--lines-per-image", type=int, default=None, help="Number of lines per image")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of searched samples per n-gram")
    parser.add_argument("--output", type=str, help="Output file")
    return parser.parse_args()


def load_labels(path):
    data = {}

    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if len(line) > 0:
                line_id, *line_labels = line.split()
                data[line_id] = [int(label) for label in line_labels]

    return data


def load_ngrams(path):
    data = []

    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if len(line) > 0:
                ngram, _ = line.split("\t")
                labels = [int(label) for label in ngram.split()]
                data.append(tuple(labels))

    return data


def load_line(txn, line_id):
    return cv2.imdecode(np.frombuffer(txn.get(line_id.encode()), dtype=np.uint8), 1)


def search_ngrams(labels, ngrams, txn, subsampling, max_samples=None):
    crops = defaultdict(list)
    counts = defaultdict(int)

    ngram_size = len(ngrams[0])
    ngrams_set = set(ngrams)

    for line_id in labels:
        line_labels = labels[line_id]
        line = None

        for i in range(len(line_labels) - ngram_size + 1):
            ngram = tuple(line_labels[i:i + ngram_size])

            if ngram in ngrams_set:
                if line is None:
                    line = load_line(txn, line_id)

                crops[ngram].append(line[:, i * subsampling:(i + ngram_size) * subsampling, :])
                counts[ngram] += 1

                if max_samples is not None and counts[ngram] >= max_samples:
                    ngrams_set.remove(ngram)

        if len(ngrams_set) == 0:
            break

    return crops


def create_image(ngrams, crops, crops_per_line=16):
    rows = []

    for ngram in ngrams:
        ngram_crops = crops[ngram]

        if len(ngram_crops) > crops_per_line:
            shuffle(ngram_crops)
            ngram_crops = ngram_crops[:crops_per_line]

        elif len(ngram_crops) < 4:
            continue

        separator = np.zeros((ngram_crops[0].shape[0], 5, 3), dtype=np.uint8)

        row_crops = []
        for i, crop in enumerate(ngram_crops):
            if i > 0:
                row_crops.append(separator)

            row_crops.append(crop)

        rows.append(np.concatenate(row_crops, axis=1))

    if len(rows) == 0:
        return None

    max_width = max([row.shape[1] for row in rows])
    rows = [np.pad(row, ((0, 0), (0, max_width - row.shape[1]), (0, 0)), mode="constant") for row in rows]
    image = np.concatenate(rows, axis=0)

    return image


def main():
    args = parse_args()

    labels = load_labels(args.labels)
    print("Labels loaded.")

    ngrams = load_ngrams(args.ngrams)
    print("N-grams loaded.")

    crops = search_ngrams(labels, ngrams, lmdb.open(args.lmdb, readonly=True, lock=False).begin(), args.subsampling,
                          max_samples=args.max_samples)
    print("Crops gathered.")

    if args.lines_per_image is not None:
        counter = 0

        while len(ngrams) > 0:
            image_ngrams = ngrams[:args.lines_per_image]
            ngrams = ngrams[args.lines_per_image:]

            image = create_image(image_ngrams, crops, crops_per_line=args.crops_per_line)
            if image is None:
                print("Nothing to visualize.")
                continue

            extension = args.output.split(".")[-1]
            output_path = args.output.replace(extension, f"{counter}.{extension}")

            cv2.imwrite(output_path, image)
            print(f"Image {counter} created.")

            counter += 1

    else:
        image = create_image(ngrams, crops, crops_per_line=args.crops_per_line)
        if image is None:
            print("Nothing to visualize.")
        else:
            print("Image created.")
            cv2.imwrite(args.output, image)

    return 0


if __name__ == "__main__":
    exit(main())


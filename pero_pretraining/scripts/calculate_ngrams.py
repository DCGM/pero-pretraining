import pickle
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, help="Input file")
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument("--n", type=int, default=3, help="N-gram size")
    parser.add_argument("--top", type=int, default=None, help="If set, only top N n-grams are stored.", required=False)
    return parser.parse_args()


def load(path):
    data = {}

    with open(path, "r") as file:
        for i, line in enumerate(file):
            line = line.strip()
            if len(line) > 0:
                line_id, *line_labels = line.split()
                data[line_id] = [int(label) for label in line_labels]

    return data


def save(path, ngrams):
    with open(path, "w") as file:
        for (ngram, count) in ngrams:
            file.write(f"{' '.join([str(label) for label in ngram])}\t{count}\n")


def calculate_ngrams(data, n):
    ngrams = defaultdict(int)
    total_lines = len(data)

    for i, line_id in enumerate(data):
        line_labels = data[line_id]

        for start in range(len(line_labels) - n + 1):
            ngram = tuple(line_labels[start:start+n])
            ngrams[ngram] += 1

    return ngrams


def main():
    args = parse_args()

    lines = load(args.labels)
    print("Labels loaded.")

    ngrams = calculate_ngrams(lines, args.n)
    print("N-grams calculated.")

    sorted_ngrams = sorted(ngrams.items(), key=lambda item: item[1], reverse=True)

    if args.top is not None:
        sorted_ngrams = sorted_ngrams[:args.top]

    save(args.output, sorted_ngrams)
    print("N-grams saved.")


if __name__ == "__main__":
    exit(main())

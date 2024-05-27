import argparse
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from pero_pretraining.scripts.common import load_pickle, save_pickle


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to a file with the pickled features")
    parser.add_argument("--k", help="Number of clusters ('K'-means).", default=4096, required=False, type=int)
    parser.add_argument("--batch-size", help="Batch size.", default=2 ** 14, required=False, type=int)
    parser.add_argument("--iters", help="Number of iterations over dataset (epochs).", default=100, required=False, type=int)
    parser.add_argument("--output", help="Path to the output file.")

    args = parser.parse_args()
    return args


def fit(dataset_file, k, batch_size=2 ** 14, epochs=100):
    kmeans = MiniBatchKMeans(n_clusters=k, init="k-means++", batch_size=batch_size, max_iter=epochs, n_init=10)

    vectors = load_pickle(dataset_file)
    print(f"Loaded '{dataset_file}' ({len(vectors)})")

    np.random.shuffle(vectors)
    print(f"Shuffled")

    kmeans = kmeans.fit(vectors)
    print(f"Inertia:{kmeans.inertia_}")

    return kmeans


def main():
    args = parse_arguments()

    k_means = fit(args.dataset, args.k, batch_size=args.batch_size, epochs=args.epochs)
    print("K-means trained")

    save_pickle(k_means, args.output)
    print(f"K-means saved to '{args.output}'")

    return 0


if __name__ == "__main__":
    exit(main())

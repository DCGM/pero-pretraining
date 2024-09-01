import argparse
import numpy as np
import time
import torch
from pykeops.torch import LazyTensor

use_cuda = torch.cuda.is_available()
dtype = torch.float32 
device_id = "cuda:0" if use_cuda else "cpu"


def KMeans(x, K=10, Niter=10, c=None, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    if c is None:
        c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    last_assignment = None
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster
        
        change = (last_assignment != cl).sum().item() if last_assignment is not None else 0
        last_assignment = cl
        # compute min, median, mean, and max number of points per cluster
        cluter_sizes = torch.bincount(cl).type_as(c)
        print(i, change, D_ij.min(dim=1).mean().item(),
              f"min: {cluter_sizes.min().item()}, median: {cluter_sizes.median().item()}, "
              f"mean: {cluter_sizes.mean().item()}, max: {cluter_sizes.max().item()}")

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )
    return cl, c


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to a file with the pickled features")
    parser.add_argument("--k", help="Number of clusters ('K'-means).", default=4096,  type=int)
    parser.add_argument("--data-size", help="The number of samples to use. If None, all samples are used.", default=None, required=False, type=int)
    parser.add_argument("--iters", help="Number of iterations over dataset .", default=100, required=False, type=int)
    parser.add_argument("--output", help="Path to the output file.")
    parser.add_argument("--test", help="Test the implementation with generated data.", action="store_true")
    args = parser.parse_args()
    return args




def main():
    args = parse_arguments()

    if args.test:
        N = 100000 if args.data_size is None else args.data_size
        D = 512
        vectors = torch.randn(N, D, dtype=dtype, device=device_id)
    else:
        vectors = np.load(args.dataset)
    print(f"DATA '{args.dataset}' ({len(vectors)})")

    #np.random.shuffle(vectors)
    print(f"SHUFFLED")

    vectors = torch.tensor(vectors, dtype=dtype, device=device_id)

    cl, c = KMeans(vectors, K=args.k, Niter=args.iters, verbose=True)

    print(cl.shape)
    print(c.shape)

    c = c.cpu().numpy()

    np.save(args.output, c)

    return 0


if __name__ == "__main__":
    exit(main())

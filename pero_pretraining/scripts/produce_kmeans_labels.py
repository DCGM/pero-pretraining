import torch
import argparse

from safe_gpu.safe_gpu import GPUOwner

import numpy as np
from pero_pretraining.scripts.common import init_model, init_dataset, load_pickle
from pero_pretraining.autoencoders.batch_operator import BatchOperator


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", help="Path to the model checkpoint.")
    parser.add_argument("--model-definition", help="Definition of the model.", default="{}")
    parser.add_argument("--kmeans-path", help="Path to the K-Means model.")
    parser.add_argument("--lines-path", help="Path to the file with lines.")
    parser.add_argument("--lmdb-path", help="Path to the LMDB.")
    parser.add_argument("--batch-size", help="Batch size.", default=32, required=False, type=int)
    parser.add_argument("--output", help="Path to the output file.")

    args = parser.parse_args()
    return args


def compute_features(model, dataset, kmeans_model):
    data = {}
    device = next(model.parameters()).device

    batch_operator = BatchOperator(device)

    with torch.no_grad():
        for batch in dataset:
            images = batch_operator.prepare_batch(batch)

            features = model(images)

            if len(features.shape) == 4:
                features = features.squeeze(2)
            print(features.shape, kmeans_model.shape)

            features = features.permute(0, 2, 1)
            features = features.cpu().numpy()

            for line_id, line_image_mask, line_features in zip(batch['ids'], batch['image_masks'], features):
                line_features = line_features[line_image_mask == 1]
                labels = kmeans_model.predict(line_features)

                data[line_id] = labels.tolist()

    return data


def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = init_model(args.model_definition, args.checkpoint_path, device)
    print("Model loaded")

    kmeans_model = np.load(args.kmeans_path)
    print("K-Means Model loaded")

    dataset = init_dataset(args.lmdb_path, args.lines_path, args.batch_size)
    print("Dataset loaded")

    labels = compute_features(model, dataset, kmeans_model)
    print(f"Labels computed ({len(labels)})")

    save_labels(labels, args.output)
    print(f"Labels saved to {args.output}")

    return 0


if __name__ == "__main__":
    gpu_owner = GPUOwner()
    exit(main())

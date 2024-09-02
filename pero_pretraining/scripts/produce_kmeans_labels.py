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


def compute_features(model, dataset, kmeans_model, output_path):
    data = {}
    device = next(model.parameters()).device

    batch_operator = BatchOperator(device)

    output_file = open(output_path, 'w')
    kmeans_model = kmeans_model.reshape(1, kmeans_model.shape[0], kmeans_model.shape[1])

    counter = 0
    with torch.no_grad():
        for batch in dataset:
            images = batch_operator.prepare_batch(batch)
            counter += images.shape[0]

            features = model(images)

            if len(features.shape) == 4:
                features = features.squeeze(2)
            print(counter, features.shape)

            # Feature shape is (batch_size, num_features, sequence_length)
            # kmeans_model shape is (num_clusters, num_features)

            # Compute feature-cluster assingments by using L2 distance and taking minimal distances

            # Compute L2 distance between each feature and each cluster center
            # features shape: (batch_size, num_features, sequence_length)
            # kmeans_model shape: (num_clusters, num_features)
            # distances shape: (batch_size, num_clusters, sequence_length)

            features = features.permute(0, 2, 1)
            features_linear = features.reshape(-1, features.shape[-1])
            distances = torch.cdist(features_linear, kmeans_model).squeeze()
            #print(distances.shape, features_linear.shape, kmeans_model.shape)
            assignment = torch.argmin(distances, dim=1)
            #print(assignment.shape, assignment)
            # reshape back to (batch_size, sequence_length)
            assignment = assignment.reshape(features.shape[0], features.shape[1])
            assignment = assignment.cpu().numpy()
            #print(assignment.shape)

            for line_id, line_image_mask, line_ids in zip(batch['ids'], batch['image_masks'], assignment):
                line_ids = line_ids[line_image_mask == 1]
                print(line_id, ' '.join([str(label) for label in line_ids]), file=output_file)

    output_file.close()
    return data


def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = init_model(args.model_definition, args.checkpoint_path, device)
    print("Model loaded")

    kmeans_model = np.load(args.kmeans_path)
    kmeans_model = torch.from_numpy(kmeans_model).float().to(device)
    print("K-Means Model loaded")

    dataset = init_dataset(args.lmdb_path, args.lines_path, args.batch_size)
    print("Dataset loaded")

    labels = compute_features(model, dataset, kmeans_model, args.output)
    print(f"Labels computed ({len(labels)})")

    return 0


if __name__ == "__main__":
    gpu_owner = GPUOwner()
    exit(main())

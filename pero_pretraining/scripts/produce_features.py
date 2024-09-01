import torch
import argparse
import numpy as np

from safe_gpu.safe_gpu import GPUOwner

from pero_pretraining.scripts.common import init_model, init_dataset, save_pickle, save_numpy
from pero_pretraining.autoencoders.batch_operator import BatchOperator


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", help="Path to the model checkpoint.")
    parser.add_argument("--model-definition", help="Definition of the model.", default="{}")
    parser.add_argument("--lines-path", help="Path to the file with lines.")
    parser.add_argument("--lmdb-path", help="Path to the LMDB.")
    parser.add_argument("--batch-size", help="Batch size.", default=32, required=False, type=int)
    parser.add_argument("--output-type", help="Type of the output.", default="numpy", choices=["numpy", "pickle"])
    parser.add_argument("--output", help="Path to the output file.")

    args = parser.parse_args()
    return args


def compute_features(model, dataset):
    all_features = []
    device = next(model.parameters()).device

    batch_operator = BatchOperator(device)

    with torch.no_grad():
        for batch in dataset:
            images = batch_operator.prepare_batch(batch)

            features = model.encoder(images)

            if len(features.shape) == 4:
                features = features.squeeze(2)

            features = features.permute(0, 2, 1)
            features = features.cpu().numpy()
            features = features[batch["image_masks"] == 1]

            all_features.append(features)

    all_features = np.vstack(all_features)

    return all_features


def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = init_model(args.model_definition, args.checkpoint_path, device)
    print("Model loaded")

    dataset = init_dataset(args.lmdb_path, args.lines_path, args.batch_size)
    print("Dataset loaded")

    features = compute_features(model, dataset)
    print(f"Features computed ({features.shape})")

    if args.output_type == "numpy":
        save_numpy(features, args.output)
    else:
        save_pickle(features, args.output)

    print(f"Features saved to {args.output} ({args.output_type})")

    return 0


if __name__ == "__main__":
    gpu_owner = GPUOwner()
    exit(main())

import sys
import torch
import argparse

from safe_gpu.safe_gpu import GPUOwner

from pero_pretraining.scripts.common import init_model, init_dataset, save_labels
from pero_pretraining.autoencoders.batch_operator import BatchOperator


def parse_arguments():
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument("--lines-path", help="Path to the input file.", required=True)
    parser.add_argument("--lmdb-path", help="Path to the lmdb.", required=True)
    parser.add_argument("--model", help="JSON string with model definition.", required=True)
    parser.add_argument("--batch-size", help="Batch size.", required=False, default=32, type=int)
    parser.add_argument("--checkpoint-path", help="Path to the checkpoint.", required=True)
    parser.add_argument("--labels-path", help="Path to the output file.", required=True)
    parser.add_argument("--widths-path", help="Path to the file with line widths.", required=True)

    args = parser.parse_args()
    return args


def compute_labels(model, dataset):
    data = {}
    device = next(model.parameters()).device

    batch_operator = BatchOperator(device)

    with torch.no_grad():
        for batch in dataset:
            images = batch_operator.prepare_batch(batch)

            tokens, labels  = model.quantize(model.encode(images))

            N, _, _, T = tokens.shape
            labels = labels.reshape(N, T)
            labels = labels.cpu().numpy()

            for line_id, line_image_mask, line_labels in zip(batch['ids'], batch['image_masks'], labels):
                data[line_id] = line_labels[line_image_mask == 1].tolist()

    return data


def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = init_model(args.model, args.checkpoint_path, device)
    print("Model loaded")

    dataset = init_dataset(args.lmdb_path, args.lines_path, args.batch_size)
    print("Dataset loaded")

    labels = compute_labels(model, dataset)
    print(f"Labels computed ({len(labels)})")

    save_labels(labels, args.output)
    print(f"Labels saved to {args.output}")

    return 0



if __name__ == "__main__":
    gpu_owner = GPUOwner()
    exit(main())

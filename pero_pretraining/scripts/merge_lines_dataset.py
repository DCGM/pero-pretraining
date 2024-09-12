import argparse
import lmdb
import json
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Takes label LMDB and concatenates lines to get minimum line length."
                                            "Saves the concateted labels to a new LMDB.")
    parser.add_argument("--input", required=True, help="Path to LMDB with labels.")
    parser.add_argument("--trn-output", required=True, help="Path to the training output LMDB.")
    parser.add_argument("--tst-output", required=True, help="Path to the training output LMDB.")
    parser.add_argument("--tst-target-size", type=int, default=1000, help="Size of the testing dataset.")
    parser.add_argument("--min-length", type=int, default=320, help="Minimum line length.")
    parser.add_argument("--separator", type=int, default=0, help="Separator for the labels.")
    args = parser.parse_args()
    return args


def estimate_concatenated_lines_count(txn: lmdb.Transaction, min_length: int, sample_size: int = 10000):
    input_size = txn.stat()["entries"]
    counter = 0
    current_length = 0
    for i in tqdm(range(0, input_size, int(input_size / sample_size))):
        labels = json.loads(txn.get(f"{i:10d}".encode()))["labels"]
        current_length += len(labels) + 1
        if current_length >= min_length:
            counter += 1
            current_length = 0

    return counter * input_size / sample_size



def main():
    args = parse_arguments()

    trn_env = lmdb.open(args.trn_output, map_size=1000000000000)
    trn_txn = trn_env.begin(write=True)

    tst_env = lmdb.open(args.tst_output, map_size=1000000000000)
    tst_txn = tst_env.begin(write=True)

    in_env = lmdb.open(args.input, readonly=True)
    estimaged_line_count = estimate_concatenated_lines_count(in_env.begin(write=False), args.min_length)
    in_txn = in_env.begin(write=False)
    input_size = in_txn.stat()["entries"]

    print(f"Estimated number of lines after concatenation {estimaged_line_count} from {input_size} lines.")
    tst_count = 0
    trn_count = 0

    next_sample = {"images": [], "labels": []}

    length_sum = 0

    # estimate the number of lines after concatenation

    for i, (key, value) in enumerate(in_txn.cursor()):
        sample = json.loads(value)
        next_sample["images"] += [sample["image"]]
        try:
            next_sample["labels"] += [int(l) for l in sample["labels"]]
        except ValueError as e:
            print(f"Skipping line {i} due to error: {e}")
            print(f"Line: {sample}")
            continue

        next_sample["labels"] += [args.separator]
        if len(next_sample["labels"]) >= args.min_length:

            # write equidistantly to testing to reach target test size
            total_output_count = tst_count + trn_count
            if tst_txn and tst_count < args.tst_target_size and tst_count / (total_output_count + 1) < args.tst_target_size / estimaged_line_count:
                tst_txn.put(f"{tst_count:10d}".encode(), json.dumps(next_sample).encode())
                tst_count += 1
            else:
                trn_txn.put(f"{trn_count:10d}".encode(), json.dumps(next_sample).encode())
                trn_count += 1

            length_sum += len(next_sample["labels"])
            next_sample = {"images": [], "labels": []}
            if total_output_count % 10000 == 0 and total_output_count > 0:
                print(f"Processed {i} lines. Average length: {length_sum / total_output_count}, {tst_count} test samples, {trn_count} training samples.")

    trn_txn.commit()
    trn_env.close()
    tst_txn.commit()
    tst_env.close()
    in_env.close()


if __name__ == "__main__":
    main()

# Convert text file with image names and token labels to LMDB.
# The format is some_file_name.jpg token1 token2 token3 ...
# The LMDB is indexed by image order in the input text file.

import argparse
import lmdb
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert text file with image names and token labels to LMDB. "
                                                 "The LMDB is then used in training dataset to allow for fast random access "
                                                 "without the need to load the whole dataset into memory.")
    parser.add_argument("--input", required=True, help="Path to the input text file.")
    parser.add_argument("--output", required=True, help="Path to the output LMDB.")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    env = lmdb.open(args.output, map_size=1000000000000)
    txn = env.begin(write=True)

    with open(args.input, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split(" ")
            image_path = parts[0]
            labels = parts[1:]
            txn.put(f"{i:10d}".encode(), json.dumps({"image": image_path, "labels": labels}).encode())
            if i % 1000 == 0:
                print(f"Processed {i} lines")
    txn.commit()
    env.close()

    return


if __name__ == "__main__":
    main()

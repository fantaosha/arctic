import subprocess
import argparse
from typing import List


def main():
    parser = argparse.ArgumentParser(description="Mesh Fitting")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seqs", type=str, required=True)
    parser.add_argument("--exps", type=str, required=True)

    args = parser.parse_args()

    seqs: List[str] = args.seqs.split(",")
    dataset: str = args.dataset
    exps: str = args.exps.split(",")

    command = ["python", "fitting.py"]

    exp_args = [f"+experiment=proto/{exp}" for exp in exps]

    for seq in seqs:
        print(f"Start fitting {seq}.")
        common_args = [f"dataset={dataset}", f"sequence={seq}"]
        for exp_arg in exp_args:
            subprocess.run(command + [exp_arg] + common_args)
        print(f"Fitted {seq}.")


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path

from arena_humansim.utils.evaluation.partitions import run_partitions


def main() -> None:
    parser = argparse.ArgumentParser(description="K per bucket under alternative driver-class partitions.")
    parser.add_argument("--recordings_dir", type=str, required=True, help="analyzed sweep dir holding pairwise_distance.csv")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--ci_seed", type=int, default=0)
    args = parser.parse_args()
    run_partitions(Path(args.recordings_dir).resolve(), Path(args.out_dir).resolve() if args.out_dir else None, args.n_bootstrap, args.ci_seed)


if __name__ == "__main__":
    main()

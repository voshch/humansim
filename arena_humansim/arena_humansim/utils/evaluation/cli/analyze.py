import argparse
from pathlib import Path

from arena_humansim.utils.evaluation.analyze import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise driver-divergence analysis (nav/bt/het buckets).")
    parser.add_argument("--recordings_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="CSV output dir (default: recordings_dir)")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="bootstrap iterations for K confidence interval")
    parser.add_argument("--ci_seed", type=int, default=0,
                        help="RNG seed for bootstrap (reproducibility)")
    parser.add_argument("--framing_threshold", type=float, default=1.2,
                        help="multiplier above K_nav for a stressor bucket to 'clear'")
    args = parser.parse_args()

    recordings_dir = Path(args.recordings_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else recordings_dir

    run_analysis(
        recordings_dir=recordings_dir,
        out_dir=out_dir,
        n_bootstrap=args.n_bootstrap,
        ci_seed=args.ci_seed,
        framing_threshold=args.framing_threshold,
    )


if __name__ == "__main__":
    main()

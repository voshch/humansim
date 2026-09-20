import argparse
from pathlib import Path

from arena_humansim.utils.evaluation.cli._resume import latest_sweep_dir
from arena_humansim.utils.evaluation.eth_gap import run_eth_gap


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory-level gap analysis of a divergence sweep against real pedestrian data (ETH/EWAP, optionally ATC).")
    parser.add_argument("--recordings_dir", type=str, default=None, help="Sweep dir (default: newest sweep under $ARENA_DATA_DIR/peds).")
    parser.add_argument("--ewap", type=str, default=None, help="Directory holding seq_eth/ and seq_hotel/ (default: fetch the light EWAP tarball into ~/.cache).")
    parser.add_argument("--atc", type=str, default=None, help="One ATC day file (atc-YYYYMMDD.csv); one local clock hour of it is added as a second real-data source.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output dir (default: the sweep dir).")
    args = parser.parse_args()

    if args.recordings_dir:
        run_dir = Path(args.recordings_dir).resolve()
    else:
        found = latest_sweep_dir(incomplete_only=False)
        if found is None:
            parser.error("no sweep found under $ARENA_DATA_DIR/peds; pass --recordings_dir")
        run_dir = found[0]
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir
    run_eth_gap([run_dir], out_dir, Path(args.ewap).resolve() if args.ewap else None, Path(args.atc).resolve() if args.atc else None)


if __name__ == "__main__":
    main()

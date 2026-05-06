import argparse
import signal
import sys
import time
from pathlib import Path

from tqdm import tqdm

BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
POLL_INTERVAL = 0.5
INITIAL_READ_TIMEOUT = 5.0


def read_progress(path: Path) -> tuple[int, int]:
    try:
        text = path.read_text().strip()
        cur, tot = text.split("/")
        return int(cur), int(tot)
    except (FileNotFoundError, ValueError, OSError):
        return 0, 0


def _initial_read(path: Path, deadline: float) -> tuple[int, int]:
    # Block briefly so we can seed `total=` from the progress file; otherwise the bar starts as 0/0 until the first poll tick.
    while time.monotonic() < deadline:
        cur, tot = read_progress(path)
        if tot > 0:
            return cur, tot
        time.sleep(0.05)
    return read_progress(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="tqdm bars over per-worker progress files plus a grand total.")
    parser.add_argument("progress_files", nargs="+")
    parser.add_argument("--total_done", type=int, default=None, help="Trials already complete before this run started (resume offset for the Total bar).")
    parser.add_argument("--total_grand", type=int, default=None, help="Total trials in the full sweep (constant denominator for the Total bar). Defaults to sum of per-worker totals.")
    args = parser.parse_args()

    paths = [Path(p) for p in args.progress_files]
    desc_width = max(len("Total"), *(len(f"Worker {i}") for i in range(len(paths))))

    deadline = time.monotonic() + INITIAL_READ_TIMEOUT
    initial_states = [_initial_read(p, deadline) for p in paths]

    bars = [
        tqdm(
            total=tot,
            initial=0,
            desc=f"Worker {i}".ljust(desc_width),
            position=i,
            bar_format=BAR_FORMAT,
            leave=True,
        )
        for i, (_cur, tot) in enumerate(initial_states)
    ]

    total_done = args.total_done if args.total_done is not None else 0
    total_grand = args.total_grand if args.total_grand is not None else sum(tot for _, tot in initial_states) + total_done
    total_bar = tqdm(
        total=total_grand,
        initial=total_done,
        desc="Total".ljust(desc_width),
        position=len(paths),
        bar_format=BAR_FORMAT,
        leave=True,
    )

    def sync(bar: tqdm, cur: int, tot: int) -> None:
        if tot != bar.total:
            bar.total = tot
        delta = cur - bar.n
        if delta > 0:
            bar.update(delta)
        # Repaint every poll so {elapsed}/{remaining} keep ticking during long trials.
        bar.refresh()

    def render() -> None:
        cur_total = 0
        for bar, path in zip(bars, paths, strict=True):
            cur, tot = read_progress(path)
            sync(bar, cur, tot)
            cur_total += cur
        sync(total_bar, total_done + cur_total, total_grand)

    def cleanup(*_: object) -> None:
        render()
        for bar in bars:
            bar.close()
        total_bar.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    while True:
        render()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

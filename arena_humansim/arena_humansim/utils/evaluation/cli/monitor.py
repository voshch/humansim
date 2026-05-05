import signal
import sys
import time
from pathlib import Path

from tqdm import tqdm

BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
POLL_INTERVAL = 0.5


def read_progress(path: Path) -> tuple[int, int]:
    try:
        text = path.read_text().strip()
        cur, tot = text.split("/")
        return int(cur), int(tot)
    except (FileNotFoundError, ValueError, OSError):
        return 0, 0


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("usage: monitor <progress_file>...", file=sys.stderr)
        sys.exit(1)

    desc_width = max(len("Total"), *(len(f"Worker {i}") for i in range(len(paths))))

    bars = [
        tqdm(
            total=0,
            desc=f"Worker {i}".ljust(desc_width),
            position=i,
            bar_format=BAR_FORMAT,
            leave=True,
        )
        for i in range(len(paths))
    ]
    total_bar = tqdm(
        total=0,
        desc="Total".ljust(desc_width),
        position=len(paths),
        bar_format=BAR_FORMAT,
        leave=True,
    )

    def sync(bar: tqdm, cur: int, tot: int) -> None:
        if tot != bar.total:
            bar.total = tot
            bar.refresh()
        delta = cur - bar.n
        if delta > 0:
            bar.update(delta)

    def render() -> None:
        cur_total = 0
        tot_total = 0
        for bar, path in zip(bars, paths, strict=True):
            cur, tot = read_progress(path)
            sync(bar, cur, tot)
            cur_total += cur
            tot_total += tot
        sync(total_bar, cur_total, tot_total)

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

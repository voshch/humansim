import argparse
from pathlib import Path

from arena_humansim.utils.evaluation.correspondence import REFERENCES, run_correspondence


def main() -> None:
    parser = argparse.ArgumentParser(description="Correspondence tests of released drivers against public reference implementations on held-out scenarios.")
    parser.add_argument("--drivers", nargs="*", default=list(REFERENCES), help="drivers to test (default: all six)")
    parser.add_argument("--out_dir", type=str, default=None, help="where to write correspondence.csv and correspondence_summary.csv")
    args = parser.parse_args()

    import rclpy

    from arena_humansim.utils.loggable import Loggable

    rclpy.init()
    node = rclpy.create_node("correspondence")
    Loggable.init_logging(node)
    try:
        table = run_correspondence(args.drivers, Path(args.out_dir).resolve() if args.out_dir else None)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    print(table.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()

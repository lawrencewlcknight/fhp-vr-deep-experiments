"""Local/Batch entry points; training never needs another algorithm's outputs."""

import argparse
import logging
from .aggregate import aggregate
from .config import THREADS
from .train import run_worker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "worker", "aggregate"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--remote-uri")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.action == "worker":
        result = run_worker(args.output_root, args.task_index, threads=args.threads, remote_uri=args.remote_uri)
    elif args.action == "aggregate":
        result = aggregate(args.output_root)
    else:
        run_worker(args.output_root, 0, smoke=True, threads=args.threads)
        result = aggregate(args.output_root, smoke=True)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

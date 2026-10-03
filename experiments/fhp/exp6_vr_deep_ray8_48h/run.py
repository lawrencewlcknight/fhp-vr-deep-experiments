"""Train/continue independent seeds; evaluation is intentionally separate."""

import argparse
import logging

from .aggregate import aggregate
from .config import THREADS
from .train import run_worker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "worker", "resume", "aggregate"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task-index", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--remote-uri")
    parser.add_argument("--resume-from", help="Trusted training_state directory, not a playable-policy .pt")
    parser.add_argument("--resume-run-id")
    parser.add_argument("--start-hours", type=int, default=0)
    parser.add_argument("--target-hours", type=int, default=48)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    segment = dict(start_hours=args.start_hours, target_hours=args.target_hours, resume_run_id=args.resume_run_id)
    if args.action == "resume" and not args.resume_from:
        parser.error("resume requires --resume-from, --start-hours and --resume-run-id")
    if args.action == "aggregate":
        result = aggregate(args.output_root, **segment)
    elif args.action in ("worker", "resume"):
        result = run_worker(args.output_root, args.task_index, threads=args.threads,
                            remote_uri=args.remote_uri, resume_from=args.resume_from, **segment)
    else:
        if args.resume_from or args.resume_run_id or args.start_hours or args.target_hours != 48:
            parser.error("smoke uses the fresh 48-hour protocol with compressed thresholds")
        run_worker(args.output_root, 0, smoke=True, threads=args.threads)
        result = aggregate(args.output_root, smoke=True)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

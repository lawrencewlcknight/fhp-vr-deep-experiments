"""Train three fresh seeds; smoke and aggregate without running the match league."""
import argparse
from . import config
from .train import run_worker
from .aggregate import aggregate

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("smoke", "worker", "aggregate"))
    p.add_argument("--output-root", required=True)
    p.add_argument("--task-index", type=int, choices=(0,1,2), default=0)
    p.add_argument("--threads", type=int, default=config.THREADS)
    p.add_argument("--remote-uri")
    a = p.parse_args()
    if a.action == "aggregate":
        result = aggregate(a.output_root)
    elif a.action == "smoke":
        run_worker(a.output_root, 0, smoke=True, threads=a.threads)
        result = aggregate(a.output_root, smoke=True)
    else:
        result = run_worker(a.output_root, a.task_index, threads=a.threads, remote_uri=a.remote_uri)
    print(result)

if __name__ == "__main__":
    main()

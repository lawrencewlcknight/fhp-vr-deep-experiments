"""Guarded entry point for the proposed first experiment."""

from __future__ import annotations

import argparse

from .config import APPROVAL_STATUS, validate_proposal


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="validate the proposal without starting training",
    )
    args = parser.parse_args(argv)
    validate_proposal()
    if args.validate_config:
        return 0
    raise RuntimeError(
        "Experiment 1 production execution is disabled while configuration "
        f"approval_status={APPROVAL_STATUS!r}. Approve the proposal before the "
        "training orchestration is enabled."
    )


if __name__ == "__main__":
    raise SystemExit(main())

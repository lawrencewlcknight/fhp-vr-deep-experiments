from pathlib import Path


def test_no_generated_or_large_artifacts_are_tracked():
    root = Path(__file__).resolve().parents[1]
    forbidden_parts = {"__pycache__", ".pytest_cache", ".ruff_cache"}
    forbidden_suffixes = {".pyc", ".pkl", ".pt", ".pth", ".npz", ".log"}
    offenders = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(root)
        if any(part in forbidden_parts for part in relative.parts):
            offenders.append(str(relative))
        elif path.suffix.lower() in forbidden_suffixes:
            offenders.append(str(relative))
    assert offenders == []

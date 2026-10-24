"""Pinned UCV Experiment 38 rich AE representation; no learner dependencies."""
import hashlib
import json
from pathlib import Path
from .strategic_features import StrategicFeatureEncoder

ENCODER_ID = "fhp_hand_board_strategic_ae_v1"

def make_encoder():
    return StrategicFeatureEncoder("AE")

def encoder_from_metadata(metadata):
    encoder = make_encoder()
    if metadata != encoder.metadata():
        raise ValueError("Rich AE encoder metadata does not match the frozen contract")
    return encoder

def provenance():
    root = Path(__file__).parent
    record = json.loads((root / "provenance.json").read_text())
    for name, hashes in record["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != hashes["vendored_sha256"]:
            raise ValueError(f"Rich AE source integrity mismatch: {name}")
    return record

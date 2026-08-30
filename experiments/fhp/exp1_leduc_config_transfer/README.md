# Experiment 1: Leduc configuration transfer

Status: **configuration pending approval**.

This experiment will train VR-DeepDCFR+ and VR-DeepPDCFR+ on the canonical FHP
game. The proposed training hyperparameters are the paper/Table 2 Leduc values
used in the Leduc ESCHER-architecture repository. The only change inside that
training configuration is `game_name: leduc_poker -> FHP`.

The proposed observation protocol adds paired seeds, 6-hour and 12-hour policy
checkpoints, and sampled seat-swapped head-to-head evaluation. These settings
are kept separate from the transferred training configuration.

Until approval, the entry point only validates the proposal:

```bash
python -m experiments.fhp.exp1_leduc_config_transfer.run --validate-config
```

Calling it without `--validate-config` deliberately refuses to train.

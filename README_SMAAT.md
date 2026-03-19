# SmaAt backbone integration (minimal DiffCast changes)

This folder contains upstream DiffCast with one minimal extension:
- Added a new backbone option `--backbone smaat` in `run.py`.
- Added `models/smaat/` implementing a DiffCast-compatible deterministic predictor (`predict(frames_in, frames_gt=None, compute_loss=False)`).
- Added diffusion training support for `--use_diff` with the same `predict(..., compute_loss=True)` contract.
- Temporal skip features are fused by a time-wise mean before decoding, so the backbone uses all observed frames instead of only the last one.

## What was intentionally changed
- `diffcast.py`: only training-path additions and bug fixes needed for `--use_diff` training with backbone residual diffusion.
- `run.py`: removed hardcoded GPU pinning and passed explicit `T_out` to `predict(...)` during training.

## What was intentionally *not* changed
- `datasets/` loaders and SEVIR split behavior.
- `utils/metrics.py` evaluation logic.

## Usage
1. Install original DiffCast dependencies (including `ema_pytorch`, `accelerate`, `diffusers`).
2. Set SEVIR path in `datasets/get_datasets.py` (`DATAPATH['sevir']`).
3. Train deterministic backbone:

```bash
python run.py --backbone smaat
```

4. Train/evaluate with diffusion:

```bash
python run.py --backbone smaat --use_diff
python run.py --backbone smaat --use_diff --eval --ckpt_milestone <path_to_ckpt>
```

## Sanity check
Run the local contract/smoke script before long training runs:
```bash
python ../../scripts/diffcast_smaat_contract_check.py
```

## Notes
- The current DiffCast training path keeps `alpha=0.5`; with that fixed value, swapping the deterministic and diffusion weights does not change the scalar total loss, but it still matters if you later expose or tune `alpha`.

## Server Loop
For reproducible baseline-vs-candidate runs with iteration audits, use:
- `../../SERVER_EXPERIMENT_LOOP.md`
- `../../scripts/server_preflight.py`
- `../../scripts/run_iteration_example.py`
- `../../scripts/extract_diffcast_metrics.py`
- `../../scripts/audit_iteration.py`

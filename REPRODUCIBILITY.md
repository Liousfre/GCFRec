# Reproducibility notes

This document summarises the artifacts needed to run the experiments reported in the manuscript.

## Code

- Proposed method: `src/gcfrec.py`
- Backbone and baselines: `src/adrec.py`, `src/diffurec.py`, `src/dreamrec.py`, `src/sasrec.py`, `src/bert4rec.py`, `src/gru4rec.py`, and `src/eulerformer.py`
- Training and evaluation: `src/main.py`, `src/trainer.py`, and `src/metrics.py`
- Shared configuration: `src/config.yaml`

## Data

Preprocessed, fixed experiment partitions are stored in `datasets/data/<dataset>/dataset.pkl` for Baby, Beauty, Toys, Sports, and MovieLens-100K.

### Sources

- **Amazon Baby, Beauty, Toys and Games, and Sports and Outdoors:** Amazon Review Data distributed by the McAuley Lab at UCSD: <https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html>. The experiments use ratings-only interaction sequences from these four categories.
- **MovieLens-100K:** GroupLens Research: <https://grouplens.org/datasets/movielens/100k/>.

### Preprocessing and partitions

Users and items with fewer than five interactions are iteratively removed until every retained user and item has at least five interactions. Each user's remaining interactions are ordered chronologically and serialized as three aligned lists:

- `train[u]`: all but the final two interactions of user `u`;
- `val[u]`: the user's penultimate interaction, stored as a one-item list;
- `test[u]`: the user's final interaction, stored as a one-item list.

Thus, validation uses `train[u]` as history, while testing uses `train[u] + val[u]` as history. The same processed partitions are used by every compared method.

Current SHA-256 checksums of the packaged processed datasets:

```text
baby     bf8f3485f9e36763e07f0234e0ddb479cb023358797615388a36f0d7238cea99
beauty   a2f6ca22b3efe2bc5030ef9a7c4f6c928db2086771d58d9a40147e660f574bc1
toys     54500b03bd7cc15ae54181d25fd8612339f4266b1693c7fea01272d4266df882
sports   d2564386d64a5a5c7b57e11d674e388ca917d7444d4bf24d9e2528b36200587d
ml-100k  dfc5dfaa7950d1a85af4fd6bb168c38efc8213092a9f72563c6d75ab00cef8d0
```

## Configuration

- Main random seeds: `2025`, `2026`, `2027`, `2028`, and `2029`
- Default hidden size: 128
- Default sequence length: 50
- Default diffusion steps: 32
- Default GCFRec fusion: Gate-S (`gcf_gate_mode: scalar`)
- Default frozen encoder: SASRec+ (`phi_model: sasrec`)
- Default gate bias: 4.0
- Paper hardware: Ubuntu server with two NVIDIA L40 GPUs (46 GB each)
- Python: 3.10.0
- CUDA: 12.4

## Checkpoints and results

Pretrained sequence-encoder checkpoints are present under `src/saved/`.

## Quick verification

From the repository root:

```bash
python -m compileall -q src
cd src
python main.py --model gcfrec --dataset baby --device cpu --epochs 1 --batch_size 8
```

The compilation check verifies imports and syntax only. The one-epoch CPU command verifies that the packaged data and SASRec+ checkpoint can enter the training pipeline; it is not expected to reproduce paper accuracy.

For the backbone compatibility analysis, run GCFRec with
`--phi_model {sasrec,bert4rec,gru4rec,eulerformer}`. Each value loads the
corresponding packaged checkpoint while leaving the trainable path and fusion
operator unchanged.

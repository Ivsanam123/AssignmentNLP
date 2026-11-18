# BioBERT baseline on the BITS Hyderabad HPC

The modern half of the classical-vs-modern comparison. Trains
`dmis-lab/biobert-base-cased-v1.1` for token classification on the *same*
`data/{train,dev,test}.json` as the CRF, and scores it with the *same*
`ner_metrics.py`, so the F1 numbers are directly comparable to the CRF's 0.8076.

## Files

| File | Purpose |
|---|---|
| `biobert_baseline.py` | Training, dev-based epoch selection, single test evaluation. |
| `setup_env.sh` | One-time venv + torch + model download. **Login node only.** |
| `job_smoke.sh` | 20-minute sanity job: 200 sentences, 1 epoch. Run this first. |
| `job_biobert.sh` | The real 2-hour job: 5 epochs on the full training set. |
| `compare_models.py` | CRF vs BioBERT table, overall and by training frequency. |

## Running it

```bash
ssh -i ~/.ssh/csisnlp_<group> csisnlp_<group>@hpc.bits-hyderabad.ac.in
```

Get the repo onto the cluster, then from the `CS_F429_Part2` directory:

```bash
# 1. one-time, on the LOGIN node (needs internet; compute nodes usually have none)
bash phase_3/biobert/setup_env.sh

# 2. sanity check (~5 min of GPU time)
sbatch phase_3/biobert/job_smoke.sh

# 3. the real run
sbatch phase_3/biobert/job_biobert.sh

# monitor
squeue -u $USER
tail -f phase_3/biobert/logs/biobert_ner_<jobid>.log
```

**Always `sbatch` from the `CS_F429_Part2` directory** — the scripts use
`$SLURM_SUBMIT_DIR` as the project root and write logs to a path relative to it.

### Outputs

- `models/biobert/` — best checkpoint, chosen by dev F1
- `results/biobert_metrics.json` — dev + test scores, same layout as `crf_metrics.json`
- `biobert_test_predictions.json` — same shape as `crf_test_predictions.json`

Then, locally or on the cluster:

```bash
python phase_3/biobert/compare_models.py
```

## Partition constraints

The accounts may only submit to `gpu_rtx_pro_6000_6_csis_hyd` under QOS
`gpu_csis_course`, with `--gres=mps:50` — half a card's compute and VRAM.
Do not change the QOS or swap `mps:50` for `gpu:1`; the scheduler rejects it
at submission. Batch 32 × max_len 256 for BioBERT-base fits that half-card
with room to spare, so there is no need to trade the batch size down.

## Hyperparameters (for the report)

| | |
|---|---|
| Checkpoint | `dmis-lab/biobert-base-cased-v1.1` (BERT-base, 12 layers, 768 hidden, 12 heads, 108M params) |
| Head | linear token classifier over 5 BIO labels |
| Epochs | 5, best selected on dev F1 |
| Learning rate | 3e-5, linear decay, 10% warmup |
| Batch size | 32 (train), 64 (eval) |
| Max length | 256 wordpieces |
| Optimiser | AdamW, weight decay 0.01, none on bias/LayerNorm |
| Grad clipping | 1.0 |
| Precision | bf16 autocast |
| Seed | 42 |

## Two implementation details worth defending in the viva

**Subword alignment.** BERT splits words into wordpieces but the BC5CDR tags
are per word. We label the first subword of each word and mask the
continuations with `-100` so they contribute no loss; at prediction time a
word's tag is read off its first subword. This is the standard HuggingFace
token-classification alignment.

**Truncation.** Words pushed past 256 wordpieces have no subword to read from,
so they are back-filled with `O`. That costs recall rather than silently
returning a short sequence — which would desynchronise the comparison with the
CRF, whose predictions are always full length. At `max_len` 256 this affects
almost nothing (mean sentence length is ~21 tokens), but the guarantee that
every prediction sequence matches its gold sequence length is what makes
`ner_metrics.evaluate` safe to apply to both systems.

## Why this matters for Part 3

The Part 2 error analysis found CRF recall at **0.971** on entity strings seen
20+ times in training but **0.498** on unseen strings, with unseen strings
causing **78.7%** of all errors. BioBERT's wordpiece vocabulary and biomedical
pretraining should close part of that gap without any special handling —
`compare_models.py` quantifies exactly how much. Whatever gap remains is the
target for the rare-entity contribution, and the honest baseline it has to beat.

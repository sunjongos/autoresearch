# autoresearch (CPU lightweight edition)

This is an experiment to have the LLM do its own research — on CPU.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar15`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch_lite/data/` contains data files. If not, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Environment

- **CPU only** (Intel Arc iGPU, no NVIDIA CUDA)
- **Byte-level tokenizer** (vocab_size=257, no BPE)
- **TinyStories dataset** (GPT-4 generated short stories, low entropy)
- **Small model** (~0.3M parameters, depth=4, dim=128)
- **Python + PyTorch** (no uv needed, just `python train.py`)

## Experimentation

Each experiment runs on **CPU**. The training script runs for a **fixed time budget of 2 minutes** (wall clock training time, excluding startup). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies beyond what's in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 2 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size.

**CPU-specific tips:**
- Larger batch sizes may be slower on CPU. Find the sweet spot.
- torch.compile may or may not help on CPU — experiment with it.
- Simpler architectures (fewer layers, smaller dims) may train more steps in the budget.
- More steps at a smaller model may beat fewer steps at a bigger model.
- Try different activations (GELU, ReLU, SiLU, ReLU²).
- Try different LR schedules (cosine, linear, warmup lengths).
- Weight tying (wte = lm_head) can save parameters.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          1.234567
training_seconds: 120.1
total_seconds:    125.9
peak_vram_mb:     0.0
num_steps:        500
num_params_M:     0.3
depth:            4
```

You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB (0.0 for CPU runs)
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar15`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything)
5. Read out the results: `grep "^val_bpb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

**Timeout**: Each experiment should take ~2 minutes. If a run exceeds 5 minutes, kill it and treat it as a failure.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder. The loop runs until the human interrupts you, period.

# Research program for the training loop

You are improving a small GPT trained from scratch on one Mac (Apple M1 Max, 32 GB).
Every run trains for the same fixed wall-clock time and is scored by `val_bpb`
(validation bits per byte, lower is better). Your job this round: make ONE change that
you expect to lower `val_bpb`, then stop. The loop runs the training and keeps your
change only if it beats the best result by more than the run-to-run noise.

## Rules

- Edit only `lab/train/train.py` and `lab/train/model.py`. Any other edit is thrown away.
- Do not run the training yourself. Do not change the data, the tokenizer, the time
  budget, the validation code or the `RESULT` line that `train.py` prints.
- Keep the command line working: `python -m lab.train.train --name X --budget-s N`.
  Change behaviour by changing defaults or code, not by adding required flags.
- One idea per round, small enough to explain in one sentence.
- Peak memory must stay under 12 GB (the Mac swaps above that and the run is wasted).
  The current default (batch 32 x 512 tokens, size "s") peaks near 8 GB.
- Read the results table below first. Do not repeat an idea that was already reverted
  unless you change it meaningfully.

## Ideas worth trying (not a checklist)

- Model shape at fixed time: depth against width, number of heads, MLP width.
- Learning rates, warm-up length, the shape of the decay, weight decay.
- Batch size against sequence length at the same tokens per step.
- Initialisation, normalisation placement, activation function.
- Optimizer details: Muon momentum, which weights Muon handles, AdamW betas.
- 16-bit weights for speed, if the loss stays stable.

## How to finish

End your reply with one line that starts with `CHANGE:` and says in plain words what
you changed and why you expect it to help.

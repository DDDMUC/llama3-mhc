# data/eval/

Evaluation data for `eval.py` (multiple-choice tasks, light subsets).

**Data is NOT in git; only the generator code is.** To regenerate the ARC-Easy
subset exactly as we use it:

```
python data/eval/prepare.py --n 50
```

That reads the first 50 examples of ARC-Easy (test split) from
`allenai/ai2_arc` (HuggingFace `datasets` library, streaming) and writes
`arc_easy_test50.json` (a local, git-ignored file for `eval.py`).

- Requires network access to huggingface.co. On this machine WSL has no
  internet, so run it with Windows Python (`datasets` + `pyarrow` installed).
- We use `--n 50` as a light subset; the model's ARC accuracy is near random
  (it is a pretrained base model) — eval.py demonstrates the tooling, not
  model capability.
- ARC dataset license: CC-BY-SA (used only for evaluation demos).

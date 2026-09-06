# data/eval/

Evaluation data for `eval.py`. **Data is NOT in git; only the generator code
is.** Regenerate with `data/eval/prepare.py`:

```
python data/eval/prepare.py --dataset arc   --n 50
python data/eval/prepare.py --dataset gsm8k --n 20
python data/eval/prepare.py --dataset mmlu  --subject abstract_algebra --n 20
```

- `arc`   -> `allenai/ai2_arc` (ARC-Easy, test) — multiple-choice.
- `gsm8k` -> `openai/gsm8k` (main, test) — math word problems (generative).
- `mmlu`  -> `cais/mmlu` (<subject>, test) — multiple-choice.

Requires network + `datasets` (run with Windows Python; WSL has no internet by
default). Licenses: ARC CC-BY-SA, GSM8K MIT, MMLU CC-BY-NC-4.0 — used only for
eval demos.

**Tooling demo, not a capability claim**: a pretrained base model scores near
random / ~0 here. Observed (200-iter TinyStories base / shakespeare char model,
`eval.py`): ARC-Easy 0.280 (random 0.250), MMLU-abstract_algebra 0.200 (random
0.250), GSM8K 0.000 (char model does not produce math). Real capability needs
a converged + SFT'd model.

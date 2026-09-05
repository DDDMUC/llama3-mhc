# data/eval/

Evaluation data for `eval.py` (multiple-choice tasks, light subsets).

- `arc_easy_test50.json` — first 50 examples of ARC-Easy (test split),
  from `allenai/ai2_arc` (ARC-Easy), accessed via HuggingFace datasets
  `datasets` library (streaming). Each item: `question`, `choices` (list),
  `answerKey`, `id`. License: CC-BY-SA (ARC); used for evaluation demos only.

Regenerate/download with Python + `datasets` (network needed):
```python
from datasets import load_dataset
import json
ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", streaming=True)
items = []
for i, item in enumerate(ds):
    if i >= 50: break
    items.append({"question": item["question"],
                  "choices": list(item["choices"]["text"]),
                  "answerKey": item["answerKey"], "id": item["id"]})
json.dump(items, open("data/eval/arc_easy_test50.json", "w", encoding="utf-8"))
```

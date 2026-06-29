# Downloaded Datasets

This directory contains local datasets for neural-symbolic learning experiments. Data files are not intended to be committed to git; `datasets/.gitignore` excludes them while preserving documentation and small sample files.

## Dataset 1: CLUTRR v1, `gen_train234_test2to10`

### Overview

- Source: raw files referenced by Hugging Face `CLUTRR/v1`, from https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/main/
- Local path: `datasets/clutrr_v1_gen_train234_test2to10/`
- Size: 12,064 train, 3,019 validation, 1,048 test examples; about 5.2 MB
- Format: CSV
- Task: infer a family relation between queried entities from a story and latent proof path
- License: CLUTRR is CC-BY-NC 4.0 according to the official repository
- Related paper: `papers/1908.06177_clutrr_diagnostic_benchmark_inductive_reasoning_text.pdf`

### Download Instructions

```python
from pathlib import Path
import requests

task = "gen_train234_test2to10"
base = f"https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/main/{task}"
out = Path("datasets/clutrr_v1_gen_train234_test2to10")
out.mkdir(parents=True, exist_ok=True)

for split in ["train", "validation", "test"]:
    response = requests.get(f"{base}/{split}.csv", timeout=60)
    response.raise_for_status()
    (out / f"{split}.csv").write_bytes(response.content)
```

### Loading the Dataset

```python
import pandas as pd

train = pd.read_csv("datasets/clutrr_v1_gen_train234_test2to10/train.csv")
validation = pd.read_csv("datasets/clutrr_v1_gen_train234_test2to10/validation.csv")
test = pd.read_csv("datasets/clutrr_v1_gen_train234_test2to10/test.csv")
```

### Sample Data

See `datasets/clutrr_v1_gen_train234_test2to10/samples/examples.json`.

Representative fields:

```json
{
  "story": "[Ashley]'s daughter, [Lillian], asked her mom to read her a story. [Nicholas]'s sister [Lillian] asked him for some help planting her garden.",
  "query": "('Ashley', 'Nicholas')",
  "target_text": "son",
  "f_comb": "daughter-brother",
  "task_name": "task_1.2"
}
```

### Notes

- Hugging Face `datasets` 5.0 no longer runs the legacy `CLUTRR/v1` dataset script, so this workspace downloads the raw CSV files directly.
- The official generator is cloned at `code/clutrr/`.
- This dataset is a good fit for symbolic-rule, proof-path, and systematic-generalization experiments.

## Dataset 2: bAbI QA

### Overview

- Source: Hugging Face `Muennighoff/babi`, https://huggingface.co/datasets/Muennighoff/babi
- Local path: `datasets/babi_muennighoff/`
- Size: 18,013 train, 1,987 validation, 20,000 test examples; about 2.5 MB
- Format: Hugging Face Dataset saved to disk
- Task: toy text question answering covering path finding, induction, deduction, counting, and related reasoning skills
- Related paper: `papers/1502.05698_babi_towards_ai_complete_question_answering.pdf`

### Download Instructions

```python
from datasets import load_dataset

dataset = load_dataset("Muennighoff/babi")
dataset.save_to_disk("datasets/babi_muennighoff")
```

### Loading the Dataset

```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/babi_muennighoff")
```

### Sample Data

See `datasets/babi_muennighoff/samples/examples.json`.

Representative record:

```json
{
  "passage": "Mary moved to the bathroom.\nJohn went to the hallway.\n",
  "question": "Where is Mary?",
  "answer": "bathroom",
  "task": 1
}
```

### Notes

- The original `facebook/babi_qa` Hugging Face loader is a legacy dataset script and is not compatible with `datasets` 5.0.
- The `Muennighoff/babi` mirror provides JSONL data files that load cleanly with the current library.
- Useful for controlled rule induction and symbolic memory baselines without visual perception overhead.

## Dataset 3: MNIST

### Overview

- Source: Hugging Face `ylecun/mnist`, https://huggingface.co/datasets/ylecun/mnist
- Local path: `datasets/mnist/`
- Size: 60,000 train, 10,000 test examples; about 17 MB
- Format: Hugging Face Dataset saved to disk
- Task: digit perception substrate for AddMNIST, multi-digit arithmetic, DeepProbLog, A-NeSI, and Scallop examples
- License: see Hugging Face dataset card

### Download Instructions

```python
from datasets import load_dataset

dataset = load_dataset("ylecun/mnist", "mnist")
dataset.save_to_disk("datasets/mnist")
```

### Loading the Dataset

```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/mnist")
```

### Sample Data

See `datasets/mnist/samples/examples.json`.

Representative record:

```json
{
  "label": 5,
  "image_mode": "L",
  "image_size": [28, 28]
}
```

### Notes

- `pillow` is installed in the local `uv` environment so image decoding works.
- This dataset supports quick weak-supervision tasks such as predicting sums/products from digit images without exposing individual digit labels to the symbolic module.

## Large Visual Datasets Not Downloaded

### CLEVR

- Source: https://cs.stanford.edu/people/jcjohns/clevr/
- Related paper: `papers/1612.06890_clevr_diagnostic_dataset_compositional_visual_reasoning.pdf`
- Reason not downloaded: full CLEVR image data is large and unnecessary for a lightweight resource phase.
- Code requiring it: `code/NSCL-PyTorch-Release/`

### CLEVR-Hans

- Source/instructions: https://github.com/ml-research/CLEVR-Hans
- Related paper: `papers/stammer_2021_right_for_the_right_concept_nesy_xil.pdf`
- Reason not downloaded: visual data is larger than the text/MNIST benchmarks and better handled on demand.
- Code requiring it: `code/NeSyXIL/`

## Validation Summary

- CLUTRR CSVs loaded with pandas and split counts were verified.
- bAbI and MNIST were loaded with `datasets.load_from_disk`.
- Sample files were written under each dataset's `samples/` directory.

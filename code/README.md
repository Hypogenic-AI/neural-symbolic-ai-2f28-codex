# Cloned Repositories

This directory contains code repositories cloned for neural-symbolic learning baselines, dataset generation, and reference implementations. Repositories were cloned with `--depth 1`; commit hashes below record the checked-out snapshots.

## Scallop

- URL: https://github.com/scallop-lang/scallop
- Location: `code/scallop/`
- Commit: `668bfb6`
- Purpose: Datalog-based neurosymbolic programming language with discrete, probabilistic, and differentiable reasoning through provenance semirings.
- Key files:
  - `examples/probabilistic/digit_sum_2.scl`
  - `experiments/clutrr-v2/`
  - `experiments/pacman_maze/`
  - `etc/scallopy/`
- Requirements: Rust nightly for core tools; Python binding `scallopy` can integrate with PyTorch.
- Notes: Strong candidate for new experiments on CLUTRR or AddMNIST-like tasks because symbolic rules are compact and readable.

## Lobster

- URL: https://github.com/P-bibs/Lobster
- Location: `code/Lobster/`
- Commit: `750af53`
- Purpose: GPU-accelerated fork of Scallop for neurosymbolic Datalog workloads.
- Key files:
  - `core/src/gpu_runtime/`
  - `evaluation/`
  - `readme.md`
- Requirements: Docker, CUDA, Rust/C++ build chain; some experiments require very high VRAM.
- Notes: Useful for scalability ideas. Not recommended as the first runnable baseline unless a compatible GPU environment is available.

## DeepProbLog

- URL: https://github.com/ML-KULeuven/deepproblog
- Location: `code/deepproblog/`
- Commit: `879bda3`
- Purpose: Probabilistic logic programming with neural predicates.
- Key files:
  - `src/deepproblog/examples/MNIST/addition.py`
  - `src/deepproblog/examples/CLUTRR/`
  - `src/deepproblog/examples/HWF/`
  - `src/deepproblog/examples/minimal/`
- Requirements: Python > 3.9, ProbLog, PySDD, PyTorch, TorchVision; approximate inference additionally needs PySwip and SWI-Prolog < 9.
- Notes: Primary probabilistic-logic baseline for MNIST addition and CLUTRR-style relational reasoning.

## LTNtorch

- URL: https://github.com/logictensornetworks/LTNtorch
- Location: `code/LTNtorch/`
- Commit: `d1bd981`
- Purpose: PyTorch implementation of Logic Tensor Networks.
- Key files:
  - `ltn/core.py`
  - `ltn/fuzzy_ops.py`
  - `tutorials/`
  - `examples/`
- Requirements: Python/PyTorch stack from `requirements.txt`; can also be installed with `pip install LTNtorch`.
- Notes: Lightweight differentiable-logic baseline. Good first implementation path for semi-supervised or constraint-guided learning.

## A-NeSI

- URL: https://github.com/HEmile/a-nesi
- Location: `code/a-nesi/`
- Commit: `2c55d61`
- Purpose: Approximate neurosymbolic inference for probabilistic neurosymbolic learning.
- Key files:
  - `anesi/experiments/mnist_op/`
  - `anesi/experiments/path_planning/`
  - `anesi/experiments/visudo/`
  - `environment.yml`
- Requirements: Python 3.9, PyTorch 1.12.1, TorchVision 0.13.1, Weights & Biases; README assumes conda.
- Notes: Treat as a reference/paper reproduction repo rather than installing into the shared `uv` environment. MNIST operation experiments are directly relevant.

## Neural Logic Machines

- URL: https://github.com/google/neural-logic-machines
- Location: `code/neural-logic-machines/`
- Commit: `3f8a896`
- Purpose: Tensorized lifted-rule learning for relational reasoning and decision-making tasks.
- Key files:
  - `scripts/graph/`
  - `scripts/blocksworld/`
  - `models/`
- Requirements: Older PyTorch 0.4-era stack plus Jacinle submodule. README assumes conda.
- Notes: Important systematic-generalization baseline, but likely needs dependency modernization before direct comparison.

## NSCL-PyTorch-Release

- URL: https://github.com/vacancy/NSCL-PyTorch-Release
- Location: `code/NSCL-PyTorch-Release/`
- Commit: `ef493d5`
- Purpose: PyTorch implementation of the Neuro-Symbolic Concept Learner.
- Key files:
  - `scripts/trainval.py`
  - `experiments/clevr/`
  - `requirements.txt`
- Requirements: PyTorch 1.0+ with CUDA, Jacinle, CLEVR data and object-detection scene files.
- Notes: Visual reasoning reference implementation. Full CLEVR was not downloaded because of size; dataset preparation links are documented in `datasets/README.md`.

## NeSyXIL

- URL: https://github.com/ml-research/NeSyXIL
- Location: `code/NeSyXIL/`
- Commit: `8a8b2a5`
- Purpose: Official implementation for concept-level explanatory interactive learning on CLEVR-Hans and ColorMNIST.
- Key files:
  - `src/color_mnist/data/generate_color_mnist.py`
  - `src/color_mnist/scripts/`
  - `src/clevr_hans/`
  - `src/docker/`
- Requirements: Docker/GPU for full CLEVR-Hans reproduction; ColorMNIST path is lighter.
- Notes: Useful for experiments about correcting concept-level shortcuts and measuring explanation-guided intervention.

## CLUTRR

- URL: https://github.com/facebookresearch/clutrr
- Location: `code/clutrr/`
- Commit: `d045fae`
- Purpose: Official generator for compositional text-based relational reasoning data.
- Key files:
  - `clutrr/main.py`
  - `clutrr/store/rules_store.yaml`
  - `requirements.txt`
- Requirements: pandas, names, tqdm; install with `python setup.py develop` if using the generator.
- Notes: Local pre-generated CLUTRR CSVs are in `datasets/clutrr_v1_gen_train234_test2to10/`.

## Quick Recommendation

Start with `LTNtorch` or Scallop for a new experiment in this workspace. Use DeepProbLog/A-NeSI as probabilistic inference baselines and NLM/NS-CL/NeSyXIL as reference implementations where dependency modernization or GPU resources are available.

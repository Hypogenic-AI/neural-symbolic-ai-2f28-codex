# Resources Catalog

## Summary

This document catalogs all resources gathered for the neural-symbolic learning research project.

- Papers downloaded: 18 PDFs, about 109 MB
- Paper chunks/manifests: 167 files in `papers/pages/`
- Datasets downloaded: 3 local datasets, about 24 MB
- Code repositories cloned: 9 repositories, about 796 MB
- Python environment: local `.venv` created with `uv`
- Dependency file: `pyproject.toml`

## Papers

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Scallop | Li, Huang, Naik | 2023 | `papers/2304.04812_scallop_language_for_neurosymbolic_programming.pdf` | Differentiable Datalog/provenance framework. |
| DeepProbLog | Manhaeve et al. | 2018 | `papers/1805.10872_deepproblog_neural_probabilistic_logic_programming.pdf` | Neural predicates in probabilistic logic programs. |
| Logic Tensor Networks | LTN authors | 2020 | `papers/2012.13635_logic_tensor_networks.pdf` | Differentiable first-order fuzzy logic. |
| LTNtorch | Carraro et al. | 2024 | `papers/2409.16045_ltntorch_pytorch_logic_tensor_networks.pdf` | Practical PyTorch LTN implementation. |
| Neural Logic Machines | Dong et al. | 2019 | `papers/1904.11694_neural_logic_machines.pdf` | Lifted rule learning and systematic generalization. |
| Neuro-Symbolic Concept Learner | Mao et al. | 2019 | `papers/1904.12584_neuro_symbolic_concept_learner.pdf` | Object-centric CLEVR reasoning with symbolic programs. |
| A-NeSI | van Krieken et al. | 2023 | `papers/2212.12393_a_nesi_approximate_neurosymbolic_inference.pdf` | Approximate scalable probabilistic inference. |
| DeepStochLog | Manhaeve et al. | 2021 | `papers/2106.12574_deepstochlog_neural_stochastic_logic_programming.pdf` | Stochastic derivation-based logic programming. |
| Logical Neural Networks | IBM Research team | 2020 | `papers/2006.13155_logical_neural_networks.pdf` | Formula-structured interpretable neural networks. |
| Lobster | Biberstein et al. | 2025 | `papers/2503.21937_lobster_gpu_accelerated_neurosymbolic_programming.pdf` | GPU acceleration for neurosymbolic Datalog. |
| ANDRE | Sharifi, Wei, Fallah | 2026 | `papers/2605.04193_andre_differentiable_rule_extractor.pdf` | Attention-based differentiable ILP/rule extraction. |
| Right for the Right Concept | Stammer et al. | 2021 | `papers/stammer_2021_right_for_the_right_concept_nesy_xil.pdf` | Concept-level intervention for shortcut correction. |
| Neuro-Symbolic AI in 2024 | Colelough, Regli | 2025 | `papers/2501.05435_neuro_symbolic_ai_2024_systematic_review.pdf` | Systematic review and gaps. |
| Neuro-Symbolic AI Explainability | Zhang, Sheng | 2024 | `papers/2411.04383_neuro_symbolic_ai_explainability_challenges_future_trends.pdf` | Explainability taxonomy and challenges. |
| Neural-Symbolic Reasoning over KGs | Survey authors | 2024 | `papers/2412.10390_neural_symbolic_reasoning_over_knowledge_graphs.pdf` | KG reasoning taxonomy and LLM/KG directions. |
| CLUTRR | Sinha et al. | 2019 | `papers/1908.06177_clutrr_diagnostic_benchmark_inductive_reasoning_text.pdf` | Text relational reasoning benchmark. |
| bAbI | Weston et al. | 2015 | `papers/1502.05698_babi_towards_ai_complete_question_answering.pdf` | Toy QA reasoning tasks. |
| CLEVR | Johnson et al. | 2016 | `papers/1612.06890_clevr_diagnostic_dataset_compositional_visual_reasoning.pdf` | Visual compositional reasoning benchmark. |

See `papers/README.md` for detailed descriptions and source URLs.

## Datasets

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| CLUTRR v1 `gen_train234_test2to10` | Raw GitHub files referenced by `CLUTRR/v1` | 12,064 train, 3,019 validation, 1,048 test; 5.2 MB | Text relational reasoning | `datasets/clutrr_v1_gen_train234_test2to10/` | Loaded with pandas; legacy HF script bypassed. |
| bAbI QA | Hugging Face `Muennighoff/babi` | 18,013 train, 1,987 validation, 20,000 test; 2.5 MB | Controlled text QA | `datasets/babi_muennighoff/` | Saved with `datasets.save_to_disk`. |
| MNIST | Hugging Face `ylecun/mnist` | 60,000 train, 10,000 test; 17 MB | Digit perception and AddMNIST-style reasoning | `datasets/mnist/` | Saved with `datasets.save_to_disk`; Pillow installed. |

See `datasets/README.md` for download and loading instructions.

## Code Repositories

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| Scallop | https://github.com/scallop-lang/scallop | Differentiable/probabilistic Datalog | `code/scallop/` | Rust nightly plus Python binding. |
| Lobster | https://github.com/P-bibs/Lobster | GPU-accelerated Scallop-style reasoning | `code/Lobster/` | Docker/CUDA heavy; scalability reference. |
| DeepProbLog | https://github.com/ML-KULeuven/deepproblog | Neural probabilistic logic programming | `code/deepproblog/` | Has MNIST, CLUTRR, HWF examples. |
| LTNtorch | https://github.com/logictensornetworks/LTNtorch | PyTorch Logic Tensor Networks | `code/LTNtorch/` | Lightweight differentiable-logic baseline. |
| A-NeSI | https://github.com/HEmile/a-nesi | Approximate probabilistic neurosymbolic inference | `code/a-nesi/` | Conda/W&B oriented reproduction repo. |
| Neural Logic Machines | https://github.com/google/neural-logic-machines | Lifted relational rule learning | `code/neural-logic-machines/` | Older PyTorch/Jacinle stack. |
| NSCL-PyTorch-Release | https://github.com/vacancy/NSCL-PyTorch-Release | Neuro-Symbolic Concept Learner | `code/NSCL-PyTorch-Release/` | Requires CLEVR preprocessing. |
| NeSyXIL | https://github.com/ml-research/NeSyXIL | Concept-level explanation/intervention | `code/NeSyXIL/` | Docker/GPU for full CLEVR-Hans; ColorMNIST path lighter. |
| CLUTRR | https://github.com/facebookresearch/clutrr | Dataset generator | `code/clutrr/` | Useful for custom relation-length splits. |

See `code/README.md` for detailed descriptions, commits, dependencies, and key files.

## Resource Gathering Notes

### Search Strategy

The search started with the paper-finder helper, as requested by the workflow. Both a diligent and a fast query timed out, so the search proceeded manually across arXiv, web academic search, Hugging Face, Papers with Code/GitHub, and official project pages. Core queries included "neural-symbolic learning", "differentiable logic", "neural probabilistic logic programming", "Logic Tensor Networks", "Scallop neurosymbolic programming", and "CLUTRR bAbI MNIST".

### Selection Criteria

Resources were selected for one of four reasons:

- They are foundational baselines: DeepProbLog, LTN, NLM, NS-CL.
- They are practical implementation platforms: Scallop, LTNtorch, DeepProbLog.
- They address known bottlenecks: A-NeSI for approximate inference, Lobster for GPU reasoning, ANDRE for differentiable rule induction.
- They provide small reproducible benchmarks: CLUTRR, bAbI, MNIST.

### Challenges Encountered

- The paper-finder CLI timed out and did not return ranked results.
- Hugging Face `datasets` 5.0 no longer supports legacy dataset scripts, affecting `CLUTRR/v1` and `facebook/babi_qa`.
- CLEVR and CLEVR-Hans were not downloaded due to size; they are documented for optional future use.
- Several major codebases assume old dependency stacks, conda, CUDA, Docker, or W&B.

### Gaps and Workarounds

- ANDRE appears to have no public code linked from the arXiv page or search results; it is included as a paper-only recent method.
- Full CLEVR visual experiments require separate data and preprocessing; use MNIST/CLUTRR/bAbI for first-pass automated experiments.
- For CLUTRR, raw CSVs were downloaded directly from the upstream file source used by the legacy Hugging Face loader.

## Recommendations for Experiment Design

1. Primary datasets: start with `datasets/clutrr_v1_gen_train234_test2to10/` and `datasets/mnist/`.
2. Baseline methods: use LTNtorch for differentiable constraints, Scallop for declarative rules, and DeepProbLog for probabilistic neural predicates.
3. Evaluation metrics: accuracy, out-of-distribution relation-length accuracy, formula/constraint satisfaction, runtime, and data-efficiency curves.
4. Code to adapt: `code/scallop/experiments/clutrr-v2/`, `code/deepproblog/src/deepproblog/examples/MNIST/`, `code/deepproblog/src/deepproblog/examples/CLUTRR/`, and `code/LTNtorch/examples/`.
5. Stretch comparison: port or run A-NeSI MNIST operation experiments if the runner can tolerate its conda/W&B assumptions.

## Research Execution Outputs

The automated research execution used the downloaded MNIST dataset for an AddMNIST compositional generalization experiment. Training excluded high-plus-high digit pairs, while OOD evaluation used only high-plus-high pairs.

Generated files:

- `planning.md`: motivation, novelty, and preregistered experimental plan.
- `src/addmnist_experiment.py`: runnable training/evaluation script.
- `src/analyze_results.py`: summary statistics, paired tests, bootstrap CIs, and plotting.
- `results/addmnist/metrics.csv`: aggregate metrics for 4 methods x 2 budgets x 3 seeds.
- `results/addmnist/predictions.csv`: 192,000 per-example predictions.
- `results/addmnist/statistical_tests.csv`: paired seed-level OOD comparisons.
- `results/addmnist/ood_sum_region_summary.csv`: seen-sum versus unseen-sum OOD breakdown.
- `figures/accuracy_by_method_budget.png`, `figures/ood_accuracy_by_sum.png`, `figures/digit_grounding.png`: report figures.
- `REPORT.md`: final research report with actual findings.

Key result: at the 5,000 pair-label budget, the plain neural-symbolic `symbolic_sum` model reached 82.5% OOD high-plus-high accuracy and 80.6% accuracy on unseen training labels 14-18, while the direct CNN baseline reached 0.3% and 0.0%, respectively.

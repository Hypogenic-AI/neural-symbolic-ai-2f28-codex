# Neural-Symbolic AddMNIST Experiment

This workspace contains a completed neural-symbolic learning experiment on AddMNIST. The study tests whether a digit-perception CNN plus symbolic addition can generalize to high-plus-high sums that never appear as training labels.

## Key Findings

- With 5,000 weak pair labels, `symbolic_sum` reached 82.5% OOD high-plus-high accuracy; the direct CNN reached 0.3%.
- On truly unseen training labels 14-18, `symbolic_sum` reached 80.6% accuracy while the direct CNN reached 0.0%.
- The planned anchor-posterior variant did not beat the simpler symbolic likelihood baseline.
- At 1,000 pair labels, symbolic structure alone was insufficient; digit grounding remained too weak.
- Full details are in [REPORT.md](REPORT.md).

## Reproduce

```bash
source .venv/bin/activate
python src/addmnist_experiment.py --output-dir results/addmnist --budgets 1000 5000 --seeds 11 23 37 --epochs 15 --batch-size 512 --eval-pairs 4000
python src/analyze_results.py --results-dir results/addmnist --figures-dir figures
```

Dependencies are managed by `uv` in `pyproject.toml`. The experiment used PyTorch 2.12.1 with CUDA on an NVIDIA RTX A6000.

## File Structure

- `planning.md`: preregistered motivation, novelty, and experimental plan.
- `src/addmnist_experiment.py`: dataset construction, models, training, and evaluation.
- `src/analyze_results.py`: aggregate metrics, statistical tests, and figures.
- `results/addmnist/`: metrics, predictions, checkpoints, metadata, and analysis tables.
- `figures/`: plots used in the report.
- `REPORT.md`: complete research report with methodology, results, limitations, and next steps.
- `literature_review.md` and `resources.md`: pre-gathered research context and resource catalog.

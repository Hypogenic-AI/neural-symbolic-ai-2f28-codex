# Neural-Symbolic Learning Research Plan

## Motivation & Novelty Assessment

### Why This Research Matters
Neural-symbolic learning promises models that learn perception from data while preserving compositional reasoning rules that generalize beyond the training distribution. This matters for AI systems that must extrapolate reliably from sparse supervision, especially where purely neural classifiers memorize observed labels and fail on unseen combinations.

### Gap in Existing Work
The gathered literature shows strong frameworks for probabilistic neural predicates (DeepProbLog, A-NeSI), differentiable logic (LTNtorch), and declarative Datalog-style reasoning (Scallop), but practical first-pass experiments often either rely on exact inference stacks with dependency drift or test only in-distribution accuracy. A useful gap is a compact, reproducible experiment that separates perceptual learning from symbolic extrapolation and explicitly tests unseen output labels created only by composition.

### Our Novel Contribution
This session tests an anchor-posterior AddMNIST learner: a shared digit CNN is trained mostly from pair-level sum labels through a symbolic marginalization layer, with a tiny set of digit anchors and a posterior entropy regularizer to ground and sharpen latent digit assignments. The "cool" stress test is high-plus-high addition: training excludes pairs where both digits are 5-9, so sums 14-18 are never directly observed as labels, but a symbolic model can still generate them by composing learned digit concepts.

### Experiment Justification
- Experiment 1: In-distribution low/high AddMNIST. This checks whether each method can learn the supervised training distribution without the OOD stressor.
- Experiment 2: High-plus-high compositional OOD test. This directly tests whether symbolic composition predicts unseen sums that a direct neural label classifier cannot learn from examples.
- Experiment 3: Data-efficiency and ablation across training sizes and seeds. This tests whether the proposed anchor-posterior components improve reliability rather than succeeding due to one lucky split.
- Experiment 4: Digit grounding/error analysis. This checks whether the symbolic model learns interpretable digit predicates, not just correct aggregate sums.

## Research Question
Can a compact neural-symbolic AddMNIST learner with tiny digit anchors and posterior sharpening achieve better systematic generalization than direct neural sum prediction when some output sums are never observed during training?

## Background and Motivation
DeepProbLog and A-NeSI demonstrate that neural predicates embedded in probabilistic logic can solve MNIST arithmetic from weak labels, while Scallop and LTNtorch show alternative differentiable or declarative reasoning routes. The core unresolved issue for a quick automated study is not whether logic can encode addition, but whether symbolic structure improves data efficiency and out-of-distribution generalization under a fair, runnable protocol.

## Hypothesis Decomposition
- H1: A direct pair-image CNN trained only on observed sum labels will perform well in-distribution but fail on held-out sums 14-18 because those labels never appear in training.
- H2: A neural-symbolic sum-likelihood model can predict unseen sums by composing digit probabilities, but may be unstable without grounding.
- H3: Adding a small balanced digit-anchor set and posterior entropy regularization improves both digit grounding and OOD sum accuracy.
- H4: Improvements should persist across random seeds and at multiple pair-label budgets.

Independent variables are model condition, pair-label budget, and random seed. Dependent variables are IID sum accuracy, OOD high-plus-high sum accuracy, digit accuracy for symbolic models, negative log-likelihood, runtime, and calibration-style confidence summaries.

## Proposed Methodology

### Approach
Use the downloaded MNIST dataset to construct AddMNIST pairs. Training pairs exclude the high-plus-high region where both digits are 5-9, while IID validation/test pairs follow the same restriction and OOD test pairs contain only high-plus-high examples. This creates a clean compositional generalization test where labels 14-18 are absent from training but mathematically valid.

### Experimental Steps
1. Load MNIST from `datasets/mnist/`, convert images to normalized tensors, and validate label counts.
2. Generate deterministic pair datasets for each seed and pair-label budget, excluding high-plus-high pairs for training and IID evaluation.
3. Train a direct pair CNN baseline that predicts the sum label from two images.
4. Train a neural-symbolic sum-likelihood model with a shared digit CNN and exact symbolic marginalization over all digit pairs consistent with each sum.
5. Train the proposed anchor-posterior model by adding 10 digit anchors per class and entropy regularization over the posterior distribution of latent digit pairs given the observed sum.
6. Evaluate all conditions on IID and OOD pairs; evaluate digit grounding for symbolic models.
7. Run paired statistical comparisons across seeds and bootstrap confidence intervals over examples for the main OOD metric.

### Baselines
- `direct_sum`: a pure neural two-image CNN trained with cross-entropy over observed sum labels.
- `symbolic_sum`: a neural-symbolic weak-label baseline trained only with sum-likelihood through a fixed addition table.
- `anchor_symbolic`: symbolic sum-likelihood plus a tiny balanced digit-anchor loss.
- `anchor_posterior`: the proposed method, adding posterior entropy sharpening to `anchor_symbolic`.

### Evaluation Metrics
- Sum accuracy on IID restricted pairs: verifies in-distribution learning.
- Sum accuracy on OOD high-plus-high pairs: primary systematic generalization metric.
- Macro F1 on sum labels: reduces sensitivity to label imbalance.
- Digit accuracy for symbolic models: measures whether learned predicates align with human digit classes.
- Negative log-likelihood and mean confidence: sanity-check calibration and overconfidence.
- Training time: documents compute cost.

### Statistical Analysis Plan
For each pair budget, compare OOD accuracy by paired seed-level differences. Use paired t-tests as descriptive statistics, report effect sizes (Cohen's dz), and add bootstrap 95% confidence intervals over example-level OOD accuracy for the final model. With only three seeds, p-values will be treated as exploratory and effect size/consistency will drive interpretation.

## Expected Outcomes
Results support the hypothesis if `anchor_posterior` or `anchor_symbolic` substantially outperforms `direct_sum` on high-plus-high OOD accuracy while preserving reasonable IID accuracy. Results refute or weaken the hypothesis if the direct baseline matches symbolic OOD performance or if symbolic methods fail to learn grounded digit predicates.

## Timeline and Milestones
- Resource and environment verification: 10-20 minutes.
- Implementation: 60-90 minutes for data construction, models, training loop, and evaluation.
- Experiment execution: 60-90 minutes for multiple seeds and budgets on GPU.
- Analysis and documentation: 45-75 minutes for tables, figures, statistical tests, and final report.

## Potential Challenges
- Weak sum labels can produce latent-label symmetries; the balanced anchor set is designed to break them.
- Direct baselines may be structurally unable to predict unseen classes; this is a valid limitation and the key compositional stress test, so IID results are reported alongside OOD.
- Training many runs can be slow; if needed, reduce epochs before reducing seeds.
- GPU package compatibility may fail; if CUDA is unavailable, the same scripts run on CPU with smaller budgets.

## Success Criteria
The research succeeds if it produces a runnable experiment, saved raw results, statistical summaries, figures, and a report with honest conclusions about when the neural-symbolic method improves over baselines. A strong positive result is a large OOD gain for symbolic methods on unseen sums 14-18 with interpretable digit accuracy above chance.

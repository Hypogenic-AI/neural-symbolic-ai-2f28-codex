# Outline: Exact Symbolic Marginalization Enables Unseen-Sum Generalization in AddMNIST

## Title
- Exact Symbolic Marginalization Enables Unseen-Sum Generalization in AddMNIST

## Abstract
- Problem: weakly supervised neural-symbolic systems should compose learned predicates beyond observed labels.
- Approach: AddMNIST with high-plus-high training pairs removed; compare direct pair CNN, symbolic sum likelihood, anchors, and posterior entropy.
- Evidence: 5,000 pair labels, symbolic_sum reaches 90.0% IID and 82.5% OOD high-plus-high; direct_sum reaches 31.8% IID and 0.3% OOD.
- Significance: exact symbolic marginalization can extrapolate to sums 14-18 that never appear as training labels, once digit grounding is learned.

## Introduction
- Hook: compositional generalization is the promise of neural-symbolic learning.
- Gap: many demonstrations mix perception and reasoning, but fewer isolate unseen output-label extrapolation in a compact reproducible setup.
- Approach: train on AddMNIST restricted pairs and evaluate on high-plus-high pairs.
- Preview: symbolic_sum gives +82.2 percentage points OOD over direct_sum at 5,000 pair labels; anchor-posterior ablation does not improve over the simple symbolic likelihood.
- Contributions: benchmark protocol, exact marginalization implementation, ablation, statistical/digit-grounding analysis.

## Related Work
- Probabilistic neural-symbolic inference: DeepProbLog and A-NeSI.
- Declarative differentiable reasoning: Scallop.
- Differentiable logic and relational generalization: Logic Tensor Networks, LTNtorch, Neural Logic Machines.
- Concept-level visual reasoning: Neuro-Symbolic Concept Learner and concept intervention work.
- Position: this paper focuses on a small, exact, reproducible unseen-label AddMNIST test.

## Methodology
- Problem formulation: two images, latent digits, sum label.
- Data: local MNIST; restricted training excludes digit pairs where both digits are 5-9; OOD is only high-plus-high.
- Models: direct_sum, symbolic_sum, anchor_symbolic, anchor_posterior.
- Objective: exact marginalization over digit pairs consistent with the observed sum; optional anchor CE and posterior entropy.
- Protocol: budgets 1,000 and 5,000; seeds 11, 23, 37; 15 epochs; batch 512; AdamW; metrics and statistical tests.

## Results
- Main table: IID/OOD accuracy and CIs by method and budget.
- Figure: accuracy by method and budget.
- Unseen label table: seen sums 10-13 vs unseen sums 14-18 at 5,000.
- Figure: OOD accuracy by sum.
- Digit grounding table/figure.
- Paired tests: symbolic_sum vs direct_sum +82.2 points, Cohen's dz 20.75, p=0.0008.

## Discussion
- Interpretation: symbolic composition helps only after learned predicates are grounded.
- Negative result: anchor-posterior is not best; anchor/entropy may over-constrain early assignments.
- Limitations: synthetic benchmark, direct baseline structurally unable to learn absent classes, three seeds, fixed symbolic rule, no visual domain shift.
- Implications: exact symbolic marginalization is a strong baseline for compositional label extrapolation.

## Conclusion
- Summarize the controlled AddMNIST experiment and key finding.
- Future work: tune anchor/posterior schedules, scale to three-digit arithmetic, compare with framework baselines, add perceptual shifts.

# Literature Review: Neural-Symbolic Learning

## Review Scope

### Research Question

What neural-symbolic learning approaches are most suitable for achieving improved data efficiency, systematic generalization, interpretability, or scalable reasoning, and what resources support rapid experimentation?

### Inclusion Criteria

- Papers combining neural learning with symbolic logic, programs, rules, knowledge graphs, or structured reasoning.
- Work with downloadable code, datasets, or clear benchmark protocols.
- Recent surveys plus foundational method papers needed for baselines.
- Benchmarks suitable for automated experiments in this workspace.

### Exclusion Criteria

- Purely neural reasoning papers without symbolic structure or constraints.
- Pure symbolic systems without learning.
- Large visual datasets that are useful but too large for the resource phase, unless documented as optional.

### Time Frame

Foundational work from 2015-2022 plus recent work through June 29, 2026.

### Sources

- Local paper-finder was attempted in diligent and fast modes but timed out.
- Manual search used arXiv, Semantic Scholar-style web search, Papers with Code/GitHub search, Hugging Face dataset search, and official project repositories.

## Search Log

| Date | Query | Source | Results | Notes |
|------|-------|--------|---------|-------|
| 2026-06-29 | "neural-symbolic learning neuro-symbolic reasoning differentiable logic" | paper-finder | timed out | Fell back to manual/API search. |
| 2026-06-29 | "neural-symbolic learning survey 2024 arXiv" | Web/arXiv | multiple | Found recent surveys and KG reasoning survey. |
| 2026-06-29 | "DeepProbLog Logic Tensor Networks Neural Logic Machines" | Web/arXiv | multiple | Found core differentiable/probabilistic logic baselines. |
| 2026-06-29 | "Scallop neurosymbolic programming GitHub" | Web/GitHub | multiple | Found Scallop and Lobster implementations. |
| 2026-06-29 | "CLUTRR bAbI MNIST Hugging Face" | Web/HF | multiple | Selected small reasoning/perception datasets. |

## Key Papers

### Scallop: A Language for Neurosymbolic Programming

- Authors: Ziyang Li, Jiani Huang, Mayur Naik
- Year/source: 2023, arXiv/PLDI
- Key contribution: A Datalog-like language with discrete, probabilistic, and differentiable reasoning via provenance semirings.
- Methodology: Symbolic reasoning programs are written declaratively; neural modules provide uncertain facts; differentiable reasoning supports end-to-end training.
- Datasets/tasks: Suite of eight tasks spanning MNIST arithmetic, handwritten formula evaluation, CLUTRR-like text reasoning, pathfinding, and other structured tasks.
- Baselines: Pure neural models and task-specific neurosymbolic systems.
- Results: Comparable or better accuracy with improvements in runtime, data efficiency, interpretability, or generalization depending on task.
- Code: `code/scallop/`.
- Relevance: Best general-purpose framework for new experiments in this workspace.

### DeepProbLog: Neural Probabilistic Logic Programming

- Authors: Robin Manhaeve, Sebastijan Dumancic, Angelika Kimmig, Thomas Demeester, Luc De Raedt
- Year/source: 2018, NeurIPS/arXiv
- Key contribution: Adds neural predicates to ProbLog so neural outputs can appear as probabilistic facts in logic programs.
- Methodology: Uses knowledge compilation and automatic differentiation through probabilistic logic inference.
- Datasets/tasks: MNIST addition, program induction, probabilistic reasoning examples.
- Baselines: CNN-only and differentiable-programming approaches.
- Results: Shows that explicit background knowledge improves sample efficiency and structured reasoning relative to black-box neural baselines.
- Code: `code/deepproblog/`.
- Relevance: Foundational probabilistic neurosymbolic baseline.

### Logic Tensor Networks and LTNtorch

- Authors: LTN authors; LTNtorch by Tommaso Carraro and collaborators
- Year/source: 2020 and 2024, arXiv
- Key contribution: Real Logic grounds first-order logic into differentiable tensor operations; LTNtorch makes this practical in PyTorch.
- Methodology: Logical axioms become satisfaction objectives; predicates/functions/constants are learnable modules.
- Datasets/tasks: Classification, multi-label learning, semi-supervised learning, regression, clustering, and embedding examples.
- Baselines: Standard supervised neural learning, fuzzy-logic variants, and other constraint-learning approaches.
- Results: Useful where domain constraints are reliable; performance depends on fuzzy operator choice and balancing rule satisfaction with predictive accuracy.
- Code: `code/LTNtorch/`.
- Relevance: Lightweight baseline for testing differentiable logical constraints.

### Neural Logic Machines

- Authors: Honghua Dong, Jiayuan Mao, Tian Lin, Chong Wang, Lihong Li, Denny Zhou
- Year/source: 2019, ICLR/arXiv
- Key contribution: Tensorized architecture for learning lifted logic rules over objects and relations.
- Methodology: Predicate tensors are transformed by neural logic layers approximating logical operations and quantifiers.
- Datasets/tasks: Family tree reasoning, graph reasoning, sorting, shortest path, and blocks world.
- Baselines: Memory Networks, differentiable ILP, DNC, and task-specific neural baselines.
- Results: Reports perfect or near-perfect generalization on several small-to-large systematic generalization tasks.
- Code: `code/neural-logic-machines/`.
- Relevance: Important benchmark for rule-learning and generalization, but dependency stack is old.

### The Neuro-Symbolic Concept Learner

- Authors: Jiayuan Mao, Chuang Gan, Pushmeet Kohli, Joshua B. Tenenbaum, Jiajun Wu
- Year/source: 2019, ICLR/arXiv
- Key contribution: Learns visual concepts and language concepts from image-question-answer pairs, then executes symbolic programs over object-centric representations.
- Methodology: Neural perception module, semantic parser, and symbolic program executor are trained through natural supervision.
- Datasets/tasks: CLEVR and CLEVR-CoGenT; paper emphasizes learning from fewer images and generalizing to longer programs/new concepts.
- Baselines: Visual reasoning models such as Relation Networks, FiLM-style models, IEP/NMN-style systems, and related VQA models.
- Results: Strong CLEVR performance and interpretability through explicit programs and object concepts.
- Code: `code/NSCL-PyTorch-Release/`.
- Relevance: Reference model for object-centric visual reasoning, but full CLEVR is large.

### A-NeSI

- Authors: Emile van Krieken, Thiviyan Thanapalasingam, Jakub M. Tomczak, Frank van Harmelen, Annette ten Teije
- Year/source: 2022/2023, NeurIPS/arXiv
- Key contribution: Approximate inference for probabilistic neurosymbolic learning without relaxing discrete logical semantics.
- Methodology: Trains prediction and explanation networks using synthetic data generated from background knowledge; uses approximate inference to avoid exponential exact inference.
- Datasets/tasks: Multi-digit MNISTAdd, path planning, visual sudoku.
- Baselines: Exact probabilistic inference methods such as DeepProbLog plus neural approximations.
- Results: Keeps logical guarantees at test time while scaling to larger combinatorial tasks than exact inference.
- Code: `code/a-nesi/`.
- Relevance: Strong candidate baseline for scalable probabilistic reasoning.

### DeepStochLog

- Authors: Robin Manhaeve, Giuseppe Marra, Angelika Kimmig, Luc De Raedt
- Year/source: 2021, arXiv
- Key contribution: Neural stochastic logic programming based on stochastic definite clause grammars.
- Methodology: Models distributions over derivations rather than possible worlds, improving inference scalability for grammar-like tasks.
- Datasets/tasks: Challenging neural-symbolic learning tasks, including structured sequence and arithmetic-style settings.
- Baselines: DeepProbLog and related probabilistic logic approaches.
- Results: Improves scalability and reports strong results on tasks where possible-world inference is expensive.
- Code: no separate repo cloned; conceptually relevant to probabilistic baselines.
- Relevance: Useful design alternative if DeepProbLog inference is the bottleneck.

### Logical Neural Networks

- Authors: IBM Research team
- Year/source: 2020, arXiv
- Key contribution: Neural units correspond to weighted real-valued logic formulas, enabling bidirectional inference and contradiction-aware learning.
- Methodology: Logic formulas define network structure and learning minimizes contradiction.
- Datasets/tasks: Knowledge reasoning and formula-grounded inference tasks.
- Baselines: Standard neural models and symbolic logic systems.
- Results: Emphasizes interpretability, incomplete knowledge, and open-world reasoning.
- Code: not cloned as primary baseline.
- Relevance: Useful conceptual baseline for interpretable formula-level reasoning.

### Lobster

- Authors: Paul Biberstein, Ziyang Li, Joseph Devietti, Mayur Naik
- Year/source: 2025, arXiv/ASPLOS
- Key contribution: GPU execution for Scallop-style neurosymbolic Datalog programs.
- Methodology: Compiles Datalog reasoning to a GPU-oriented intermediate representation and supports several provenance modes.
- Datasets/tasks: Eight applications across NLP, image processing, program reasoning, bioinformatics, and planning.
- Baselines: Scallop, CPU Datalog engines, and task-specific neurosymbolic systems.
- Results: Reports multi-fold speedups over Scallop on supported workloads.
- Code: `code/Lobster/`.
- Relevance: Scalability direction, but hardware/build requirements are high.

### ANDRE

- Authors: Iman Sharifi, Peng Wei, Saber Fallah
- Year/source: 2026, arXiv
- Key contribution: Attention-based differentiable rule extractor for ILP under noise and probabilistic predicates.
- Methodology: Learns first-order rules by optimizing a continuous rule space with attention-driven conjunction/disjunction operators.
- Datasets/tasks: Classical ILP benchmarks, large-scale knowledge bases, and synthetic noisy probabilistic datasets.
- Baselines: Symbolic ILP and differentiable ILP systems.
- Results: Claims competitive or superior predictive performance and better rule recovery under uncertainty.
- Code: no implementation located during search.
- Relevance: Recent idea for novel rule induction, but implementation availability is a gap.

### Right for the Right Concept

- Authors: Wolfgang Stammer, Patrick Schramowski, Kristian Kersting
- Year/source: 2021, CVPR
- Key contribution: Concept-level explanatory interactive learning for neuro-symbolic concept learners.
- Methodology: Uses semantic/concept explanations to revise model behavior, e.g. avoiding shortcut concepts.
- Datasets/tasks: CLEVR-Hans and ColorMNIST.
- Baselines: CNN explanation-guided learning and neuro-symbolic concept learner variants.
- Results: Demonstrates value of semantic explanations for correcting confounded visual reasoning.
- Code: `code/NeSyXIL/`.
- Relevance: Good experimental direction for shortcut learning and intervention quality.

### Surveys and Benchmark Papers

- `Neuro-Symbolic AI in 2024` identifies heavy concentration in learning/inference, logic/reasoning, and knowledge representation, with gaps in explainability, trustworthiness, and meta-cognition.
- `Neuro-Symbolic AI: Explainability, Challenges, and Future Trends` argues that symbolic components do not automatically make a system explainable; explanation should be evaluated at design and behavior levels.
- `Neural-Symbolic Reasoning over Knowledge Graphs` surveys KG reasoning taxonomies and LLM/KG integration opportunities.
- CLUTRR, bAbI, and CLEVR papers define the downloaded or documented benchmarks.

## Common Methodologies

- Neural predicates in probabilistic logic: DeepProbLog and A-NeSI connect neural perception with discrete symbolic reasoning.
- Differentiable fuzzy logic: LTN encodes rules as loss terms through fuzzy truth values.
- Differentiable Datalog/provenance semirings: Scallop and Lobster execute declarative rules with differentiable or probabilistic semantics.
- Object-centric program execution: NS-CL separates perception, parsing, and symbolic execution for visual reasoning.
- Tensorized lifted rules: NLM learns reusable relational rules that generalize across entity counts.
- Interactive concept correction: NeSyXIL intervenes on semantic concepts rather than raw pixels.

## Standard Baselines

- Pure neural: CNN/MLP for MNIST/AddMNIST, RNN/Transformer/Memory Network for bAbI and CLUTRR, CNN/ResNet for ColorMNIST or CLEVR-Hans.
- Probabilistic neurosymbolic: DeepProbLog, A-NeSI, DeepStochLog.
- Differentiable logic: LTN/LTNtorch, differentiable ILP variants, logical neural networks.
- Declarative neurosymbolic: Scallop and Lobster.
- Relational generalization: Neural Logic Machines, differentiable ILP, graph neural networks.

## Evaluation Metrics

- Accuracy: primary metric for bAbI QA, CLUTRR relation classification, MNIST arithmetic, CLEVR QA, and ColorMNIST.
- Systematic generalization: accuracy by relation/path length, number of entities, number of digits, or program length.
- Logical satisfaction: average formula satisfaction or constraint-violation rate for LTN-style systems.
- Explanation quality: concept-rule correctness, rule recovery, explanation faithfulness, and intervention success.
- Runtime/scalability: training time, inference time, timeout rate, speedup versus exact inference or CPU symbolic execution.
- Data efficiency: accuracy as labeled examples are reduced.

## Datasets in the Literature

- MNIST/AddMNIST: used by DeepProbLog, A-NeSI, Scallop, and many neurosymbolic examples for weakly supervised perception plus arithmetic.
- CLUTRR: text-based family-relation reasoning; suitable for rules, proof paths, and out-of-distribution relation length generalization.
- bAbI: controlled text QA tasks covering prerequisite reasoning skills.
- CLEVR/CLEVR-CoGenT: visual question answering and compositional visual reasoning; important but large.
- CLEVR-Hans/ColorMNIST: confounded visual reasoning and shortcut correction.
- Family tree, graph, sorting, shortest path, blocks world: NLM-style relational and planning tasks.

## Gaps and Opportunities

- Scalability remains unresolved: exact probabilistic logic inference is often exponential; GPU Datalog and approximate inference are promising but not yet broadly easy to use.
- Explainability is not guaranteed by adding symbols; it needs behavioral validation and intervention tests.
- Many benchmarks are synthetic, so robustness to noisy perception and ambiguous language should be tested deliberately.
- Dependency drift is significant: several important repos assume old PyTorch/conda/Jacinle stacks.
- Recent differentiable rule extraction work such as ANDRE is promising but lacks readily available code.

## Recommendations for Our Experiment

- Primary small datasets: CLUTRR for text relational reasoning, bAbI for controlled QA, and MNIST for AddMNIST-style perception plus logic.
- First baselines: LTNtorch for constraint-learning, Scallop for declarative differentiable rules, and DeepProbLog for probabilistic neural predicates.
- Stretch baselines: A-NeSI for scalable approximate inference; NLM for systematic generalization if dependency modernization is feasible.
- Avoid first-pass dependency risk: do not start with full CLEVR/NS-CL unless GPU and CLEVR preprocessing are available.
- Recommended metrics: accuracy, OOD/path-length accuracy, constraint satisfaction, data-efficiency curves, and runtime.
- Promising novel direction: compare a declarative-rule model against an approximate probabilistic model on CLUTRR and AddMNIST, then add learned rule selection or concept-intervention diagnostics to test whether improvements come from real symbolic structure rather than shortcut learning.

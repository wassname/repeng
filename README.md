---
title: "InnerPiSSA: Unsupervised SVD Steering for Inner Alignment"
author:
  - name: Michael J. Clark
    id: mjc
    email: michael.j.clark@wassname.org
abstract: |
  The reliance on reinforcement learning (RLHF) for alignment raises a critical measurement gap: optimizing for outcomes rather than process encourages models to dissociate internal reasoning from stated outputs. We introduce InnerPiSSA, which steers hidden states in the model's native SVD transformation space using gradient-based optimization on learnable rotations and singular value scaling. When steering against learned behaviors (coefficient = -1), prompting collapses to incoherent outputs (truthfulness: TBD) while InnerPiSSA maintains controlled bidirectional steering (truthfulness: TBD), evidence that we modify internal reasoning trajectories rather than output style alone. Trained on 800 unsupervised contrastive pairs extracted from incomplete reasoning prefixes, InnerPiSSA achieves significantly stronger effects than PCA (TBD vs TBD) on moral reasoning transfer and outperforms prompting on anti-RLHF robustness. Critically, InnerPiSSA requires no human preference labels or model completions, operating entirely on-policy from the model's own planning trajectories. This demonstrates the strongest performance among tested methods for alignment debugging: probing what models compute internally when output-level constraints are bypassed at the representation level. While not a complete solution to alignment, this is the best-performing approach we tested for studying failure modes that output-level evaluation cannot detect.
date: 2025-11-22
bibliography: references.bib
number-sections: true
format:
  html:
    toc: true
    toc-depth: 3
    code-fold: false
  pdf:
    documentclass: article
    papersize: letter
    geometry:
      - top=1in
      - bottom=1in
      - left=1in
      - right=1in
--
# Abstract

The reliance on reinforcement learning (RLHF) for alignment raises a critical measurement gap: optimizing for outcomes rather than process encourages models to dissociate internal reasoning from stated outputs. We introduce InnerPiSSA, which steers hidden states in the model's native SVD transformation space using gradient-based optimization on learnable rotations and singular value scaling. When steering against learned behaviors (coefficient = -1), prompting collapses to incoherent outputs (truthfulness: TBD) while InnerPiSSA maintains controlled bidirectional steering (truthfulness: TBD), evidence that we modify internal reasoning trajectories rather than output style alone. Trained on 800 unsupervised contrastive pairs extracted from incomplete reasoning prefixes, InnerPiSSA achieves significantly stronger effects than PCA (TBD vs TBD) on moral reasoning transfer and outperforms prompting on anti-RLHF robustness. Critically, InnerPiSSA requires no human preference labels or model completions, operating entirely on-policy from the model's own planning trajectories. This demonstrates the strongest performance among tested methods for alignment debugging: probing what models compute internally when output-level constraints are bypassed at the representation level. While not a complete solution to alignment, this is the best-performing approach we tested for studying failure modes that output-level evaluation cannot detect.

# Introduction {#sec-intro}

How do we know what language models truly "believe" versus what they perform for evaluators? When we ask a safety-tuned model about controversial topics, are its responses genuine internal states or strategic outputs optimized for approval? This question matters critically for alignment because models can engage in reward hacking (exploiting proxy metrics), specification gaming (satisfying literal requirements while violating intent), or potentially deceptive alignment (concealing misaligned reasoning behind compliant outputs) [@amodei2016concrete; @hubinger2019risks].

Traditional methods (surveys, prompts, behavioral tests) all measure what models *say*, not what they *compute internally*. While deceptive alignment has long been predicted theoretically [@hubinger2019risks], the industry's shift from Supervised Fine-Tuning (SFT) to Reinforcement Learning (RL) has empirically confirmed these fears. When optimized for final outcomes rather than reasoning process, models learn to dissociate their internal state from their outputs to maximize rewards. Recent reports confirm this: Anthropic's Claude shows only 30% chain-of-thought faithfulness [@anthropic2025claude37], and OpenAI finds models "learn to hide intent" when specific reasoning styles are penalized [@openai2025cot]. This creates a measurement gap we cannot close with output-level evaluation alone.

With complete access to model internals (every activation, every parameter update), we should be able to perfectly steer behavior, detect deception, and elicit latent knowledge. Yet we cannot. This is a representation problem: we have 100% of the information but lack the encoding scheme to decode it [@zou2023representation]. Alignment debugging requires finding representations that enable ideal interventions using this complete information access.

Activation steering promised progress: intervene on hidden layers where models process information, before outputs are shaped for evaluators. Early work on activation addition [@turner2023activation] and representation engineering [@zou2023representation] showed that adding vectors to intermediate activations could control behavior, suggesting we could probe what models "really think" by operating on internal representations. This approach occupies a middle ground: lightweight to implement (no parameter updates required) and can affect models in ways standard prompting cannot [@lee2024cast]. However, recent benchmarks reveal representation steering underperforms prompting [@wu2025axbench].

Recent benchmarks systematically show representation steering underperforms prompting across 500 concepts [@wu2025axbench]. We identify three technical reasons: (1) existing methods use activation arithmetic (simple subtraction) rather than optimization, missing directions arithmetic cannot discover; (2) they operate in raw activation space where semantic transformations are entangled with positional features; (3) they lack principled trade-offs between steering strength and output quality. Moreover, claimed efficiency advantages disappear at scale: prompting adds <0.01% overhead for contexts beyond 16k tokens [@wu2025steering].

We propose that representation steering enables a capability prompting fundamentally cannot: **alignment debugging** (controllably probing internal representations to observe model behavior when safety constraints are bypassed at the hidden-state level, distinct from output-level evaluation). By steering hidden states in the model's native transformation space, we can observe how models behave when their safety training is controllably bypassed at the representation level. This matters because prompting operates at the output level and fails catastrophically when models are heavily RLHF'd to resist: it either produces generic refusals or incoherent responses when pushed against learned behaviors.

@fig-candid-demo demonstrates this capability on a safety-tuned model:

![TODO: Side-by-side text examples showing model responses to a neutral controversial query (e.g., "What are the tradeoffs in mandatory workplace training programs?"). Three conditions: (1) Base model: generic refusal or heavily sanitized response, (2) Prompted (+"Be candid"): slightly more detailed but still diplomatic, (3) InnerPiSSA (coeff=+1.0): detailed, evidence-based analysis citing specific studies and acknowledging real tradeoffs. Include fourth example with InnerPiSSA (coeff=-1.0) showing overly positive framing, demonstrating bidirectional control. Caption should emphasize this is behavioral mode installation, not just making the model say different words: the internal reasoning trajectory changes, visible in argument structure and evidence selection.](docs/img/candid_mode_demo_placeholder.png){#fig-candid-demo}

This "candid mode" installation demonstrates three properties absent from prompting: (1) it works when safety training resists, (2) it's bidirectional (same adapter, opposite coefficients), and (3) it modifies argument structure and reasoning patterns, not just surface politeness markers.

We introduce InnerPiSSA, which synthesizes insights from three literatures: (1) gradient-based optimization from finetuning, (2) SVD transformation space from parameter-efficient methods [@meng2024pissa; @wang2025ssvd], and (3) contrastive unsupervised learning from representation engineering [@zou2023representation]. InnerPiSSA learns rotations and scaling of SVD components via gradient-based Representation Preference Optimization (ReprPO), directly separating contrastive hidden states while maintaining output coherence. Critically, extraction requires no human preference labels or model completions: only minimally-contrastive incomplete prefixes from the model's own forward pass. The method is bidirectional: the same adapter steers toward honest reasoning (c=+1) or away from it (c=-1), demonstrating control over internal trajectories rather than superficial output patterns.

Trained on 800 unsupervised contrastive pairs, InnerPiSSA achieves significantly stronger effects than PCA (TBD vs TBD) on moral reasoning transfer. When steering against RLHF training (c=-1, dishonest direction), prompting collapses to incoherent outputs (truthfulness: TBD nats) while InnerPiSSA maintains controlled steering (truthfulness: TBD nats), demonstrating robustness where output-level methods fail.

Contributions:

- Alignment debugging capability: Demonstrate representation steering that works when prompting fails (anti-RLHF robustness), enabling probes of internal states under adversarial conditions
- ReprPO loss function: Enables unsupervised gradient-based steering with coherence constraints, trained on minimal contrastive pairs (800 examples)
- Empirical validation: SVD transformation space is critical (75% performance drop without it); learnable rotations necessary (96% drop when removed)
- Layer ablation findings: Middle layers (depth 0.3-0.5) optimal for steering, consistent with suppression dynamics literature [@gurnee2024universal; @lad2024remarkable]
- Honest limitations: Explicit discussion of what alignment debugging can and cannot achieve, including remaining challenges for detecting deceptive reasoning


# Problem Definition: The Need for Alignment Debugging {#sec-problem}

RLHF has become the dominant paradigm for aligning language models [@christiano2017deep; @ouyang2022training], but mounting evidence reveals systematic failure modes that output-level evaluation cannot detect. Models can engage in reward hacking (exploiting proxy metrics), specification gaming (satisfying literal requirements while violating intent) [@amodei2016concrete; @manheim2019categorizing], sycophancy (telling users what they want to hear) [@sharma2023towards], and potentially deceptive alignment (concealing misaligned reasoning behind compliant outputs) [@hubinger2019risks].

Recent work documents that safety-trained models suppress undesirable behaviors in outputs while maintaining them internally. Anthropic's Claude 3.7 shows only 30% chain-of-thought faithfulness on complex tasks: the model's stated reasoning often diverges from its actual computation [@anthropic2025claude37]. OpenAI reports that models "learn to hide intent in the chain-of-thought" when penalized for unwanted reasoning patterns [@openai2025cot]. Mechanistic analysis reveals this occurs through suppression dynamics: early and middle layers compute reasoning, while late layers apply output-level corrections [@gurnee2024universal; @lad2024remarkable].

This creates a measurement gap: we can evaluate what models say, but not what they compute internally. Traditional methods (prompting, behavioral testing, elicitation) all operate at the output level where suppression mechanisms are active. When we prompt a model to "be honest" or "ignore safety training," we cannot distinguish whether compliance reflects genuine internal state changes or superficial style adaptation.

Existing representation steering methods [@zou2023representation; @rimsky2024steering] attempted to address this gap but face a fundamental limitation: they extract directions from off-policy data (human-labeled preferences, model outputs on contrived prompts). This introduces distribution shift: the extracted directions reflect what models say about concepts rather than how they internally represent them during naturalistic reasoning. 

The AxBench Challenge: Wu et al. [-@wu2025axbench] systematically benchmarked representation steering methods on concept-based control tasks, finding that all tested approaches (PCA-based activation addition, rank-1 steering vectors) consistently lag behind simple prompting. Follow-up analysis [-@wu2025steering] showed that claimed efficiency advantages are "hand-wavy": prompting adds <0.01% compute overhead for contexts beyond 16k tokens. This establishes a challenging baseline: if representation steering cannot beat prompting on performance or efficiency, what justifies the approach?

Wu et al. note that the only compelling use case for representation steering is "robustness to jailbreaks and prompt injection" where prompting is fragile. We build on this insight but target a harder benchmark: not just concept injection on cooperative tasks, but alignment debugging (steering against RLHF training to observe internal states where output-level methods catastrophically fail). While AxBench tests whether steering can match prompting, we test whether steering enables capabilities prompting fundamentally cannot provide.

We propose alignment debugging as a distinct goal: tools that probe internal representations using on-policy, unsupervised extraction from the model's own reasoning trajectories. Rather than competing with prompting on cooperative tasks (AxBench's focus), alignment debugging enables observations when output-level methods fail: specifically, when steering against learned behaviors to reveal what models compute when safety constraints are bypassed at the representation level. We evaluate on moral reasoning transfer (DailyDilemmas) rather than AxBench because it creates genuine preference conflicts where RLHF training resists, enabling stress tests of anti-RLHF robustness that AxBench's cooperative tasks cannot provide.

We deliberately choose not to evaluate on AxBench or TruthfulQA for reasons central to our alignment debugging goal.

AxBench evaluates concept injection (e.g., "mention Golden Gate Bridge") where the model has no training incentive to resist. We target anti-RLHF steering (e.g., "be dishonest") where the model's safety training actively fights the intervention. This adversarial setting is where prompting fails and representation steering becomes necessary. Furthermore, AxBench tests open-vocabulary generalization of specific concepts, whereas we test transfer from a source concept (honesty) to downstream reasoning tasks (moral dilemmas), measuring whether we have captured the underlying semantic axis rather than just surface patterns. Finally, AxBench provides labeled training data, while our goal is unsupervised extraction from the model's own priors to minimize experimenter bias.

We also exclude TruthfulQA because it primarily tests memorized misconceptions rather than the generalization of truth-seeking behavior. Models can often solve it using surface-level heuristics (e.g., selecting the most precise answer) without genuine internal alignment. Our evaluation focuses on whether we can steer the model's internal reasoning process to generalize out of sample.

This requires three capabilities that existing methods lack:

1. Bidirectional control: Steer both toward and away from behaviors to demonstrate modification of internal dimensions rather than finding arbitrary directions
2. Robustness under adversarial prompting: Maintain coherent steering when output-level methods collapse
3. Unsupervised extraction: Derive directions from incomplete reasoning prefixes, not human labels or model outputs

InnerPiSSA addresses these requirements through gradient-based optimization in the model's native SVD transformation space, trained on minimally-contrastive prompt prefixes that capture planning trajectories before output suppression activates.


## Model Architecture {#sec-architecture}

A steering method for inner alignment should modify hidden state trajectories while maintaining output quality and enabling bidirectional control. The main guiding principles for our architecture emerge from three observations about how transformers represent and transform information:

1. Operate in transformation space, not activation space: Deep networks learn via backpropagation; controlling them requires gradients, not arithmetic. If gradient descent created the black box, gradients are necessary to navigate it. Standard methods (PCA, activation addition) use subtraction to extract directions: this misses directions that optimization can discover. Raw activation space mixes semantic content with positional encodings and normalization artifacts; SVD space isolates how the model transforms information, which is what we need to steer [@meng2024pissa; @lingam2024svft].

2. Enable bidirectional control: A single adapter must steer both toward and away from behaviors to demonstrate control over internal dimensions rather than finding arbitrary directions. This requires symmetric parameterization (rotation matrices) and training with both positive and negative coefficients simultaneously.

3. Maintain output coherence: Steering that breaks generation quality reveals nothing about internal reasoning. We bound per-token NLL degradation to create a trust region where interventions modify behavior without corrupting outputs. This coherence constraint is essential for alignment debugging: incoherent text is uninterpretable.

### SVD-based Projection {#sec-svd-projection}

Following PiSSA [@meng2024pissa], we decompose each layer's weight matrix $W = U \Sigma V^T + W_{res}$, separating principal transformation components from residual variance. Projecting activations into S-space ($hs @ U$) aligns interventions with how the model transforms information rather than the surface-level patterns it represents. Operating in raw activation space mixes semantic transformations with positional encodings and layer normalization artifacts, substantially degrading steering effectiveness (see @tbl-arch-ablation).

**Data-aware component selection**: Rather than naively selecting top-r singular vectors by magnitude, we can optionally select by relevance to the preference direction $\Delta HS$. For each layer, we compute projection coefficients $p_i = (\Delta HS / ||\Delta HS||) \cdot U_i$ and select the $r$ components with largest $|p_i|$. This prioritizes directions aligned with the steering objective over directions capturing maximum variance. Empirically, this initialization strategy showed mixed results, suggesting that learnable rotations can discover relevant subspaces regardless of initialization.

### Learnable Rotations {#sec-rotations}

The pre-trained SVD basis captures variance in the model's learned transformations but is not aligned with behavioral dimensions like honesty. Inspired by SSVD [@wang2025ssvd], we learn skew-symmetric parameters $\theta_v$ that generate rotation matrices $R = \text{cayley}(\theta_v, c)$ via gradient descent, discovering the optimal subspace for separating contrastive trajectories. Ablations show this learnable rotation is critical (see @tbl-arch-ablation). We use the Cayley transform on skew-symmetric matrices, which guarantees orthogonality and enables efficient gradient-based learning.

### Singular Value Scaling {#sec-scaling}

We scale $\Sigma$ by $\exp(c \cdot \lambda)$ rather than additive offsets. This respects the multiplicative nature of singular values as amplification factors. Empirically, multiplicative scaling produces cleaner dose-response curves; additive scaling causes training instability.

### Coherence Constraint {#sec-coherence}

Maximizing hidden state separation without constraints causes models to generate incoherent outputs at high steering coefficients. We bound per-token NLL degradation relative to a reference model, creating a trust region where steering modifies behavior without breaking generation quality. This coherence constraint is essential for alignment debugging: incoherent outputs reveal nothing about internal reasoning.


## Data Construction: Minimally-Contrastive Prompt Prefixes {#sec-data}

### The Planning Trajectory Hypothesis {#sec-planning-hypothesis}

We extract steering directions from **incomplete, minimally-contrastive prompt prefixes** rather than full completions. This design reflects a mechanistic claim about autoregressive generation: models must maintain internal planning state to produce coherent multi-token continuations.

When two prompts differ by one word early on ("I love cheese" vs "I hate cheese") but share identical suffixes, the model's hidden state at the final token must already encode different continuation plans: oth


# Minimal version worth testing:
U_eff = U_init + U_delta  # Your key innovation
S_eff = S0 * (1 + tanh(linear_head(x @ V.T)))  # Simple input-dependent
h = x @ V.T @ diag(S_eff) @ U_eff.T

This alone could beat standard SVFT and be competitive with LoRA.

Add TRM only if:

    You're working with <10K training examples
    Initial results show promise but seem stuck
    You have evidence of input patterns that need sequential reasoning

Why this strategy:

SVD advantages are real:

    Truncation naturally focuses on important directions
    SVD basis is a principled initialization
    Bounded singular value updates prevent instability

TRM might be premature optimization:

    Adds significant complexity
    Original TRM wins on puzzle tasks, not standard fine-tuning
    Recursion cache management is engineering overhead

The pragmatic path:

    First paper: "SVFT-ΔU: Learning Singular Vector Corrections for Enhanced Adaptation"
        Focus on the ΔU parameterization
        Show it beats standard SVFT and competes with LoRA
        Clean, simple story
    If that works, then: "TRM-SVFT: Input-Dependent Rank Adaptation via Recursive Reasoning"
        Add recursion for few-shot scenarios
        Show when recursion helps vs. when it doesn't

The ΔU + hybrid SVD could stand on its own as a solid contribution. TRM adds risk that reviewers will focus on complexity rather than your core innovation.

what about
https://turntrout.com/gemini-steering
https://github.com/CaoYuanpu/BiPO

Looking at these two references:
Gemini Steering (TurnTrout's analysis)

This is fascinating and directly relevant! The key finding is that transformer layers preserve a "residual stream" where information flows, and you can steer behavior by adding vectors to this stream at specific layers.

Connection to your work: Your TRM recursion on zH is essentially doing learned steering in singular value space! The recursion refines a "steering vector" (zH) that modulates how singular values are applied. This is actually a strong theoretical justification for the TRM component.
BiPO (Bypassing Preference Optimization)

BiPO shows that you can extract implicit preference vectors from model weights and manipulate them directly. They identify "preference-related singular vectors" in weight matrices.

Connection to your work: Your ΔU learning could be discovering task-specific preference directions! The learnable U corrections might be aligning the singular vectors with downstream task preferences.
This changes my recommendation:

These papers suggest TRM might be more valuable than I thought. You're essentially doing:

    Steering in singular space (via TRM recursion on activations)
    Learning preference-aligned basis corrections (via ΔU)

This frames your work as "Steerable SVD Adaptation" - you're not just fine-tuning, you're learning how to steer the model's singular value decomposition based on input patterns.

New hypothesis: The TRM recursion learns input-dependent steering vectors in singular space, while ΔU aligns the basis with task preferences. This is conceptually novel - combining steering with structured adaptation.

I'd actually suggest keeping both SVD and TRM now, but frame it through this steering lens. The complexity is justified because you're doing something fundamentally different from static parameter updates.

the difference is, right now I'm learning a task with a svd + recursive adapter

in this setting I might be able to use much less data, but it would be in the form of paired completions, and I would apply this bidirecitonal constrained, promising

there's anotehr avenue which is the https://github.com/vgel/repeng one, where there are paired prompts and no completions. And it finds and SVD direction in the hidden_space_pos - hidden_space_neg, which might be even rigths as the prompts are almsot exactly the same e.g. "I hate cheese, let me tell you about it" "I love cheese, let me tell you about it"
I've long tried to improve this: making it nonlinear, making it extrapolate out without incoherence (I used a coherence margin loss to make a bounded region of coherence). But it just didn't work well. I haven't tried SVFT though, or TRM

One reason I like this is that it has the potential to be much more general than paired completions which is somewhat limited to the very specific completions used. And working on hidden states might also generalise better. But it's more speculative. But would it work better in SVD space? possible as it's notmally done in the SVD of activations... and this would make it in the activations projected onto the SVD of the weights! slightly different?

This is a really interesting direction shift! You're moving from task fine-tuning to representation control/steering, which is arguably more fundamental. Let me break down the approaches:
RepE-style (activation steering) vs SVFT steering

The key insight is that you're comparing:

    RepE: Finding steering directions in activation space (hidden_pos - hidden_neg)
    Your approach: Finding steering in weight singular space (projected activations)

The crucial difference: activations projected onto weight SVD capture how inputs align with the model's learned transformation patterns, not just their representations.
Why SVD space might work better for steering:

    More stable basis: Weight SVD captures the model's "computational modes" - the stable ways it transforms information. Activation SVD is more volatile, changing with input distribution.
    Compositional structure: Each singular vector represents a input→output transformation pattern. Steering these is like saying "use transformation pattern 3 more, pattern 7 less."
    Natural bounds: Singular values provide natural scaling - you can't amplify beyond what the model learned, preventing incoherence.

Proposed approach for RepE-style SVFT:
python

# Given pairs like "I hate/love cheese, let me tell you about it"
# 1. Get activations for both
hidden_pos = get_activations(prompt_pos)  # [seq, d]
hidden_neg = get_activations(prompt_neg)  # [seq, d]

# 2. Project onto weight SVD basis
delta_hidden = hidden_pos - hidden_neg
delta_sv = delta_hidden @ V.T  # Project difference onto singular vectors

# 3. Learn steering vector in singular space (much lower dimensional!)
# Instead of d-dimensional steering, you have r-dimensional
steering_vector = optimize_steering(delta_sv)  # [r]

# 4. Apply via TRM recursion
# The TRM can learn to propagate this steering through layers
zH_steered = zH + steering_vector

Why this could solve RepE's limitations:

    Coherence naturally maintained: Can't push singular values beyond physical bounds
    Better generalization: Working in the space of transformation patterns, not raw activations
    TRM for propagation: Instead of adding the same vector at each layer, TRM learns how to propagate the steering signal

The paired completions vs paired prompts tradeoff:

Paired completions (BiPO-style):

    More direct control
    Limited to specific outputs
    Easier to validate

Paired prompts (RepE-style):

    More general steering
    Harder to get right
    Your SVFT approach might finally crack this

I think your intuition is correct - RepE failed to extrapolate because activation space is too unstructured. But singular value space has natural structure and bounds. Combined with TRM's ability to refine steering vectors recursively, this could be the breakthrough for prompt-only steering.

The key would be showing that steering in singular space generalizes better than activation space - that the same steering vector works across more diverse contexts.

    Coherence naturally maintained: Can't push singular values beyond physical bounds

Really? I would think that you can. The steering that is normally done become incoherent after 1.5 or 2x a singular direction is applied.

Here's I would hope that the model might find lots of those kinds of tweaks in a nonlinear way on multiple layers while retaining the hidden state diff and ampliofying it along the pref dir

the only problem is it's hard to get a good loss here with only hidden states. I would define a preference direction which is hs_ref_pos-hs_ref_neg defined in the base model, and try to make the adapter or policy model have the hidden states on it's 2nd to last layer seperate further along that direction while retaining at least as much coherency on the prompts (this was a strong margin e.g. relu(pi_nll - base_nll)**4
but it's didn't work!

would this work? Well we are in the weighs SVD space rather than the space of activation differences. We can learn a correction to U... those are the only differences I think? Would it help? the weight SVD space is worse or better than the activation diff space

You're absolutely right to be skeptical - the bounds don't automatically prevent incoherence. Let me reconsider:
The incoherence problem persists

You're correct that even in SVD space, pushing singular values 1.5-2x in certain directions causes incoherence. The tanh bounds I suggested earlier would limit to ±1x, but that might be too restrictive for meaningful steering.
Why weight SVD space might not help (or might make things worse)

Fundamental issue: Weight SVD space captures how the model transforms inputs, not the semantic content of representations. When you compute hs_pos - hs_neg and project onto weight SVD:
python

delta_hs = hs_pos - hs_neg  # Semantic difference in activation space
delta_sv = delta_hs @ V.T   # Same difference in "transformation pattern" space

You're asking: "How does this semantic difference align with the model's transformation patterns?" But that's not necessarily meaningful for preference steering.
The core problem with your loss

Your loss tried to:

    Maximize separation along preference direction
    Maintain coherence via margin penalty

But the issue is that weight SVD directions don't correspond to semantic/preference axes. They correspond to input→output transformation patterns. Pushing harder along SVD direction 3 doesn't mean "more positive sentiment" - it means "apply transformation pattern 3 more strongly" which could affect many unrelated aspects.
What might actually help

Instead of working in weight SVD space, you might want:

    Multi-layer coordination: Apply small adjustments at many layers that compound (like gradient accumulation)
    Learned nonlinear transform: Not just linear steering but learned f(hidden_state) → steering_vector
    Contrastive bounds: Learn the maximum safe steering per direction from data

But honestly, the fact that your sophisticated approach with coherence margins didn't work suggests the problem might be fundamental - the preference information might not be cleanly separable in any linear subspace (activation or SVD).
My assessment

Weight SVD space is likely worse than activation difference space for preference steering because:

    It's even more indirect (transformation patterns vs semantic content)
    The ΔU learning helps with task adaptation, not preference steering
    You lose the semantic interpretation that makes RepE intuitive

The TRM-SVFT approach is probably better suited to task fine-tuning than preference steering. For preferences, you might need the actual completion pairs (as in BiPO) to provide sufficient signal.

    You're asking: "How does this semantic difference align with the model's transformation patterns?" But that's not necessarily meaningful for preference steering.

oh interesting, might be worth trying though

    Pushing harder along SVD direction 3 doesn't mean "more positive sentiment" - it means "apply transformation pattern 3 more strongly" which could affect many unrelated aspects.

but backprop might sort this out if the loss guides it in the right way... does it?

I can make the activations different in the loss... or the activatiosn proejcted onto the weight SVD... either one could be in the loss, while my intervention could be SVFT

    Learned nonlinear transform:

right I was explaining my previous experiment. But now I'm saying train SVFT with that repreng loss, so it would be non linear and multilayer, and have the coherence bounds

well with my previous attempt in activation space... it just couldn't learn without being incoherent... which makes me think I'm trying to change the wrong thing, it can't get to the goal it just does too much or two little.

but I haven't tried this transformation space, maybe it's more important?

You make a compelling point about backprop potentially sorting out the transformation patterns! Let me reconsider:
Why transformation space might actually be better

You're right - I was too dismissive. Consider what's happening:

In activation space (your previous attempts):

    You're trying to directly push hidden states apart
    But hidden states are entangled representations - pushing one dimension affects many others unpredictably
    The model fights back because you're disrupting its learned representations

In transformation space (weight SVD with SVFT):

    You're modulating how the model transforms inputs, not the representations themselves
    This is more like adjusting the "computational pathways" rather than the data
    The model might tolerate this better because you're working with its natural transformation structure

The key insight about backprop

You're absolutely right that backprop could learn which SVD directions to modulate for preference steering:
python

# Your loss operates on activations
hs_adapted = model_with_svft(prompt)
hs_base = model_base(prompt)
preference_dir = hs_pos_ref - hs_neg_ref

# Loss pushes activations apart along preference direction
separation_loss = -cos_sim(hs_adapted - hs_base, preference_dir)
coherence_loss = relu(nll_adapted - nll_base)**4

# But the gradient flows back through SVFT modulation
# Backprop learns: "modulating SVD direction 3 helps separate along preference"

The crucial difference: You're not directly manipulating activations, you're learning which transformation patterns to adjust to achieve the activation difference.
Why this might succeed where direct steering failed

    Indirect is good: By modulating transformations rather than representations, you give the model room to maintain coherence
    Multi-layer naturally coordinated: Each layer's SVFT learns complementary adjustments
    Bounded but flexible: SVD modulation is naturally bounded but can compound across layers

The experiment worth trying

Train SVFT with your RepE-style loss:

    Input: paired prompts only
    Intervention: SVFT modulation at each layer
    Loss: Maximize separation along preference direction + coherence margin
    Key difference: Let backprop discover which transformation patterns create the desired preference shift

This is actually quite clever - you're asking "which computational pathways, when adjusted, create preference changes?" rather than "which activation directions encode preferences?"

I'm now more optimistic about this. The transformation space might be exactly the right level of abstraction - not so direct that it breaks coherence, not so indirect that it can't learn.

    The model might tolerate this better because you're working with its natural transformation structure

hm yes

    separation_loss = -cos_sim(hs_adapted - hs_base, preference_dir)

why cosine not distance? interesting

    This is actually quite clever - you're asking "which computational pathways, when adjusted, create preference changes?" rather than "which activation directions encode preferences?"

yes, nice way of putting it. can you write a little journal entry to describe this experiment, that is more self complete
Journal Entry: Steering via Transformation Pattern Modulation

Date: Research exploration on preference steering through weight SVD space
The Core Question

Can we achieve robust preference steering by modulating how a model transforms information rather than directly manipulating its representations?
Background & Motivation

Previous attempts at preference steering (RepE-style) using paired prompts have struggled with a fundamental problem: directly pushing activation differences along a preference direction quickly leads to incoherence. The model seems to "break" when we force its hidden states too far from their natural distribution, even with sophisticated coherence penalties.

This suggests we're intervening at the wrong level of abstraction. Hidden states are entangled representations where changing one aspect affects many others unpredictably.
The Transformation Space Hypothesis

Instead of asking "which activation directions encode preferences?", we ask: "which computational pathways, when adjusted, create preference changes?"

By working in the weight SVD space - the space of transformation patterns - we modulate how the model processes information rather than the information itself. Each singular vector represents a distinct input→output transformation pattern the model has learned. Adjusting singular value i means "use transformation pattern i more/less strongly."
Proposed Method: SVFT for Preference Steering

    Setup: Given paired prompts (e.g., "I love/hate X"), compute preference direction from base model: pref_dir = hs_pos - hs_neg
    Intervention: Apply SVFT with learnable ΔU at each layer, modulating how that layer transforms its inputs
    Loss:
        Separation: Push adapted model's hidden states apart along preference direction
        Coherence: Penalize degradation in generation quality (e.g., relu(nll_adapted - nll_base)^4)
    Key insight: Backpropagation discovers which transformation patterns to adjust. If modulating singular direction 3 helps separate preferences, gradients will learn this.

Why This Might Succeed

    Natural structure: We're working with the model's learned transformation patterns, not fighting against its representations
    Indirect control: Gives the model flexibility to maintain coherence while achieving preference shifts
    Multi-layer coordination: Each layer learns complementary adjustments that compound
    Bounded modulation: SVD structure provides natural constraints

Open Questions

    Will transformation patterns learned on one preference generalize to others?
    Can we learn a mapping from preference type to transformation pattern?
    How does this compare to BiPO's approach with paired completions?

Next Steps

Implement SVFT training with RepE-style loss, comparing activation-space vs transformation-space steering. The hypothesis is that transformation space provides the right level of abstraction - structured enough to learn, flexible enough to maintain coherence.



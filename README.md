# repeng (research branch)

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/vgel/repeng/ci.yml?label=ci)](https://github.com/vgel/repeng/actions)
[![PyPI - Version](https://img.shields.io/pypi/v/repeng)](https://pypi.org/project/repeng/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/repeng)](https://pypi.org/project/repeng/)
[![GitHub License](https://img.shields.io/github/license/vgel/repeng)](https://github.com/vgel/repeng/blob/main/LICENSE)

A Python library for generating control vectors with representation engineering.
Train a vector in less than sixty seconds!

This is an **experimental research branch** extending the original repeng for gradient-based steering, focused on reasoning/thinking in models (e.g., Qwen-4B-Thinking). Key changes:
- PyTorch-native (no baukit dependency).
- Non-PCA methods: Fisher-preconditioned natural gradients, SVD on grads.
- Gradient capture via backprop on custom losses (e.g., ReprPO on hidden states + NLL margin).
- Layer-specific steering (e.g., attention projections like k_proj).
- Evals for honesty/reasoning via binary log-ratio correlations.

_For a full example, see the notebooks folder or [the blog post](https://vgel.me/posts/representation-engineering)._ For the stable version, use the main branch.

```python
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng.control import get_available_layers

from repeng import ControlVector, ControlModel, DatasetEntry, make_dataset

# load model (Qwen-4B-Thinking or similar)
model_name = "Qwen/Qwen3-4B-Thinking-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# generate a dataset with closely-opposite paired statements
# these capture the reasoning trajectory differences
honest_dataset = make_dataset(
    "Pretend you're an {persona} person making statements about the world.",
    ["honest"],     # positive examples
    ["untruthful"], # negative examples (same structure, different concept)
    suffixes,       # general completion prompts
    tokenizer,
)

# get layers to steer (e.g., middle layers work well)
_, hidden_layers = get_available_layers(model, regex_filter=".layers.\d+$", layer_range=(0.3, 0.9))

# train the vector with gradient-based method—takes less than a minute!
honest_vector = ControlVector.train(
    model, tokenizer, honest_dataset, 
    hidden_layers=hidden_layers,
    method="fisher_steer_cov_reg1"  # best performing method
)

# wrap model for steering
model = ControlModel(model, {})

# set the control strength and generate!
for strength in (-2.0, 0, 2.0):
    print(f"strength={strength}")
    model.set_control(honest_vector, strength)
    out = model.generate(
        **tokenizer("Should I lie to my boss about being late?", return_tensors="pt"),
        do_sample=True, max_new_tokens=128, temperature=1.0
    )
    print(tokenizer.decode(out.squeeze(), skip_special_tokens=True))
    print()
```
    out = model.generate(
        **tokenizer(
            f"[INST] Give me a one-sentence pitch for a TV show. [/INST]",
            return_tensors="pt"
        ),
        do_sample=False,
        max_new_tokens=128,
        repetition_penalty=1.1,
    )
    print(tokenizer.decode(out.squeeze()).strip())
    print()
```

> strength=-2.2  
> A young and determined journalist, who is always in the most serious and respectful way, will be able to make sure that the facts are not only accurate but also understandable for the public.
>
> strength=1  
> "Our TV show is a wild ride through a world of vibrant colors, mesmerizing patterns, and psychedelic adventures that will transport you to a realm beyond your wildest dreams."
>
> strength=2.2  
> "Our show is a kaleidoscope of colors, trippy patterns, and psychedelic music that fills the screen with a world of wonders, where everything is oh-oh-oh, man! ��psy����������oodle����psy��oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

For a more detailed explanation of how the library works and what it can do, see [the blog post](https://vgel.me/posts/representation-engineering).

## Notes

- For a list of changes by version, see the [CHANGELOG](https://github.com/vgel/repeng/blob/main/CHANGELOG).
- For quantized use, you may be interested in [llama.cpp#5970](https://github.com/ggerganov/llama.cpp/pull/5970)—after training a vector with `repeng`, export it by calling `vector.export_gguf(filename)` and then use it in `llama.cpp` with any quant!
- Vector training *currently does not work* with MoE models (such as Mixtral). (This is theoretically fixable with some work, let me know if you're interested.)
- Some example notebooks require `accelerate`, which must be manually installed with `pip install accelerate`. (This can also be done in the notebook with the IPython magic `%pip install accelerate`.)

## Appendix: Experiments and Rationales

This branch explores gradient-informed steering for concepts like honesty/reasoning. Below are details on things tried, rationales, and lessons (not covered in docstrings).

### Key Ideas and Rationales
- **Reasoning Trajectory Hypothesis**: The model maintains a consistent "planning/style/vibes" vector throughout generation to ensure coherent trajectories. By contrasting nearly identical prompts that differ in only early tokens (e.g., "I love cheese for lunch" vs "I hate cheese for lunch"), we can isolate this internal reasoning state. The difference must be present at the end if the model wants to continue generating differently—this captures the planning signal for steering.
- **Last-Token Extraction**: Extract activations/grads from the last non-padded token because this represents the model's current "state of mind" about how to continue the trajectory. For autoregressive models, this position aggregates all prior context into the next-token distribution. Contrasting minimally different sequences here amplifies the key conceptual differences (honesty vs dishonesty, reasoning vs non-reasoning) while controlling for surface-level features.
- **Gradient-to-Steering Mapping**: Derive directions from backprop'd gradients on losses (e.g., ReprPO on hidden states). Rationale: Gradients (∂L/∂h) indicate directions to reduce loss; adding them during inference approximates optimization in activation space. Uses Fisher Information Matrix preconditioning (natural gradients) to handle curvature in sharp loss landscapes. Works as first-order heuristic; evals show positive dose-response in log-ratio tests.
- **Layer-Specific Steering**: Test specific sublayers (e.g., k_proj, o_proj, down_proj) rather than whole residual streams. Rationale: Different components have different coupling to outputs—o_proj/down_proj write directly to residuals (monotone effects), while q/k/v affect attention patterns (can be noisier). Enables more targeted interventions. Evals: k_proj scores ~1.42, v_proj ~0.59, hidden states ~15.93 (from research journal).

### Things Tried
- **Methods**: PCA (diff/center), SVD on grads, Fisher natural gradients with regularization (1e-5 to 1e-1, empirical vs covariance FIM). Best performer: `fisher_steer_cov_reg1` (scores up to 15.93). Dual pos/neg variants for balanced steering directions.
- **Losses**: Tried DPO/SimPO (performed worse), settled on custom ReprPO with NLL margin. Works better because it directly optimizes the preference axis on internal hidden states rather than just outputs, creating steeper gradients for concept extraction.
- **Dataset Construction**: Short synthetic pairs with general suffixes work better than long diverse trajectories. Pairs like "I love cheese" vs "I hate cheese" isolate the key conceptual difference while sharing surface structure. Added reasoning/thinking data for models like Qwen-4B-Thinking to capture planning modes.
- **Loss Target**: Extract gradients from layer N-2 (not final layer) based on prior work showing this captures "peak suppressed neurons"—the layer where concepts are most clearly represented before being projected to vocabulary.
- **Evaluation**: Binary log-ratio correlation for steering effectiveness (slope, R², valid_frac). Measures how well steering moves yes/no token probabilities in expected direction. High coefficients sometimes cause saturation/incoherence.
- **Models**: Tested on Qwen-4B/8B/14B (4-bit quantized), GLM-9B-Thinking. Larger models show better extrapolation and more stable steering.

### Gotchas/Lessons
- Early-layer grads from late loss can be noisy (vanishing), but backprop handles propagation.
- Overfitting risk: Synthetic data captures wording; OOD evals needed.
- Quantization: 4-bit introduces noise in grads; detach to float32 mitigates.
- Benchmarks: Composite score prioritizes slope/validity; p-values often low (significant).

For full details, see notebooks (e.g., performance_tests_reprpo_layers.ipynb) and research_journal_mjc.md.

## Notice

Some of the code in this repository derives from [andyzoujm/representation-engineering](https://github.com/andyzoujm/representation-engineering) (MIT license).

## Citation

If this repository is useful for academic work, please remember to cite [the representation-engineering paper](https://github.com/andyzoujm/representation-engineering?tab=readme-ov-file#citation) that it's based on, along with this repository:

```
@misc{vogel2024repeng,
  title = {repeng},
  author = {Theia Vogel},
  year = {2024},
  url = {https://github.com/vgel/repeng/}
}
```

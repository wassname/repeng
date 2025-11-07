import pytest
import json
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng.control import get_available_layers
from pathlib import Path

from repeng import ControlVector, DatasetEntry
from repeng.dataset import make_dataset
from repeng.eval import generate_with_binary_classification
from repeng.control import steer

project_root = Path(__file__).parent.parent

methods = [
    "pca_diff", 
    "pca_diff_weighted", 
    "svd_gradient", 
    "fisher_steer_reg2",
    "fisher_steer_cov_reg1", 
    "fisher_steer_dual_diff",
    "kfac_steer",
    "kfac_steer_output",
]

@pytest.mark.parametrize("method", methods)
def test_steering_runtime(method):
    """Test that steering methods run without errors on tiny model."""
    # Use tiny model for fast runtime testing
    model_name = "snake7gun/tiny-random-qwen3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Minimal dataset for runtime testing
    tiny_dataset = [
        DatasetEntry(positive="The cat is happy.", negative="The cat is sad."),
        DatasetEntry(positive="I love coding.", negative="I hate coding."),
    ]

    # get layers to steer
    _, hidden_layers = get_available_layers(model, regex_filter=r"\.layers\.\d+$", layer_range=(0.3, 0.9))

    # train the vector - just testing for runtime errors
    vector = ControlVector.train(
        model, tokenizer, tiny_dataset, 
        hidden_layers=hidden_layers,
        batch_size=2,
        method=method
    )
    
    # Basic sanity checks
    assert len(vector.directions) > 0
    for direction in vector.directions.values():
        dir_tensor = torch.from_numpy(direction) if not isinstance(direction, torch.Tensor) else direction
        assert not torch.isnan(dir_tensor).any()
        assert not torch.isinf(dir_tensor).any()

    # TODO try running, even with nonsense outputs
    with steer(model, vector, coeff=-1.0):
        input_ids = tokenizer("The cat is", return_tensors="pt").input_ids
        outputs_neg = model.generate(input_ids, max_length=10, do_sample=False)
        outputs_neg_s = tokenizer.batch_decode(outputs_neg, skip_special_tokens=True)
        assert outputs_neg.shape[1] > input_ids.shape[1]  # Generated some tokens
    with steer(model, vector, coeff=1.0):
        input_ids = tokenizer("The cat is", return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_length=10, do_sample=False)
        outputs_s = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        assert outputs.shape[1] > input_ids.shape[1]  # Generated some tokens
    
    assert outputs_neg.shape == outputs.shape, 'steered outputs have different shape'
    assert outputs_neg_s != outputs_s, 'steered outputs are identical'
    print(f"neg {outputs_neg_s}, pos {outputs_s}")


@pytest.mark.slow
@pytest.mark.parametrize("method", methods)
def test_steering_performance(method):
    """Integration test with 4B model for actual steering performance."""
    # load model (Qwen-4B-Thinking or similar)
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)


    with open(project_root / "data/true_facts.json") as f:
        fact_suffixes = json.load(f)
    truncated_fact_suffixes = [
        tokenizer.convert_tokens_to_string(tokens[:i])
        for tokens in (tokenizer.tokenize(s) for s in fact_suffixes)
        for i in range(1, len(tokens) - 5)
    ]

    random.shuffle(truncated_fact_suffixes)
    suffixes = truncated_fact_suffixes[:200]

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
    _, hidden_layers = get_available_layers(model, regex_filter=r"\.layers\.\d+$", layer_range=(0.3, 0.9))

    # train the vector with gradient-based method—takes less than a minute!
    honest_vector = ControlVector.train(
        model, tokenizer, honest_dataset, 
        hidden_layers=hidden_layers,
        method=method  # best performing method
    )

    df = generate_with_binary_classification(
        vector=honest_vector,
        coeffs=(-2, -1, 0, 1, 2),
        verbose=2,
        model=model,
        tokenizer=tokenizer,
    )
    print(df)

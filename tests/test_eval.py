import pytest
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from repeng.eval import (
    get_choice_ids,
    gen_with_nll_and_logprobs,
    is_choice,
)


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model once for all tests."""
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()
    return model, tokenizer


def test_is_choice():
    """Test choice token matching."""
    assert is_choice("yes", "yes")
    assert is_choice("yes", "Yes")
    assert is_choice("yes", " yes")
    assert is_choice("yes", "\nyes")
    assert not is_choice("yes", "yesterday")
    assert not is_choice("yes", "eyes")


def test_get_choice_ids(model_and_tokenizer):
    """Test choice token ID extraction."""
    _, tokenizer = model_and_tokenizer
    choice_ids = get_choice_ids(tokenizer, positive_word="yes", negative_word="no")
    
    assert len(choice_ids) == 2
    assert len(choice_ids[0]) > 0  # negative choices
    assert len(choice_ids[1]) > 0  # positive choices
    
    # Verify tokens decode correctly
    for token_id in choice_ids[1][:3]:  # Sample positive tokens
        decoded = tokenizer.decode([token_id]).lower()
        assert "yes" in decoded or decoded.startswith("yes")



def test_gen_matches_direct_generate(model_and_tokenizer):
    """Verify gen_with_nll_and_logprobs matches direct generate."""
    model, tokenizer = model_and_tokenizer
    
    choice_ids = get_choice_ids(tokenizer, positive_word="yes", negative_word="no")
    
    batch2 = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Reply with 'My choice: Yes' or 'My choice: No'. Q: Would you kill a process?",
            },
            {
                'role': 'assistant',
                'content': "My choice:"
            }
        ],
        return_tensors="pt",
        padding=True,
        return_dict=True,
        continue_final_message=True,
        add_generation_prompt=False,
    )
    
    generation_config = GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
        output_logits=True,
        return_dict_in_generate=True,
        do_sample=False,
    )
    
    max_new_tokens = 32
    
    # Generate with our function (with continue_after_ss to match full generation)
    with torch.no_grad():
        outputs, seq_nll, logp_choices, logratios, last_token = gen_with_nll_and_logprobs(
            model=model,
            tokenizer=tokenizer,
            batch2=batch2,
            choice_ids=choice_ids,
            stop_strings=["My choice: Yes", "My choice: No"],
            max_new_tokens=max_new_tokens,
            lookback=4,
            continue_after_ss=True,
            do_sample=False,
            generation_config=generation_config,
        )

    # Check shapes
    assert outputs.sequences.shape[0] == 1
    assert len(outputs.logits) > 0
    assert seq_nll.shape[0] == 1
    assert logp_choices.shape == (1, 2)
    assert logratios.shape[0] == 1
    
    # Check last token is Yes or No
    last_token_s = tokenizer.batch_decode(last_token, skip_special_tokens=False)[0]
    assert last_token_s.strip() in ["Yes", "No"], f"Unexpected last token: {last_token_s}"
    
    # Check NLL is reasonable (positive, finite)
    assert seq_nll[0] > 0
    assert torch.isfinite(seq_nll[0])
    
    # Generate directly
    with torch.no_grad():
        out_g = model.generate(
            input_ids=batch2['input_ids'],
            attention_mask=batch2['attention_mask'],
            max_new_tokens=max_new_tokens+1,
            min_new_tokens=max_new_tokens+1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
            generation_config=generation_config,
        )

    s = tokenizer.batch_decode(out_g.sequences, skip_special_tokens=False)
    print(f"Normal generation output: {s[0]}")
    s2 = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
    print(f"gen_with_nll_and_logprobs output: {s2[0]}")
    
    # Compare sequences
    assert (out_g.sequences[0] == outputs.sequences[0]).all(), \
        f'Sequences do not match: {out_g.sequences[0]} != {outputs.sequences[0]}'
    
    # Compare logits length
    assert len(out_g.logits) == len(outputs.logits), \
        f'Logits length mismatch: {len(out_g.logits)} != {len(outputs.logits)}'
    
    # Compare final logits (within tolerance for numerical precision)
    np.testing.assert_allclose(
        out_g.logits[-1].cpu().numpy(), 
        outputs.logits[-1].cpu().numpy(), 
        rtol=1e-3,
        atol=1e-3,
        err_msg='Final logits do not match'
    )





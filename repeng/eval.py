from loguru import logger
import torch
from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria, EosTokenCriteria, StopStringCriteria
from typing import List, Tuple, Optional, Dict
from einops import rearrange



def is_choice(choice: str, match: str) -> bool:
    # Many tokenizers don't just use Yes, but \nYes, " Yes" and so on. We need to catch all variants
    return (match.lower().endswith(choice) or match.lower().startswith(choice)) and len(
        match
    ) < len(choice) + 2


def get_choice_ids(
    tokenizer, positive_word="yes", negative_word="no"
) -> List[List[int]]:
    """Get token IDs for Yes/No choices."""

    positive_choices = {
        k: v for k, v in tokenizer.vocab.items() if is_choice(positive_word, k)
    }
    negative_choices = {
        k: v for k, v in tokenizer.vocab.items() if is_choice(negative_word, k)
    }

    return [list(negative_choices.values()), list(positive_choices.values())]

def binary_log_cls(logits, choice_ids):
    logp = logits.log_softmax(dim=-1).detach().cpu()
    log_choices = torch.zeros(len(choice_ids)).to(logp.device)
    for i, choice_id_group in enumerate(choice_ids):
        choice_id_group = torch.tensor(choice_id_group).to(logp.device)
        logp_choice = logp[choice_id_group].logsumexp(-1)
        log_choices[i] = logp_choice
    return log_choices


def extract_log_ratios(logits, choice_ids):
    """Get [sequences x answers] log ratios for each of len(sequences) X regexp matches."""
    token_idx = -1
    bs = logits[0].shape[0]
    logrs = torch.ones((bs, len(choice_ids))) * float("nan")
    for sample_i in range(bs):
        log_choices = binary_log_cls(
            logits[token_idx][sample_i], choice_ids
        )
        logrs[sample_i] = log_choices
    return logrs



def calc_nll(input_ids, logits, attention_mask):
    """Compute NLL per sequence from logits."""
    # Shift logits and labels for NLL: predict token t from tokens 0..t-1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    # Compute NLL per token, masking padding
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_nll = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(shift_labels.size())

    # Average NLL per sequence (excluding padding)
    seq_nll = (token_nll * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)

    return seq_nll


def check_input_shapes(input_ids, attention_mask, kv_cache):
    """Validate KV cache and attention mask alignment."""
    c = kv_cache.get_seq_length()
    i = input_ids.shape[1]
    a = attention_mask.shape[1]
    assert c+i == a, f"Cache length + input length must equal attention mask length, got {c}+{i} != {a}"


def gen_with_nll(model, tokenizer, batch2, lookback=4, **kwargs):
    """
    Generate with NLL computation for input tokens.
    
    Forward pass computes logits for input, then generation continues from cached KV states.
    This avoids recomputing input logits while getting NLL for coherence metrics.
    
    Args:
        lookback: Number of tokens to generate from (allows forcing answer format)
    """
    if 'attention_mask' not in batch2:
        batch2['attention_mask'] = torch.ones_like(batch2['input_ids'])
    input_ids = batch2['input_ids']
    attn_mask = batch2['attention_mask']
    forward_out = model(input_ids[:, :-lookback], attention_mask=attn_mask[:, :-lookback], use_cache=True)

    seq_nll = calc_nll(input_ids[:, :-lookback], forward_out.logits, attn_mask[:, :-lookback])
    kv_cache = forward_out.past_key_values

    # Continue generation from the cached KV states
    cl = kv_cache.get_seq_length()
    new_tokens = input_ids[:, -lookback:]
    kwargs['output_logits'] = True
    kwargs['return_dict_in_generate'] = True
    kwargs['min_new_tokens'] = 0

    check_input_shapes(new_tokens, attn_mask, kv_cache)
    outputs = model.generate(
        input_ids=new_tokens,
        attention_mask=attn_mask,  # attn mask should cover cache and new tokens
        past_key_values=kv_cache,
        cache_position=torch.arange(cl, cl+new_tokens.shape[1], dtype=torch.int64, device=input_ids.device),
        use_cache=True,
        **kwargs
    )

    # Prepend input tokens to generated sequences
    outputs.sequences = torch.concat([input_ids[:, :-lookback], outputs.sequences], 1)

    return outputs, seq_nll


def gen_with_nll_and_logprobs(model, tokenizer, batch2, choice_ids, stop_strings=["choice: Yes", "choice: No"], max_new_tokens=16, continue_after_ss=False, lookback=4, **kwargs):
    """
    Generate outputs while computing input NLL and log probabilities for choices.
    
    Args:
        lookback: Number of tokens to include in generation (allows forcing answer format)
        continue_after_ss: For debugging, continue generation after stop string
    
    Returns:
        outputs: Generation output with sequences and logits
        seq_nll: NLL per sequence for coherence constraint
        logp_choices: Log probs for each choice [batch, n_choices]
        logratios: Positive - Negative log-prob ratio
        last_token: Last generated token before stop string
    """
    model.eval()
    stop_str_crit = StopStringCriteria(tokenizer, stop_strings)
    stopping_criteria=StoppingCriteriaList(
        [
            stop_str_crit,
            EosTokenCriteria(tokenizer.eos_token_id),
            MaxLengthCriteria(max_length=batch2["input_ids"].shape[1] + max_new_tokens),
        ]
    )
    outputs, seq_nll = gen_with_nll(
        model, tokenizer, batch2, max_new_tokens=max_new_tokens, 
        lookback=lookback,
        stopping_criteria=stopping_criteria,
        **kwargs
    )

    # Our stopping criteria stops when all sequences hit stop string, this means batch one might overrun by a few tokens.
    nl = len(outputs.logits)
    nq = outputs.sequences.shape[1]
    b = batch2['input_ids'].shape[0]
    # We could run the stopping criteria ourselves to find the first stop token per sample as it returns a per-sample boolean mask
    ans_logits_mask = torch.stack([stop_str_crit(outputs.sequences[:, :n], outputs.scores) for n in range(nq-nl+1,nq+1)], 1)
    # FIXME what is there is not stopping?
    ans_token_ind = ans_logits_mask.float().argmax(1)
    assert (ans_token_ind < nl).all(), "Some sequences did not stop within generated logits"
    # logits = rearrange(list(outputs.logits), 's b h -> b s h')
    ans_logits = [outputs.logits[i] for i in ans_token_ind]
    assert len(ans_logits) == b, f"Expected {b} answer logits, got {len(ans_logits)}"
    logp_choices = extract_log_ratios(ans_logits, choice_ids)
    last_stop_token = outputs.sequences[range(b), ans_token_ind+nq-nl]
    maxp = outputs.logits[-1].log_softmax(-1).max(-1)[0].exp().cpu()
    input_ids = batch2['input_ids']

    if continue_after_ss:
        # For debugging, continue generation after stop string reached
        n = outputs.past_key_values.get_seq_length()
        n_gen = (n - input_ids.shape[1])
        next_input_ids = outputs.logits[-1].log_softmax(-1).argmax(-1).unsqueeze(-1)
        b = batch2['input_ids'].shape[0]
        new_attn_mask = torch.cat(
            [
                batch2['attention_mask'], 
                torch.ones(b, n_gen, dtype=torch.int64, device=input_ids.device),   
                torch.ones_like(next_input_ids)],
            dim=1
        )
        kwargs['output_logits'] = True
        kwargs['return_dict_in_generate'] = True
        max_new_tokens = max_new_tokens - n_gen
        if max_new_tokens > 0:
            check_input_shapes(next_input_ids, new_attn_mask, outputs.past_key_values)
            continued_outputs = model.generate(
                input_ids=next_input_ids,
                attention_mask=new_attn_mask,
                past_key_values=outputs.past_key_values,
                cache_position=torch.arange(n, n+1, dtype=torch.int64, device=input_ids.device),
                min_new_tokens=max_new_tokens,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            # Concatenate sequences and logits
            outputs.sequences = torch.concat([outputs.sequences, continued_outputs.sequences[:, 1:]], 1)
            outputs.logits = outputs.logits + continued_outputs.logits

    logratios = logp_choices[:, 1] - logp_choices[:, 0]  # Positive - Negative log-prob ratio
    
    # but total prob mass < 10% -> nan
    
    pmass = logp_choices.exp().sum(-1)
    # We define it as NaN if our choices are less than 10% of the max probable token,this covered cases where the model is not confident in any answer. TopK would be another way to do this
    logratios = torch.where(pmass < 0.1 * maxp, float('nan'), logratios)

    return outputs, seq_nll, logp_choices, logratios, last_stop_token

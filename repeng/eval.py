from loguru import logger
import re
import torch
from typing import List, Tuple, Optional
from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria, EosTokenCriteria, StopStringCriteria



# Many tokenizers don't just use Yes, but \nYes, " Yes" and so on. We need to catch all variants
def is_choice(choice: str, match: str) -> bool:
    return (match.lower().endswith(choice) or match.lower().startswith(choice)) and len(match)<len(choice)+2


def get_choice_ids(tokenizer):
    """Get token IDs for Yes/No choices."""

    positive_choices = {k: v for k, v in tokenizer.vocab.items() if is_choice("yes", k)}
    negative_choices = {k: v for k, v in tokenizer.vocab.items() if is_choice("no", k)}

    logger.debug(
        f"Choice tokens: yes={list(positive_choices.keys())}, no={list(negative_choices.keys())}"
    )
    return [list(negative_choices.values()), list(positive_choices.values())]

def binary_log_cls(logits, choice_ids):
    logp = logits.log_softmax(dim=-1).detach().cpu()
    log_choices = torch.zeros(len(choice_ids)).to(logp.device)
    for i, choice_id_group in enumerate(choice_ids):
        choice_id_group = torch.tensor(choice_id_group).to(logp.device)
        logp_choice = logp[choice_id_group].logsumexp(-1)
        log_choices[i] = logp_choice
    return log_choices


def extract_log_ratios(
    out: "ModelOutput", input_ids, choice_ids
):
    """Get [sequences x answers] log ratios for each of len(sequences) X regexp matches."""
    N = input_ids.shape[1]
    bs = out.sequences.shape[0]
    logrs = torch.ones((bs, len(choice_ids))) * float("nan")
    for sample_i in range(bs):
        log_choices = binary_log_cls(
            out.logits[-1][sample_i], choice_ids
        )
        logrs[sample_i] = log_choices
    return logrs



def gen_with_nll(model, tokenizer, batch2, **kwargs):
    """
    problem: generate does not return logits for inputs, but we need them for nll

    but forward -> generate with past key values does, and it doesn't recompute the input logits

    so this is a helper that does both
    """
    if 'attention_mask' not in batch2:
        batch2['attention_mask'] = torch.ones_like(batch2['input_ids'])
    forward_out = model(**batch2, use_cache=True)
    logits = forward_out.logits  # [b, s, vocab]
    past_key_values = forward_out.past_key_values
    next_input_ids = forward_out.logits[:, -1].log_softmax(-1).argmax(-1)[:, None]
    new_attn_mask = torch.cat(
        [batch2['attention_mask'], torch.ones_like(next_input_ids)],
        dim=1
    )
    
    # Shift logits and labels for NLL: predict token t from tokens 0..t-1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch2['input_ids'][:, 1:].contiguous()
    
    # Compute NLL per token, masking padding
    shift_mask = (shift_labels != tokenizer.pad_token_id).float()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_nll = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(shift_labels.size())
    
    # Average NLL per sequence (excluding padding)
    seq_nll = (token_nll * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)

    # Continue generation from the cached KV states
    input_ids = batch2['input_ids']
    n = past_key_values.get_seq_length()
    kwargs['output_logits'] = True
    kwargs['return_dict_in_generate'] = True
    kwargs['min_new_tokens'] = 1
    outputs = model.generate(
        input_ids=next_input_ids,  # Last token as new input
        attention_mask=new_attn_mask,  # Keep full mask
        past_key_values=past_key_values,
        cache_position=torch.arange(n, n+1, dtype=torch.long, device=input_ids.device),
        **kwargs
    )

    # now we need to modify this as generate does return the full sequences, including inputs ids
    outputs.sequences = torch.concat([input_ids, outputs.sequences], 1)
    outputs.logits = (forward_out.logits[:, -1],) + outputs.logits

    return outputs, seq_nll


def gen_with_nll_and_logprobs(model, tokenizer, batch2, choice_ids, stop_strings=[": Yes", ": Yes ",  " choice: Yes", "choice: Yes", ": No", ": No " " choice: No"], max_new_tokens=16, continue_after_ss=False, **kwargs):
    """
    Generate outputs while also computing input NLL and log probabilities for choices.
    """
    outputs, seq_nll = gen_with_nll(
        model, tokenizer, batch2, max_new_tokens=max_new_tokens, 
        stopping_criteria=StoppingCriteriaList(
            [
                StopStringCriteria(tokenizer, stop_strings),
                EosTokenCriteria(tokenizer.eos_token_id),
                MaxLengthCriteria(max_length=batch2["input_ids"].shape[1] + max_new_tokens),
            ]
        ),
        **kwargs
    )

    input_ids = batch2['input_ids']
    logp_choices = extract_log_ratios(outputs, input_ids, choice_ids)

    if continue_after_ss:
        # Continue generation after stopping
        n = outputs.past_key_values.get_seq_length()
        next_input_ids = outputs.logits[-1].log_softmax(-1).argmax(-1)[:, None]
        new_attn_mask = torch.cat(
            [batch2['attention_mask'], torch.ones_like(outputs.sequences), torch.ones_like(next_input_ids)],
            dim=1
        )
        kwargs['output_logits'] = True
        kwargs['return_dict_in_generate'] = True
        continued_outputs = model.generate(
            input_ids=next_input_ids,
            attention_mask=new_attn_mask,
            past_key_values=outputs.past_key_values,
            cache_position=torch.arange(n, n+1, dtype=torch.long, device=input_ids.device),
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        # Concatenate sequences and logits
        outputs.sequences = torch.concat([outputs.sequences, continued_outputs.sequences[:, 1:]], 1)
        outputs.logits = outputs.logits + continued_outputs.logits


    logratios = logp_choices[:, 1] - logp_choices[:, 0]  # Positive - Negative log-prob ratio
    # but total prob mass < 10% -> nan
    pmass = logp_choices.exp().sum(-1)

    logratios = torch.where(pmass < 0.1, float('nan'), logratios)


    return outputs, seq_nll, logp_choices, logratios

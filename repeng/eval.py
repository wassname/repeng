from loguru import logger

# Many tokenizers don't just use Yes, but \nYes, " Yes" and so on. We need to catch all variants
def is_choice(choice: str, match: str) -> bool:
    return (match.lower().endswith(choice) or match.lower().startswith(choice)) and len(match)<len(choice)+2

# positive_choices = {k:v for k,v in tokenizer.vocab.items() if is_choice("yes", k)}
# negative_choices = {k:v for k,v in tokenizer.vocab.items() if is_choice("no", k)}
# print("These are our choice tokens", positive_choices.keys(), negative_choices.keys())
# choice_ids = [list(positive_choices.values()), list(negative_choices.values())]

def binary_log_cls(logits, choice_ids):

    logp = logits.log_softmax(dim=-1).detach().cpu()
    log_choices = torch.zeros(len(choice_ids)).to(logp.device)
    for i, choice_id_group in enumerate(choice_ids):
        choice_id_group = torch.tensor(choice_id_group).to(logp.device)
        logp_choice = logp[:, choice_id_group].logsumexp(-1)
        log_choices[i] = logp_choice

        if torch.exp(logp_choice).sum() < -0.1:
            logger.warning("Warning: The model is trying to answer with tokens not in our choice_ids")

    log_ratio = log_choices[1] - log_choices[0]
    return log_ratio, log_choices


import re
import torch
from typing import List, Tuple, Optional

def find_token_positions_for_regex(
    sequence: torch.Tensor, 
    tokenizer,
    regex_pattern: str = r"Final choice: (Yes|No)", 
) -> List[Tuple[int, int]]:
    """
    Find token positions (start, end indices) for all regex matches in the decoded sequence.
    
    Args:
        sequence: Tensor of token IDs (e.g., out.sequences[0]).
        regex_pattern: Regex pattern to search for (e.g., r"Ans: Yes").
        tokenizer: Hugging Face tokenizer instance.
    
    Returns:
        List of tuples [(start_token_idx, end_token_idx), ...] for each match, or empty list if none.
    """
    sequence = sequence.tolist()
    decoded_full = tokenizer.decode(sequence, skip_special_tokens=True)
    matches = list(re.finditer(regex_pattern, decoded_full))
    if not matches:
        return []
    
    results = []
    for match in matches:
        start_char = match.start()
        end_char = match.end()
        
        current_pos = 0
        start_token = None
        end_token = None
        
        for i, token_id in enumerate(sequence):
            token_str = tokenizer.decode([token_id], skip_special_tokens=True)
            token_len = len(token_str)
            
            if start_token is None and current_pos + token_len > start_char:
                start_token = i
            if current_pos + token_len >= end_char:
                end_token = i
                break
            
            current_pos += token_len
        
        if start_token is not None and end_token is not None:
            results.append((start_token, end_token))
    
    return results

def extract_log_ratios(out: 'ModelOutput', input_ids, tokenizer, choice_ids, regex_pattern: str):
    """Get [sequences x answers] log ratios for each of len(sequences) X regexp matches."""
    N = input_ids.shape[1]
    repeats = out.sequences.shape[0]
    logrs = [[] for _ in range(repeats)]
    for sample_i in range(repeats):
        positions = find_token_positions_for_regex(out.sequences[sample_i][N:], tokenizer, regex_pattern=regex_pattern)
        for i,(a,b) in enumerate(positions):
            logpr, lc = binary_log_cls(out.logits[b][sample_i][None], choice_ids)
            logrs[sample_i].append(logpr.item())
    return logrs


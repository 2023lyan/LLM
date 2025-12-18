from transformers import PreTrainedTokenizerBase
import torch
from torch import Tensor
from typing import Callable

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    batch_prompt_tokens = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]

    batch_output_tokens = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]

    input_ids = []
    labels = []
    response_mask = []

    concat_tokens = []
    prompt_lens = []
    output_lens = []
    
    for p, o in zip(batch_prompt_tokens, batch_output_tokens):
        concat = p + o
        concat_tokens.append(concat)
        prompt_lens.append(len(p))
        output_lens.append(len(o))
    max_len = max([len(x) for x in concat_tokens])

    for toks, p_len, o_len in zip(concat_tokens, prompt_lens, output_lens):
        pad_len = max_len - len(toks)
        ids = toks + [tokenizer.pad_token_id] * pad_len
        masks = mask = (
            [0] * p_len
            + [1] * o_len
            + [0] * pad_len
        )
        input_ids.append(ids[:-1])
        labels.append(ids[1:])
        response_mask.append(mask[1:])

    results = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
    }
    return results

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    # logits: [batch_size, seq_len, vocab_size]
    # output: [batch_size, seq_len]
    log_Z = torch.logsumexp(logits, dim=-1)
    log_p = logits - log_Z.unsqueeze(-1)
    p = log_p.exp()
    entropy = -(p * log_p).sum(dim=-1)

    return entropy

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    log_Z = torch.logsumexp(logits, dim=-1)
    log_p = logits - log_Z.unsqueeze(-1)
    log_probs = torch.gather(
        log_p, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    result = {
        "log_probs": log_probs
    }

    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy

    return result

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim=dim) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    bs, _ = policy_log_probs.shape
    masked_log_probs = policy_log_probs * response_mask
    loss = -masked_log_probs.sum()
    loss = loss / normalize_constant / gradient_accumulation_steps / bs
    loss.backward()

    metadata = {
        "sum_log_probs": masked_log_probs.sum().detach(),
        "num_tokens": response_mask.sum().detach(),
    }

    return loss.detach(), metadata


def log_generations(
    model,
    tokenizer,
    prompts: list[str],
    ground_truth: list[str],
    rewards: list[dict],
    token_entropies: torch.Tensor,
    max_new_tokens: int = 256,
):
    """
    Logs model generations for validation or training inspection.

    Args:
        model: HF or vLLM model (must implement generate())
        tokenizer: tokenizer
        prompts: list[str]
        ground_truth: list[str]
        rewards: list[dict]   # reward model outputs
        token_entropies: torch.Tensor  # shape (batch, seq_len)

    Returns:
        logs: list[dict]  # each item contains logged information
    """

    logs = []

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    responses = []
    for prompt, gen in zip(prompts, generated_texts):
        if gen.startswith(prompt):
            responses.append(gen[len(prompt):].strip())
        else:
            responses.append(gen)

    for i in range(len(prompts)):
        resp = responses[i]

        ent = token_entropies[i]
        avg_entropy = ent.mean().item()

        resp_len = len(tokenizer(resp).input_ids)

        logs.append({
            "prompt": prompts[i],
            "response": resp,
            "ground_truth": ground_truth[i],
            "reward": rewards[i],
            "avg_token_entropy": avg_entropy,
            "response_length": resp_len,
        })

    return logs

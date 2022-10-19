import numpy as np
import json


def load_vocab_file(vocab_file_path):
    with open(vocab_file_path, "r", encoding="utf-8") as content:
        return json.load(content)


# converts text to tokens
def tokenize(text, tokenizer):
    input_ids = tokenizer(text)['input_ids']
    input_ids = np.array(input_ids).reshape(1, -1)
    return input_ids


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    summation = e_x.sum(axis=-1, keepdims=True)
    return e_x / summation


def process_logits(input_ids, scores, eos_token_id, min_length=0):
    cur_length = input_ids.shape[-1]
    if cur_length < min_length:
        scores[:, eos_token_id] = -float("inf")
    return scores


def get_top_k_logits(scores, top_k):
    filter_value = -float("inf")
    top_k = min(max(top_k, 1), scores.shape[-1])
    top_k_scores = -np.sort(-scores)[:, :top_k]
    indices_to_remove = scores < np.min(top_k_scores)
    filtred_scores = np.ma.array(scores, mask=indices_to_remove,
                                 fill_value=filter_value).filled()
    return filtred_scores


def generate_sequence(input_ids, eos_token_id, compiled_model, output_keys, max_sequence_length=128):
    while True:
        cur_input_len = len(input_ids[0])
        pad_len = max_sequence_length - cur_input_len
        model_input = np.concatenate((input_ids,
                                      [[eos_token_id] * pad_len]), axis=-1)
        # passing the padded sequnce into the model
        outputs = compiled_model(inputs=[model_input])[output_keys]
        next_token_logits = outputs[:, cur_input_len - 1, :]
        # pre-process distribution
        next_token_scores = process_logits(input_ids,
                                           next_token_logits, eos_token_id)
        top_k = 20
        next_token_scores = get_top_k_logits(next_token_scores, top_k)
        # get next token id
        probs = softmax(next_token_scores)
        next_tokens = np.random.choice(probs.shape[-1], 1,
                                       p=probs[0], replace=True)
        # break the loop if max length or end of text token is reached
        if cur_input_len == max_sequence_length or next_tokens == eos_token_id:
            break
        else:
            input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
    return input_ids

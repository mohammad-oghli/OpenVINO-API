import numpy as np
import utils.tokens_bert as tokens

vocab = tokens.load_vocab_file("utils/vocab.txt")
# Define special tokens.
cls_token = vocab["[CLS]"]
sep_token = vocab["[SEP]"]


# A generator of a sequence of inputs.
def prepare_input(entity_tokens, context_tokens, input_keys):
    input_ids = [cls_token] + entity_tokens + [sep_token] + \
                context_tokens + [sep_token]
    # 1 for any index.
    attention_mask = [1] * len(input_ids)
    # 0 for entity tokens, 1 for context part.
    token_type_ids = [0] * (len(entity_tokens) + 2) + \
                     [1] * (len(context_tokens) + 1)

    # Create an input to feed the model.
    input_dict = {
        "input_ids": np.array([input_ids], dtype=np.int32),
        "attention_mask": np.array([attention_mask], dtype=np.int32),
        "token_type_ids": np.array([token_type_ids], dtype=np.int32),
    }

    # Some models require additional position_ids.
    if "position_ids" in [i_key.any_name for i_key in input_keys]:
        position_ids = np.arange(len(input_ids))
        input_dict["position_ids"] = np.array([position_ids], dtype=np.int32)

    return input_dict


def postprocess(output_start, output_end, entity_tokens,
                context_tokens_start_end, input_size):
    def get_score(logits):
        out = np.exp(logits)
        return out / out.sum(axis=-1)

    # Get start-end scores for the context.
    score_start = get_score(output_start)
    score_end = get_score(output_end)

    # Index of the first context token in a tensor.
    context_start_idx = len(entity_tokens) + 2
    # Index of last+1 context token in a tensor.
    context_end_idx = input_size - 1

    # Find the product of all start-end combinations to find the best one.
    max_score, max_start, max_end = find_best_entity_window(
        start_score=score_start, end_score=score_end,
        context_start_idx=context_start_idx, context_end_idx=context_end_idx
    )

    # Convert to context text start-end index.
    max_start = context_tokens_start_end[max_start][0]
    max_end = context_tokens_start_end[max_end][1]

    return max_score, max_start, max_end


def find_best_entity_window(start_score, end_score,
                            context_start_idx, context_end_idx):
    context_len = context_end_idx - context_start_idx
    score_mat = np.matmul(
        start_score[context_start_idx:context_end_idx].reshape(
            (context_len, 1)),
        end_score[context_start_idx:context_end_idx].reshape(
            (1, context_len)),
    )
    # reset candidates with end before start
    score_mat = np.triu(score_mat)
    # reset long candidates (>16 words)
    score_mat = np.tril(score_mat, 16)
    # find the best start-end pair
    max_s, max_e = divmod(score_mat.flatten().argmax(), score_mat.shape[1])
    max_score = score_mat[max_s, max_e]

    return max_score, max_s, max_e


def get_best_entity(entity, context, vocab, compiled_model):
    # Convert the context string to tokens.
    context_tokens, context_tokens_end = tokens.text_to_tokens(
        text=context.lower(), vocab=vocab)
    # Convert the entity string to tokens.
    entity_tokens, _ = tokens.text_to_tokens(text=entity.lower(), vocab=vocab)

    network_input = prepare_input(entity_tokens, context_tokens, list(compiled_model.inputs))
    input_size = len(context_tokens) + len(entity_tokens) + 3

    # OpenVINO inference.
    output_start_key = compiled_model.output("output_s")
    output_end_key = compiled_model.output("output_e")
    result = compiled_model(network_input)

    # Postprocess the result getting the score and context range for the answer.
    score_start_end = postprocess(output_start=result[output_start_key][0],
                                  output_end=result[output_end_key][0],
                                  entity_tokens=entity_tokens,
                                  context_tokens_start_end=context_tokens_end,
                                  input_size=input_size)

    # Return the part of the context, which is already an answer.
    return context[score_start_end[1]:score_start_end[2]], score_start_end[0]

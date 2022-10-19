import operator
from urllib import parse
import numpy as np
from openvino.runtime import Core
import utils.html_reader as reader
import utils.tokens_bert as tokens

# The path to the vocabulary file.
vocab_file_path = "utils/vocab.txt"

# Create a dictionary with words and their indices.
vocab = tokens.load_vocab_file(vocab_file_path)

# Define special tokens.
cls_token = vocab["[CLS]"]
pad_token = vocab["[PAD]"]
sep_token = vocab["[SEP]"]

# Model setting
ie = Core()
model_name = "bert-small-uncased-whole-word-masking-squad-int8-0002"
model_path = f"model/named_er/intel/{model_name}/FP16-INT8/{model_name}.xml"
model_weights_path = f"model/named_er/intel/{model_name}/FP16-INT8/{model_name}.bin"

model_qa = ie.read_model(model=model_path, weights=model_weights_path)
compiled_model_qa = ie.compile_model(model=model_qa, device_name="CPU")

# Get input and output names of nodes.
input_keys = list(compiled_model_qa.inputs)
output_keys = list(compiled_model_qa.outputs)

# Get the network input size.
input_size = compiled_model_qa.input(0).shape[1]


# A function to load text from given urls.
def load_context(sources):
    input_urls = []
    paragraphs = []
    for source in sources:
        result = parse.urlparse(source)
        if all([result.scheme, result.netloc]):
            input_urls.append(source)
        else:
            paragraphs.append(source)

    paragraphs.extend(reader.get_paragraphs(input_urls))
    # Produce one big context string.
    return "\n".join(paragraphs)


# A generator of a sequence of inputs.
def prepare_input(question_tokens, context_tokens):
    # A length of question in tokens.
    question_len = len(question_tokens)
    # The context part size.
    context_len = input_size - question_len - 3

    if context_len < 16:
        raise RuntimeError("Question is too long in comparison to input size. No space for context")

    # Take parts of the context with overlapping by 0.5.
    for start in range(0, max(1, len(context_tokens) - context_len), context_len // 2):
        # A part of the context.
        part_context_tokens = context_tokens[start:start + context_len]
        # The input: a question and the context separated by special tokens.
        input_ids = [cls_token] + question_tokens + [sep_token] + part_context_tokens + [sep_token]
        # 1 for any index if there is no padding token, 0 otherwise.
        attention_mask = [1] * len(input_ids)
        # 0 for question tokens, 1 for context part.
        token_type_ids = [0] * (question_len + 2) + [1] * (len(part_context_tokens) + 1)

        # Add padding at the end.
        (input_ids, attention_mask, token_type_ids), pad_number = pad(input_ids=input_ids,
                                                                      attention_mask=attention_mask,
                                                                      token_type_ids=token_type_ids)

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

        yield input_dict, pad_number, start


# A function to add padding.
def pad(input_ids, attention_mask, token_type_ids):
    # How many padding tokens.
    diff_input_size = input_size - len(input_ids)

    if diff_input_size > 0:
        # Add padding to all the inputs.
        input_ids = input_ids + [pad_token] * diff_input_size
        attention_mask = attention_mask + [0] * diff_input_size
        token_type_ids = token_type_ids + [0] * diff_input_size

    return (input_ids, attention_mask, token_type_ids), diff_input_size


# Based on https://github.com/openvinotoolkit/open_model_zoo/blob/bf03f505a650bafe8da03d2747a8b55c5cb2ef16/demos/common/python/openvino/model_zoo/model_api/models/bert.py#L163
def postprocess(output_start, output_end, question_tokens, context_tokens_start_end, padding, start_idx):
    def get_score(logits):
        out = np.exp(logits)
        return out / out.sum(axis=-1)

    # Get start-end scores for the context.
    score_start = get_score(output_start)
    score_end = get_score(output_end)

    # An index of the first context token in a tensor.
    context_start_idx = len(question_tokens) + 2
    # An index of the last+1 context token in a tensor.
    context_end_idx = input_size - padding - 1

    # Find product of all start-end combinations to find the best one.
    max_score, max_start, max_end = find_best_answer_window(start_score=score_start,
                                                            end_score=score_end,
                                                            context_start_idx=context_start_idx,
                                                            context_end_idx=context_end_idx)

    # Convert to context text start-end index.
    max_start = context_tokens_start_end[max_start + start_idx][0]
    max_end = context_tokens_start_end[max_end + start_idx][1]

    return max_score, max_start, max_end


# Based on https://github.com/openvinotoolkit/open_model_zoo/blob/bf03f505a650bafe8da03d2747a8b55c5cb2ef16/demos/common/python/openvino/model_zoo/model_api/models/bert.py#L188
def find_best_answer_window(start_score, end_score, context_start_idx, context_end_idx):
    context_len = context_end_idx - context_start_idx
    score_mat = np.matmul(
        start_score[context_start_idx:context_end_idx].reshape((context_len, 1)),
        end_score[context_start_idx:context_end_idx].reshape((1, context_len)),
    )
    # Reset candidates with end before start.
    score_mat = np.triu(score_mat)
    # Reset long candidates (>16 words).
    score_mat = np.tril(score_mat, 16)
    # Find the best start-end pair.
    max_s, max_e = divmod(score_mat.flatten().argmax(), score_mat.shape[1])
    max_score = score_mat[max_s, max_e]

    return max_score, max_s, max_e


def get_best_answer(question, context):
    # Convert the context string to tokens.
    context_tokens, context_tokens_start_end = tokens.text_to_tokens(text=context.lower(),
                                                                     vocab=vocab)
    # Convert the question string to tokens.
    question_tokens, _ = tokens.text_to_tokens(text=question.lower(), vocab=vocab)

    results = []
    # Iterate through different parts of the context.
    for network_input, padding, start_idx in prepare_input(question_tokens=question_tokens,
                                                           context_tokens=context_tokens):
        # Get output layers.
        output_start_key = compiled_model_qa.output("output_s")
        output_end_key = compiled_model_qa.output("output_e")

        # OpenVINO inference.
        result = compiled_model_qa(network_input)
        # Postprocess the result, getting the score and context range for the answer.
        score_start_end = postprocess(output_start=result[output_start_key][0],
                                      output_end=result[output_end_key][0],
                                      question_tokens=question_tokens,
                                      context_tokens_start_end=context_tokens_start_end,
                                      padding=padding,
                                      start_idx=start_idx)
        results.append(score_start_end)

    # Find the highest score.
    answer = max(results, key=operator.itemgetter(0))
    # Return the part of the context, which is already an answer.
    return context[answer[1]:answer[2]], answer[0]

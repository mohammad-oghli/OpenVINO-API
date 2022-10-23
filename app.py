from collections import namedtuple
import itertools
import subprocess
from itertools import groupby
import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core, Dimension
from tokenizers import SentencePieceBPETokenizer
#from transformers import GPT2Tokenizer
import utils.tokens_bert as tokens
from cv_utils import load_image, to_rgb, to_gray, convert_result_to_image, multiply_by_ratio, run_preprocesing_on_crop
from ner_utils import get_best_entity
#from qa_utils import load_context, get_best_answer
from gpt2_utils import load_vocab_file, tokenize, generate_sequence

# OpenVino Global Variables
# Models Configurations
ie = Core()
# imagenet_classes = open("utils/imagenet_2012.txt").read().splitlines()
# model_ac = ie.read_model(model="model/animal_classify/v3-small_224_1.0_float.xml")
# compiled_model_ac = ie.compile_model(model=model_ac, device_name="CPU")
#
# output_layer_ac = compiled_model_ac.output(0)
#
# model_rs = ie.read_model(model="model/road_segmentation/road-segmentation-adas-0001.xml")
# compiled_model_rs = ie.compile_model(model=model_rs, device_name="CPU")
#
# input_layer_rs = compiled_model_rs.input(0)
# output_layer_rs = compiled_model_rs.output(0)



# model_ocr = ie.read_model(model="model/ocr/horizontal-text-detection-0001.xml")
# compiled_model_ocr = ie.compile_model(model=model_ocr, device_name="CPU")
#
# input_layer_ocr = compiled_model_ocr.input(0)
# output_layer_ocr = compiled_model_ocr.output("boxes")

# recognition_model = ie.read_model(model="model/ocr/text-recognition-resnet-fc.xml",
#                                   weights="model/ocr/text-recognition-resnet-fc.bin")
#
# recognition_compiled_model = ie.compile_model(model=recognition_model, device_name="CPU")
#
# recognition_output_layer = recognition_compiled_model.output(0)
# recognition_input_layer = recognition_compiled_model.input(0)
#
# model_name = "bert-small-uncased-whole-word-masking-squad-int8-0002"
# model_path = f"model/named_er/intel/{model_name}/FP16-INT8/{model_name}.xml"
# model_weights_path = f"model/named_er/intel/{model_name}/FP16-INT8/{model_name}.bin"
#
# model_ner = ie.read_model(model=model_path, weights=model_weights_path)
# # Assign dynamic shapes to every input layer on the last dimension.
# for input_layer in model_ner.inputs:
#     input_shape = input_layer.partial_shape
#     input_shape[1] = Dimension(1, 384)
#     model_ner.reshape({input_layer: input_shape})
#
# compiled_model_ner = ie.compile_model(model=model_ner, device_name="CPU")
#
# # Get input names of nodes.
# input_keys = list(compiled_model_ner.inputs)
#
# # Set a confidence score threshold.
# confidence_threshold = 0.4
#
# model_name = "machine-translation-nar-en-de-0002"
# model_path = f"model/translate_eg/intel/{model_name}/FP32/{model_name}.xml"
# model = ie.read_model(model=model_path)
# compiled_model = ie.compile_model(model)
# input_name = "tokens"
# output_name = "pred"
# model.output(output_name)
# max_tokens = model.input(input_name).shape[1]
#
# src_tokenizer = SentencePieceBPETokenizer.from_file(
#     'model/translate_eg/intel/machine-translation-nar-en-de-0002/tokenizer_src/vocab.json',
#     'model/translate_eg/intel/machine-translation-nar-en-de-0002/tokenizer_src/merges.txt'
# )
# tgt_tokenizer = SentencePieceBPETokenizer.from_file(
#     'model/translate_eg/intel/machine-translation-nar-en-de-0002/tokenizer_tgt/vocab.json',
#     'model/translate_eg/intel/machine-translation-nar-en-de-0002/tokenizer_tgt/merges.txt'
# )

# model_path = "model/gpt_2/public/gpt-2/FP16/gpt-2.xml"
# model_weights_path = "model/gpt_2/public/gpt-2/FP16/gpt-2.bin"
#
# model_gpt2 = ie.read_model(model=model_path, weights=model_weights_path)
#
# # assign dynamic shapes to every input layer
# for input_layer in model_gpt2.inputs:
#     input_shape = input_layer.partial_shape
#     input_shape[0] = -1
#     input_shape[1] = -1
#     model_gpt2.reshape({input_layer: input_shape})
#
# # compile the model for CPU devices
# compiled_model_gpt2 = ie.compile_model(model=model_gpt2, device_name="CPU")
#
# # get input and output names of nodes
# input_keys_gpt2 = next(iter(compiled_model_gpt2.inputs))
# output_keys_gpt2 = next(iter(compiled_model_gpt2.outputs))
#
# vocal_file_path = "model/gpt_2/public/gpt-2/gpt2/vocab.json"
# vocab_gpt2 = load_vocab_file(vocal_file_path)
#
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#
# eos_token_id = len(vocab_gpt2) - 1
# tokenizer._convert_id_to_token(len(vocab_gpt2) - 1)

# # A path to a vocabulary file.
# vocab_file_path = "utils/vocab.txt"
#
# # Create a dictionary with words and their indices.
# vocab = tokens.load_vocab_file(vocab_file_path)
#
# template = ["building", "company", "persons", "city",
#             "state", "height", "floor", "address"]

download_command_ch = 'omz_downloader --name handwritten-simplified-chinese-recognition-0001 --output_dir model/handwritten_ocr --precision FP16'
#download_command_jap = 'omz_downloader --name handwritten-japanese-recognition-0001 --output_dir model/handwritten_ocr --precision FP16'

#res = subprocess.call(download_command_ch)
#print(res.stdout)
# res = subprocess.run(download_command_jap, capture_output=True)
# print(res.stdout)
import os
os.system(download_command_ch)

Language = namedtuple(
    typename="Language", field_names=["model_name", "charlist_name"]
)
chinese_files = Language(
    model_name="handwritten-simplified-chinese-recognition-0001",
    charlist_name="chinese_charlist.txt",
)
japanese_files = Language(
    model_name="handwritten-japanese-recognition-0001",
    charlist_name="japanese_charlist.txt",
)

# load chinese language files
path_to_model_weights = f'model/handwritten_ocr/intel/{chinese_files.model_name}/FP16/{chinese_files.model_name}.bin'

path_to_model = f'model/handwritten_ocr/intel/{chinese_files.model_name}/FP16/{chinese_files.model_name}.xml'
model_ch_ocr = ie.read_model(model=path_to_model)

compiled_model_ch_ocr = ie.compile_model(model=model_ch_ocr, device_name="CPU")

recognition_output_layer_ch = compiled_model_ch_ocr.output(0)
recognition_input_layer_ch = compiled_model_ch_ocr.input(0)

# load japanese language files
# path_to_model_weights = f'model/handwritten_ocr/intel/{japanese_files.model_name}/FP16/{japanese_files.model_name}.bin'
#
# path_to_model = f'model/handwritten_ocr/intel/{japanese_files.model_name}/FP16/{japanese_files.model_name}.xml'
# model_jap_ocr = ie.read_model(model=path_to_model)
#
# compiled_model_jap_ocr = ie.compile_model(model=model_ch_ocr, device_name="CPU")
#
# recognition_output_layer_jap = compiled_model_jap_ocr.output(0)
# recognition_input_layer_jap = compiled_model_jap_ocr.input(0)


# def cv_animal_classify(image_source):
#     global imagenet_classes
#     raw_image = load_image(image_source)
#     # The MobileNet model expects images in RGB format.
#     image = to_rgb(raw_image)
#
#     # Resize to MobileNet image shape.
#     input_image = cv2.resize(src=image, dsize=(224, 224))
#
#     # Reshape to model input shape.
#     input_image = np.expand_dims(input_image, 0)
#     result_infer = compiled_model_ac([input_image])[output_layer_ac]
#     result_index = np.argmax(result_infer)
#     # Convert the inference result to a class name.
#     # The model description states that for this model, class 0 is a background.
#     # Therefore, a background must be added at the beginning of imagenet_classes.
#     imagenet_classes = ['background'] + imagenet_classes
#
#     class_result = imagenet_classes[result_index].split()[1:]
#     class_result = " ".join(class_result)
#     return class_result
#
#
# def cv_road_segmentation(image_source):
#     # The segmentation network expects images in BGR format.
#     image = load_image(image_source)
#
#     rgb_image = to_rgb(image)
#     image_h, image_w, _ = image.shape
#
#     # N,C,H,W = batch size, number of channels, height, width.
#     N, C, H, W = input_layer_rs.shape
#
#     # OpenCV resize expects the destination size as (width, height).
#     resized_image = cv2.resize(image, (W, H))
#
#     # Reshape to the network input shape.
#     input_image = np.expand_dims(
#         resized_image.transpose(2, 0, 1), 0
#     )
#     # Run the inference.
#     result = compiled_model_rs([input_image])[output_layer_rs]
#
#     # Prepare data for visualization.
#     segmentation_mask = np.argmax(result, axis=1)
#     segmentation_result = segmentation_mask.transpose(1, 2, 0)
#     return segmentation_result
#
#
# def cv_ocr(image_source):
#     # Text detection models expect an image in BGR format.
#     image = load_image(image_source)
#
#     # N,C,H,W = batch size, number of channels, height, width.
#     N, C, H, W = input_layer_ocr.shape
#
#     # Resize the image to meet network expected input sizes.
#     resized_image = cv2.resize(image, (W, H))
#
#     # Reshape to the network input shape.
#     input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
#
#     # Create an inference request.
#     boxes = compiled_model_ocr([input_image])[output_layer_ocr]
#
#     # Remove zero only boxes.
#     boxes = boxes[~np.all(boxes == 0, axis=1)]
#
#     # Get the height and width of the input layer.
#     _, _, H, W = recognition_input_layer.shape
#
#     # Calculate scale for image resizing.
#     (real_y, real_x), (resized_y, resized_x) = image.shape[:2], resized_image.shape[:2]
#     ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
#
#     # Convert the image to grayscale for the text recognition model.
#     grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Get a dictionary to encode output, based on the model documentation.
#     letters = "~0123456789abcdefghijklmnopqrstuvwxyz"
#
#     # Prepare an empty list for annotations.
#     annotations = list()
#     #cropped_images = list()
#     # fig, ax = plt.subplots(len(boxes), 1, figsize=(5,15), sharex=True, sharey=True)
#     # Get annotations for each crop, based on boxes given by the detection model.
#     for i, crop in enumerate(boxes):
#         # Get coordinates on corners of a crop.
#         (x_min, y_min, x_max, y_max) = map(int, multiply_by_ratio(ratio_x, ratio_y, crop))
#         image_crop = run_preprocesing_on_crop(grayscale_image[y_min:y_max, x_min:x_max], (W, H))
#
#         # Run inference with the recognition model.
#         result = recognition_compiled_model([image_crop])[recognition_output_layer]
#
#         # Squeeze the output to remove unnecessary dimension.
#         recognition_results_test = np.squeeze(result)
#
#         # Read an annotation based on probabilities from the output layer.
#         annotation = list()
#         for letter in recognition_results_test:
#             parsed_letter = letters[letter.argmax()]
#
#             # Returning 0 index from `argmax` signalizes an end of a string.
#             if parsed_letter == letters[0]:
#                 break
#             annotation.append(parsed_letter)
#         annotations.append("".join(annotation))
#
#     boxes_with_annotations = list(zip(boxes, annotations))
#
#     # For each detection, the description is in the [x_min, y_min, x_max, y_max, conf] format:
#     # The image passed here is in BGR format with changed width and height. To display it in colors expected by matplotlib, use cvtColor function
#     ocr_result = convert_result_to_image(image, resized_image, boxes_with_annotations, conf_labels=True)
#     return ocr_result, annotations
#
#
# def analyze_entities(context):
#     print(f"Context: {context}\n", flush=True)
#
#     if len(context) == 0:
#         print("Error: Empty context or outside paragraphs")
#         return
#
#     if len(context) > 380:
#         print("Error: The context is too long for this particular model. "
#               "Try with context shorter than 380 words.")
#         return
#
#     extract = []
#     for field in template:
#         entity_to_find = field + "?"
#         entity, score = get_best_entity(entity=entity_to_find,
#                                         context=context,
#                                         vocab=vocab,
#                                         compiled_model=compiled_model_ner)
#         if score >= confidence_threshold:
#             extract.append({"Entity": entity, "Type": field,
#                             "Score": f"{score:.2f}"})
#     res = {"Extraction": extract}
#     return res


def handwritten_ocr(image_source, lang):
    if lang.lower() == "ch":
        compiled_model = compiled_model_ch_ocr
        recognition_input_layer = recognition_input_layer_ch
        recognition_output_layer = recognition_output_layer_ch
        charlist = chinese_files.charlist_name
    elif lang.lower() == "jap":
        compiled_model = compiled_model_jap_ocr
        recognition_input_layer = recognition_input_layer_jap
        recognition_output_layer = recognition_output_layer_jap
        charlist = japanese_files.charlist_name
    else:
        return "Invalid language selected!"
    # Text detection models expect an image in grayscale format.
    raw_image = load_image(image_source)
    image = to_gray(raw_image)
    # Fetch the shape.
    image_height, _ = image.shape

    # B,C,H,W = batch size, number of channels, height, width.
    _, _, H, W = recognition_input_layer.shape

    # Calculate scale ratio between the input shape height and image height to resize the image.
    scale_ratio = H / image_height

    # Resize the image to expected input sizes.
    resized_image = cv2.resize(
        image, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_AREA
    )

    # Pad the image to match input size, without changing aspect ratio.
    resized_image = np.pad(
        resized_image, ((0, 0), (0, W - resized_image.shape[1])), mode="edge"
    )

    # Reshape to network input shape.
    input_image = resized_image[None, None, :, :]
    # With both models, there should be blank symbol added at index 0 of each charlist.
    blank_char = "~"

    with open(f"utils/charlists/{charlist}", "r", encoding="utf-8") as charlist:
        letters = blank_char + "".join(line.strip() for line in charlist)
    # Run inference on the model
    predictions = compiled_model([input_image])[recognition_output_layer]
    # Remove a batch dimension.
    predictions = np.squeeze(predictions)

    # Run the `argmax` function to pick the symbols with the highest probability.
    predictions_indexes = np.argmax(predictions, axis=1)
    # Use the `groupby` function to remove concurrent letters, as required by CTC greedy decoding.
    output_text_indexes = list(groupby(predictions_indexes))

    # Remove grouper objects.
    output_text_indexes, _ = np.transpose(output_text_indexes, (1, 0))

    # Remove blank symbols.
    output_text_indexes = output_text_indexes[output_text_indexes != 0]

    # Assign letters to indexes from the output array.
    output_text = [letters[letter_index] for letter_index in output_text_indexes]
    result_text = "".join(output_text)
    return result_text


# def question_answering(sources, example_question):
#     context = load_context(sources)
#     if len(context) == 0:
#         print("Error: Empty context or outside paragraphs")
#         return
#
#     answer, score = get_best_answer(question=example_question, context=context)
#
#     result_qa = {
#         "Question": example_question,
#         "Answer": answer,
#         "Score": f"{score:.2f}"
#     }
#     return result_qa


# def translate_eg(sentence: str) -> str:
#     """
#     Tokenize the sentence using the downloaded tokenizer and run the model,
#     whose output is decoded into a human readable string.
#
#     :param sentence: a string containing the phrase to be translated
#     :return: the translated string
#     """
#     # Remove leading and trailing white spaces
#     sentence = sentence.strip()
#     assert len(sentence) > 0
#     tokens = src_tokenizer.encode(sentence).ids
#     # Transform the tokenized sentence into the model's input format
#     tokens = [src_tokenizer.token_to_id('<s>')] + \
#              tokens + [src_tokenizer.token_to_id('</s>')]
#     pad_length = max_tokens - len(tokens)
#
#     # If the sentence size is less than the maximum allowed tokens,
#     # fill the remaining tokens with '<pad>'.
#     if pad_length > 0:
#         tokens = tokens + [src_tokenizer.token_to_id('<pad>')] * pad_length
#     assert len(tokens) == max_tokens, "input sentence is too long"
#     encoded_sentence = np.array(tokens).reshape(1, -1)
#
#     # Perform inference
#     enc_translated = compiled_model({input_name: encoded_sentence})
#     output_key = compiled_model.output(output_name)
#     enc_translated = enc_translated[output_key][0]
#
#     # Decode the sentence
#     sentence = tgt_tokenizer.decode(enc_translated)
#
#     # Remove <pad> tokens, as well as '<s>' and '</s>' tokens which mark the
#     # beginning and ending of the sentence.
#     for s in ['</s>', '<s>', '<pad>']:
#         sentence = sentence.replace(s, '')
#
#     # Transform sentence into lower case and join words by a white space
#     sentence = sentence.lower().split()
#     sentence = " ".join(key for key, _ in itertools.groupby(sentence))
#     return sentence

# def gpt_2(text):
#     input_ids = tokenize(text, tokenizer)
#     output_ids = generate_sequence(input_ids, eos_token_id, compiled_model_gpt2, output_keys_gpt2)
#     S = " "
#     # Convert IDs to words and make the sentence from it
#     for i in output_ids[0]:
#         S += tokenizer.convert_tokens_to_string(tokenizer._convert_id_to_token(i))
#     return S


if __name__ == "__main__":
    # img = "data/animal_classify/test.jpg"
    # result = cv_animal_classify(img)
    # print(result)
    # img = "data/road_segmentation/empty_road_mapillary.jpg"
    # result = cv_road_segmentation(img)
    # plt.imshow(result)
    # plt.show()
    # img = "data/ocr/intel_rnb.jpg"
    # result, annotations = cv_ocr(img)
    # plt.imshow(result)
    # plt.show()
    # print(annotations)
    # source_text = "Intel was founded in Mountain View, California, " \
    #               "in 1968 by Gordon E. Moore, a chemist, and Robert Noyce, " \
    #               "a physicist and co-inventor of the integrated circuit."
    # result = analyze_entities(source_text)
    # print(result)
    img = "data/handwritten_ocr/handwritten_chinese_test.jpg"
    result = handwritten_ocr(img, "CH")
    print(result)
    # sources = ["https://en.wikipedia.org/wiki/OpenVINO"]
    # question = "What does OpenVINO refers to?"
    # result = question_answering(sources, question)
    # print(result)
    # sentence = "My name is openvino"
    # result = translate_eg(sentence)
    # print(result)
    # text = "VR Games is a type of virtual reality technology"
    # result = gpt_2(text)
    # print(result)

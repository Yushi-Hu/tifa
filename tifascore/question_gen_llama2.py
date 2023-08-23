import torch
import json
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

categories = ['object', 'human', 'animal', 'food', 'activity', 'attribute',
              'counting', 'color', 'material', 'spatial', 'location', 'shape', 'other']


def get_llama2_pipeline(model_name="tifa-benchmark/llama2_tifa_question_generation"):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    return pipeline


# format dataset. Follow LLaMA 2 style
def create_qg_prompt(caption):

    INTRO_BLURB = """Given an image description, generate one or two multiple-choice questions that verifies if the image description is correct.
Classify each concept into a type (object, human, animal, food, activity, attribute, counting, color, material, spatial, location, shape, other), and then generate a question for each type.
"""

    formated_prompt = f"<s>[INST] <<SYS>>\n{INTRO_BLURB}\n<</SYS>>\n\n"
    formated_prompt += f"Description: {caption} [/INST] Entities:"
    return formated_prompt


def llama2_completion(pipeline, caption):

    prompt = create_qg_prompt(caption)

    sequences = pipeline(
        prompt, do_sample=False, num_beams=5, num_return_sequences=1, max_length=512)

    output = sequences[0]['generated_text'][len(prompt):]
    output = output.split('\n\n')[0]
    return output


def parse_resp(resp):
    resp = resp.split('\n')

    question_instances = []

    this_entity = None
    this_type = None
    this_question = None
    this_choices = None
    this_answer = None

    for line_number in range(6, len(resp)):
        line = resp[line_number]
        if line.startswith('About '):
            whole_line = line[len('About '):-1]
            this_entity = whole_line.split(' (')[0]
            this_type = whole_line.split(' (')[1].split(')')[0]

        elif line.startswith('Q: '):
            this_question = line[3:]
        elif line.startswith('Choices: '):
            this_choices = line[9:].split(', ')
        elif line.startswith('A: '):
            this_answer = line[3:]

            if this_entity and this_question and this_choices:
                question_instances.append(
                    (this_entity, this_question, this_choices, this_answer, this_type))
            this_question = None
            this_choices = None
            this_answer = None

    return question_instances


def get_llama2_question_and_answers(pipeline, caption):
    resp = llama2_completion(pipeline, caption)
    question_instances = parse_resp(resp)

    this_caption_qas = []

    for question_instance in question_instances:
        this_qa = {}
        this_qa['caption'] = caption
        this_qa['element'] = question_instance[0]
        this_qa['question'] = question_instance[1]
        this_qa['choices'] = question_instance[2]
        this_qa['answer'] = question_instance[3]
        this_qa['element_type'] = question_instance[4]

        if question_instance[4] not in categories:
            continue

        if this_qa['element_type'] in ['animal', 'human']:
            this_qa['element_type'] = 'animal/human'

        this_caption_qas.append(this_qa)

    return this_caption_qas


if __name__ == "__main__":
    pipeline = get_llama2_pipeline(
        "tifa-benchmark/llama2_tifa_question_generation")

    test_caption_1 = "a blue rabbit and a red plane"
    print(get_llama2_question_and_answers(pipeline, test_caption_1))
    print('-------------------'*10)

    test_caption_2 = "a painting of a fox in the style of starry night"
    print(get_llama2_question_and_answers(pipeline, test_caption_2))
    print('-------------------'*10)

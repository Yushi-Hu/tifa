import json, os
from collections import defaultdict
from tqdm import tqdm
from .vqa_models import VQAModel
from statistics import mean, stdev

def tifa_score_benchmark(vqa_model_name, question_answer_path, id2img_path):
    
    # load the questions and answers
    with open(question_answer_path) as f:
        question_answer_pairs = json.load(f)
    
    # load the image paths    
    with open(id2img_path) as f:
        caption_id2img_fn = json.load(f)
    id2img_parent_path = os.path.dirname(id2img_path)
    
    # load the VQA model
    vqa_model = VQAModel(vqa_model_name)
    
    tifa_statistics = {"scores": defaultdict(list), 
                    "type_scores": defaultdict(list)}
    question_logs = defaultdict(dict)
    
    for question_answer_pair in tqdm(question_answer_pairs):
        # get text input id
        caption_id = question_answer_pair['id']
        
        # read the question, choices, and answers
        if question_answer_pair['question'] not in question_logs[caption_id]:
            question_logs[caption_id][question_answer_pair['question']] = question_answer_pair
        choices=question_answer_pair['choices']
            
        # get the corresponding generated image path
        if str(caption_id) not in caption_id2img_fn:
            raise KeyError(f"Image corresponding to {caption_id} not found in {id2img_path}!")
        img_fn = caption_id2img_fn[str(caption_id)]
        if not os.path.isabs(img_fn):
            img_fn = os.path.join(id2img_parent_path, img_fn)
            
        # get VQA answer
        vqa_answer = vqa_model.multiple_choice_vqa(img_fn, question_answer_pair['question'], choices=choices)
         
        free_form_answer, mc_answer = vqa_answer["free_form_answer"], vqa_answer["multiple_choice_answer"]
        question_logs[caption_id][question_answer_pair['question']]['free_form_vqa'] = free_form_answer
        question_logs[caption_id][question_answer_pair['question']]['multiple_choice_vqa'] = mc_answer
        
        # compute multiple choice score
        score = int(mc_answer == question_answer_pair['answer'])
        question_logs[caption_id][question_answer_pair['question']]['scores'] = score
        
        # statistics of the scores
        tifa_statistics['scores'][caption_id].append(score)
        tifa_statistics['type_scores'][question_answer_pair['element_type']].append(score)
        
    question_logs = dict(question_logs)
    result_dict = {}
    
    # compute the average score
    averaged_scores = [mean(scores) for caption_id, scores in tifa_statistics["scores"].items()]
    
    result_dict = {"tifa_average": mean(averaged_scores),
                   "tifa_stdev": stdev(averaged_scores),
                   "accuracy_by_type": {type_: mean(scores) for type_, scores in tifa_statistics["type_scores"].items()}
                   } 
    
    print(f"Average TIFA is {result_dict['tifa_average']}")
    
    # record the details of each question  
    result_dict["question_details"] = question_logs
    
     # record the scores averaged by captions
    result_dict["caption_scores"] = {caption_id: mean(scores) for caption_id, scores in tifa_statistics["scores"].items()}
    
    return result_dict



def tifa_score_single(vqa_model, question_answer_pairs, img_path):
    
    tifa_scores = []
    question_logs = {}
    
    for question_answer_pair in tqdm(question_answer_pairs):
        
        # read the question, choices, and answers
        if question_answer_pair['question'] not in question_logs:
            question_logs[question_answer_pair['question']] = question_answer_pair
        choices=question_answer_pair['choices']
            
        # get VQA answer
        vqa_answer = vqa_model.multiple_choice_vqa(img_path, question_answer_pair['question'], choices=choices)
         
        free_form_answer, mc_answer = vqa_answer["free_form_answer"], vqa_answer["multiple_choice_answer"]
        question_logs[question_answer_pair['question']]['free_form_vqa'] = free_form_answer
        question_logs[question_answer_pair['question']]['multiple_choice_vqa'] = mc_answer
        
        # compute multiple choice score
        score = int(mc_answer == question_answer_pair['answer'])
        question_logs[question_answer_pair['question']]['scores'] = score
        
        # statistics of the scores
        tifa_scores.append(score)
        
    question_logs = dict(question_logs)
    result_dict = {}
    
    # compute the average score
    averaged_scores = mean(tifa_scores)
    
    result_dict = {"tifa_score": averaged_scores} 
    
    # record the details of each question  
    result_dict["question_details"] = question_logs
    
    return result_dict
import json, os
from tqdm import tqdm
from collections import defaultdict
from word2number import w2n

FREE_FORM_THRESHOLD = 0.6

def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP) != 0 else 0
        recall = TP / float(TP+FN) if (TP+FN) != 0 else 0
        F1 = 2 * precision * recall / \
            float(precision + recall) if (precision+recall) != 0 else 0
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return float(F1)

def filter_question_and_answers(qa_model, caption_qas):
    
    filtered_question_instances = []
    
    question_set = set()
    
    for question_instance in caption_qas:
        question = question_instance['question']
        caption = question_instance['caption']
        choices = question_instance['choices']
        
        # avoid duplicate questions
        if question in question_set:
            continue
        else:
            question_set.add(question)
            
        # validate the anwer
        qa_answer = qa_model.mcqa(question, caption, choices=choices)
        if qa_answer != question_instance['answer']:
            continue
        
        # validate free form answer
        if question_instance['answer'] not in ['yes', 'no']:
            free_form_answer = qa_model.qa(question, caption).strip()
            
            gpt3_answer = question_instance['answer']
            if gpt3_answer.isnumeric():
                try:
                    free_form_answer = str(w2n.word_to_num(free_form_answer))
                except:
                    pass
            
            if compute_prf(gpt3_answer.split(), free_form_answer.split()) <= FREE_FORM_THRESHOLD:
                continue
            
        filtered_question_instances.append(question_instance)
        
    return filtered_question_instances
        
        
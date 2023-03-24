from transformers import AutoProcessor, AutoModelForCausalLM, BlipForQuestionAnswering, ViltForQuestionAnswering
import torch
from PIL import Image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.multi_modal import OfaPreprocessor

from promptcap import PromptCap_VQA
from lavis.models import load_model_and_preprocess

from .mc_sbert import SBERTModel

vqa_models = {
    "git-large": ("GIT", "microsoft/git-large-vqav2"),
    "git-base": ("GIT", "microsoft/git-base-vqav2"),
    "blip-large": ("BLIP", "Salesforce/blip-vqa-capfilt-large"),
    "blip-base": ("BLIP", "Salesforce/blip-vqa-base"),
    "vilt": ("VILT", "dandelin/vilt-b32-finetuned-vqa"),
    "promptcap-t5large": ("PromptCap", "vqascore/promptcap-coco-vqa"),
    "ofa-large": ("OFA", "damo/ofa_visual-question-answering_pretrain_large_en"),
    "mplug-large": ("MPLUG", "damo/mplug_visual-question-answering_coco_large_en"),
    "blip2-flant5xl": ("BLIP2", "pretrain_flant5xl"),
}


class GIT:
    def __init__(self, ckpt="microsoft/git-large-vqav2"):
        # ckpts: "microsoft/git-large-vqav2", "microsoft/git-base-vqav2"
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = AutoModelForCausalLM.from_pretrained(ckpt)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def vqa(self, image, question):
        image = Image.open(image).convert('RGB')
        # prepare image
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        # prepare question
        input_ids = self.processor(text=question, add_special_tokens=False).input_ids
        input_ids = [self.processor.tokenizer.cls_token_id] + input_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        input_ids = input_ids.to(self.device)
    
        input_len = input_ids.shape[-1]
    
        generated_ids = self.model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
        generated_ids = generated_ids[..., input_len:]
    
        generated_answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
   
        return generated_answer[0]
    
    
class BLIP:
    def __init__(self, ckpt="Salesforce/blip-vqa-capfilt-large"):
        # ckpts: "Salesforce/blip-vqa-capfilt-large", "Salesforce/blip-vqa-base"
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = BlipForQuestionAnswering.from_pretrained(ckpt)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def vqa(self, image, question):
        image = Image.open(image).convert('RGB')
        # prepare image + question
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(**inputs, max_length=50)
        generated_answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
       
        return generated_answer[0]


class VILT:
    def __init__(self, ckpt="dandelin/vilt-b32-finetuned-vqa"):
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = ViltForQuestionAnswering.from_pretrained(ckpt)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def vqa(self, image, question):
        image = Image.open(image).convert('RGB')
        # prepare image + question

        encoding = self.processor(images=image, text=question, truncation=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoding)

        predicted_class_idx = outputs.logits.argmax(-1).item()
    
        return self.model.config.id2label[predicted_class_idx]
    

class OFA:
    def __init__(self, ckpt='damo/ofa_visual-question-answering_pretrain_large_en'):
        from modelscope.outputs import OutputKeys
        from modelscope.preprocessors.multi_modal import OfaPreprocessor
        preprocessor = OfaPreprocessor(model_dir=ckpt)
        self.ofa_pipe = pipeline(
            Tasks.visual_question_answering,
            model=ckpt,
            preprocessor=preprocessor)
        
    def vqa(self, image, question):
        question = question.lower()
        input = {'image': image, 'text': question}
        result = self.ofa_pipe(input)
        return result[OutputKeys.TEXT][0]


class PromptCap:
    def __init__(self, ckpt='vqascore/promptcap-coco-vqa'):
        self.vqa_model = PromptCap_VQA(promptcap_model=ckpt, qa_model="allenai/unifiedqa-v2-t5-large-1363200")

        if torch.cuda.is_available():
            self.vqa_model.cuda()
        
    def vqa(self, image, question):
        return self.vqa_model.vqa(question, image)
    
class MPLUG:
    def __init__(self, ckpt='damo/mplug_visual-question-answering_coco_large_en'):
        self.pipeline_vqa = pipeline(Tasks.visual_question_answering, model=ckpt)
        
    def vqa(self, image, question):
        input_vqa = {'image': image, 'question': question}
        result = self.pipeline_vqa(input_vqa)
        return result['text']
    
class BLIP2:
    def __init__(self, ckpt='pretrain_flant5xl'):
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type=ckpt, is_eval=True, device=self.device)
        
    def vqa(self, image, question, choices = []):
        image = Image.open(image).convert('RGB')
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        
        if len(choices) == 0:
            answer = self.model.generate({"image": image, "prompt": f"Answer the question. Question: {question} Answer:"})
        else:
            answer = self.model.generate({"image": image, "prompt": f"Answer the multiple-choice question. Question: {question} Choices: {', '.join(choices)} Answer:"})
        return answer[0]


class VQAModel:
    def __init__(self, model_name='mplug-large'):
        print(f"Loading {model_name}...")
        self.model_name = model_name
        class_name, ckpt = vqa_models[model_name]
        self.model = eval(class_name)(ckpt)
        print(f"Finish loading {model_name}")
        
        # use SBERT to find the closest choice
        self.sbert_model = SBERTModel("sentence-transformers/all-mpnet-base-v2")
        
    def vqa(self, image, question, choices=[]):
        with torch.no_grad():
            if (len(choices) != 0) and (self.model_name.startswith("blip2")):
                return self.model.vqa(image, question, choices)
            else:
                return self.model.vqa(image, question)
            
    def multiple_choice_vqa(self, image, question, choices):
        
        # Get VQA model's answer
        free_form_answer = self.vqa(image, question, choices)
        
        # Limit the answer to the choices
        multiple_choice_answer = free_form_answer
        if free_form_answer not in choices:
            multiple_choice_answer = self.sbert_model.multiple_choice(free_form_answer, choices)
        return {"free_form_answer": free_form_answer, "multiple_choice_answer": multiple_choice_answer}
    

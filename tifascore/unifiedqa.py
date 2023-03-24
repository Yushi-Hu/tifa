from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


class UnifiedQAModel:
    
    def __init__(self, model_name = "allenai/unifiedqa-v2-t5-large-1363200"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model.cuda()
            self.model.eval()
            
    def run_model(self, input_string, **generator_args):
        with torch.no_grad():
            input_ids = self.tokenizer.encode(input_string, return_tensors="pt")
            res = self.model.generate(input_ids.to(self.model.device), max_new_tokens=30, **generator_args)
            return self.tokenizer.batch_decode(res, skip_special_tokens=True)
        
    def qa(self, question, context):
        answer = self.run_model(f"{question} \n {context}")[0]
        # normalize the answer
        return ''.join(c for c in answer if c.isalnum() or c.isspace()).strip().lower()
    
    def mcqa(self, question, context, choices=["yes", "no"]):
        
        choice_text = ""
        
        if len(choices) > 0:
            choice_text = ""
            headings = ["(A)", "(B)", "(C)", "(D)"]
            for i, choice in enumerate(choices):
                if i < len(headings):
                    choice_text += f"{headings[i]} {choice} "    
        
        return self.run_model(f"{question} \n {context} \n {choice_text}")[0]


if __name__ == "__main__":
    model = UnifiedQAModel("allenai/unifiedqa-v2-t5-large-1363200")
    print(model.qa("how many cats are there", "Two cats laying on a red couch."))
    print(model.mcqa("how many cats are there", "Two cats laying on a red couch.", choices=["one", "two", "three", "four"]))
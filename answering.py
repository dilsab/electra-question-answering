import torch

from utils import load_model_tokenizer, last_save_path


class ElectraQuestionAnswering:
    def __init__(self, model_tokenizer_save_path):
        self.model, self.tokenizer = load_model_tokenizer(last_save_path(model_tokenizer_save_path))
        self.model.eval()
        
    def predict_answers(self, context, questions):
        answers = []
        for question in questions:
            answers.append(self.predict_answer(context, question))
            
        return answers

    def predict_answer(self, context, question):
        inputs = self.tokenizer(question, context, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].tolist()[0]

        output = self.model(**inputs)

        answer_start = torch.argmax(output.start_logits)
        answer_end = torch.argmax(output.end_logits) + 1

        return self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

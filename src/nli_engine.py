from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class NLIBaselineEngine:
    def __init__(self, model_name="cross-encoder/nli-deberta-v3-small"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        
        # DeBERTa v3 NLI mapping: 0 -> Contradiction, 1 -> Entailment, 2 -> Neutral
        self.id2label = {0: "Contradiction", 1: "Entailment", 2: "Neutral"}

    def check_entailment(self, premise, hypothesis):
        """ Evaluates if the premise supports the hypothesis (claim). """
        inputs = self.tokenizer(
            premise, hypothesis, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_id].item()
            
        return self.id2label[pred_id], confidence
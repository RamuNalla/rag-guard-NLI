from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class NLIBaselineEngine:
    def __init__(self, model_name="cross-encoder/nli-deberta-v3-small"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        
        # DeBERTa v3 NLI mapping: 0 -> Contradiction, 1 -> Entailment, 2 -> Neutral
        self.id2label = {0: "Contradiction", 1: "Entailment", 2: "Neutral"}


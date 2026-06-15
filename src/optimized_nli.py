from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import time

class OptimizedNLIEngine:
    def __init__(self, model_path="models/nli_onnx_quantized"):
        print(f"Loading ONNX Quantized model from {model_path}...")
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load the ONNX model (Executes on CPU by default via onnxruntime)
        self.model = ORTModelForSequenceClassification.from_pretrained(model_path)
        
        # DeBERTa v3 NLI mapping: 0 -> Contradiction, 1 -> Entailment, 2 -> Neutral
        self.id2label = {0: "Contradiction", 1: "Entailment", 2: "Neutral"}

    def check_entailment(self, premise, hypothesis):
        """ Evaluates entailment using the quantized ONNX graph. """
        # Tokenize input
        inputs = self.tokenizer(
            premise, hypothesis, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        # ONNX Runtime inference
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()
            
        return self.id2label[pred_id], confidence
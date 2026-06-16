from src.claim_extraction import ClaimExtractor
from src.semantic_router import SemanticRouter
from src.nli_engine import NLIBaselineEngine
from src.optimized_nli import OptimizedNLIEngine
import json

class RAGGuardPipeline:
    def __init__(self):
        print("Initializing RAGGuard Pipeline...")
        self.extractor = ClaimExtractor()
        self.router = SemanticRouter()
        self.nli_engine = OptimizedNLIEngine()
        print("Pipeline ready! Using Apple MPS / CPU.\n")

    def evaluate(self, generated_text, source_text):
        # 1. Break texts into atomic components
        claims = self.extractor.extract(generated_text)
        source_sentences = self.extractor.extract(source_text)
        
        results = []
        for claim in claims:
            # 2. Route claim to the best context
            premise = self.router.get_top_k_context(claim, source_sentences, top_k=2)
            
            # 3. Predict hallucination via NLI
            label, confidence = self.nli_engine.check_entailment(premise=premise, hypothesis=claim)
            
            results.append({
                "claim": claim,
                "matched_context": premise,
                "nli_label": label,
                "confidence": round(confidence, 4)
            })
            
        return results

if __name__ == "__main__":

    source_document = """
    Acme Corp released its Q3 earnings report yesterday. The company revenue grew by 20% compared to last year. 
    The current CEO, Jane Doe, stated that the growth was primarily driven by their new AI software division. 
    However, the hardware division saw a 5% decline in sales.
    """
    
    # Notice the subtle hallucination (200% instead of 20%) and the fabricated fact (John Smith)
    llm_generated_response = """
    Acme Corp's revenue grew by 200% in Q3. The CEO, Jane Doe, attributed this to the AI software division. 
    The hardware division was led by John Smith.
    """
    
    pipeline = RAGGuardPipeline()
    report = pipeline.evaluate(
        generated_text=llm_generated_response, 
        source_text=source_document
    )
    
    print(json.dumps(report, indent=2))
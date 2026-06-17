from typing import Any, Optional, Dict
from langchain_core.evaluation import StringEvaluator
from src.core import RAGGuardPipeline

class RAGGuardLangChainEvaluator(StringEvaluator):
    def __init__(self):
        self.pipeline = RAGGuardPipeline()

    @property
    def requires_input(self) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        return True

    @property
    def evaluation_name(self) -> str:
        return "nli_hallucination_check"

    def _evaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not reference:
            raise ValueError("Source context (reference) is required for RAG evaluation.")
        
        # 'prediction' = generated LLM text. 'reference' = source context.
        report = self.pipeline.evaluate(generated_text=prediction, source_text=reference)
        
        # Calculate a Faithfulness Score: (1 - Contradiction Rate)
        total = len(report['details'])
        if total == 0:
            score = 1.0
        else:
            contradictions = sum(1 for d in report['details'] if d['nli_label'] == 'Contradiction')
            score = (total - contradictions) / total

        return {
            "score": score,  # 0.0 to 1.0
            "value": "PASS" if score >= 0.8 else "FAIL",
            "reasoning": report['details']
        }
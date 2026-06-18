from typing import Any, Optional, Sequence
from llama_index.core.evaluation import BaseEvaluator, EvaluationResult
from src.core import RAGGuardPipeline

class RAGGuardLlamaIndexEvaluator(BaseEvaluator):
    def __init__(self):
        self.pipeline = RAGGuardPipeline()

    def _get_prompts(self):
        return {}

    def _update_prompts(self, prompts):
        pass

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        
        if not response or not contexts:
            raise ValueError("Both 'response' and 'contexts' must be provided.")
        
        # Combine contexts into a single source text string
        source_text = " ".join(contexts)
        report = self.pipeline.evaluate(generated_text=response, source_text=source_text)
        
        # Calculate Faithfulness Score
        total = len(report['details'])
        score = 1.0
        if total > 0:
            contradictions = sum(1 for d in report['details'] if d['nli_label'] == 'Contradiction')
            score = (total - contradictions) / total
            
        return EvaluationResult(
            query=query,
            contexts=contexts,
            response=response,
            passing=bool(score >= 0.8),
            score=score,
            feedback=str(report['details'])
        )
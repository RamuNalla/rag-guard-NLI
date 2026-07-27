from typing import Any, Optional, Dict
from src.core import RAGGuardPipeline

# ---------------------------------------------------------------------------
# Optional LangChain integration
# When langchain_core is installed the evaluator plugs in as a proper
# StringEvaluator. When it is not installed it works as a standalone class
# with the same interface so tests and scripts always work.
# ---------------------------------------------------------------------------
try:
    from langchain_core.evaluation import StringEvaluator as _Base
except ImportError:
    class _Base:  # type: ignore
        """Fallback base when langchain_core is not installed."""
        pass

class RAGGuardLangChainEvaluator(_Base):
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
        # pipeline.evaluate() returns a plain list of claim-result dicts.
        details = self.pipeline.evaluate(generated_text=prediction, source_text=reference)
        
        # Calculate a Faithfulness Score: (1 - Contradiction Rate)
        total = len(details)
        if total == 0:
            score = 1.0
        else:
            contradictions = sum(1 for d in details if d['nli_label'] == 'Contradiction')
            score = (total - contradictions) / total

        return {
            "score": score,  # 0.0 to 1.0
            "value": "PASS" if score >= 0.8 else "FAIL",
            "reasoning": details
        }

    def evaluate_strings(self, *, prediction: str, reference: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Public alias matching the LangChain StringEvaluator interface."""
        return self._evaluate_strings(prediction=prediction, reference=reference, **kwargs)
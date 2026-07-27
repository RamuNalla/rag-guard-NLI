import json
from src.integrations.langchain_evaluator import RAGGuardLangChainEvaluator

print("=" * 55)
print("  Testing LangChain Integration")
print("=" * 55)

evaluator = RAGGuardLangChainEvaluator()

result = evaluator.evaluate_strings(
    prediction="The Eiffel Tower is in London.",
    reference="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
)

print(json.dumps(result, indent=2, default=str))
print("\nExpected → score: 0.0 | value: FAIL (Paris ≠ London)")

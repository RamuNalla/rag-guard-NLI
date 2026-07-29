# RAG-Guard NLI

**Enterprise-grade hallucination detection for RAG pipelines using ONNX-optimized Natural Language Inference.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)
![ONNX Runtime](https://img.shields.io/badge/ONNX-Optimized-orange)
![Tests](https://img.shields.io/badge/Tests-11%20Passing-brightgreen)
![CI](https://github.com/ramunalla/rag-guard-NLI/actions/workflows/ci.yml/badge.svg)

---

## Overview

LLMs hallucinate. Evaluating RAG outputs for hallucinations with "LLM-as-a-judge" is slow, expensive, and itself prone to hallucination.

**RAG-Guard NLI** solves the RAG hallucination problem by decomposing generated responses into atomic claims and verifying each one against source documents using a **Cross-Encoder** model.

By combining **INT8 Dynamic Quantization** with **ONNX Runtime**, the pipeline that I built in this project delivers a **3.19× latency speedup** and a **70% reduction in model size** — running at ~6 ms per claim on a standard CPU with zero accuracy loss.

---

## Features

- **Atomic Claim Extraction** — `spaCy` sentencizer breaks monolithic responses into individually verifiable facts
- **Semantic Routing** — `sentence-transformers` (MiniLM) finds the most relevant source sentences per claim before NLI inference
- **High-Throughput Inference** — ONNX INT8-quantized DeBERTa runs locally faster than most cloud API round-trips
- **Explainability** — pinpoints exactly *which* claim is hallucinated and *why* (`Contradiction` vs `Neutral`)
- **Ecosystem Integrations** — plug-and-play wrappers for **LangChain** and **LlamaIndex**

---

## Benchmarks

### Hallucination Detection on HaluEval (1,000 QA samples)

Evaluated against the [pminervini/HaluEval](https://huggingface.co/datasets/pminervini/HaluEval) QA benchmark with human-verified labels.

| Metric | Baseline Rule | Optimised Rule |
|--------|:---:|:---:|
| **Accuracy** | 61.5% | 56.9% |
| **Precision** | 71.0% | 55.2% |
| **Recall** | 43.9% | **91.2%** |
| **F1-Score** | 54.2% | **68.8%** |

> The optimised rule (flag when Entailment confidence < 0.99) trades precision for recall — catching 91% of real hallucinations.  
> Choose the baseline rule when false positives are costly; choose the optimised rule in production RAG where missing a hallucination is the bigger risk.

![Confusion Matrix](assets/benchmark_results_improved.png)

### Model Optimisation: PyTorch fp32 vs ONNX INT8

| | Baseline (PyTorch fp32) | Optimized (ONNX INT8) | Delta |
|---|:---:|:---:|:---:|
| **Disk Size** | 541 MB | 165 MB | **−70%** |
| **Avg Latency** | 18.6 ms | 5.8 ms | **3.19× faster** |
| **p95 Latency** | 28.0 ms | 6.7 ms | |
| **Accuracy** | 66.7% | 66.7% | **0% degradation** |

![Model Comparison](assets/nli_comparison.png)

---

## Quick Start

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/RamuNalla/rag-guard-NLI.git
cd rag-guard-NLI

docker build -t rag-guard .
docker run -p 8000:8000 rag-guard
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

### Option 2 — Local

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn api.server:app --reload --port 8000
```

### Test the API

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "generated_text": "The Eiffel Tower is in London.",
    "source_text": "The Eiffel Tower is in Paris, France."
  }'
```

```json
{
  "execution_time_seconds": 0.047,
  "total_claims_checked": 1,
  "details": [
    {
      "claim": "The Eiffel Tower is in London.",
      "nli_label": "Contradiction",
      "confidence": 0.9996
    }
  ]
}
```

---

## Framework Integrations

### LangChain

```python
from src.integrations.langchain_evaluator import RAGGuardLangChainEvaluator

evaluator = RAGGuardLangChainEvaluator()
result = evaluator.evaluate_strings(
    prediction="The company revenue grew by 200%.",
    reference="The company revenue grew by 20% in Q3."
)
print(result)
# {'score': 0.0, 'value': 'FAIL', 'reasoning': [...]}
```

### LlamaIndex

```python
from src.integrations.llamaindex_evaluator import RAGGuardLlamaIndexEvaluator

evaluator = RAGGuardLlamaIndexEvaluator()
result = await evaluator.aevaluate(
    response="The Eiffel Tower is in London.",
    contexts=["The Eiffel Tower is in Paris, France."]
)
print(result.passing)  # False
```

---

## Architecture

```
Generated Text + Source Document
          │
          ▼
   ClaimExtractor (spaCy sentencizer)
   → ["Claim 1", "Claim 2", ...]
          │
          ▼ (per claim)
   SemanticRouter (MiniLM sentence-transformers)
   → Top-k most relevant source sentences
          │
          ▼
   NLIEngine (DeBERTa-v3 INT8 ONNX)
   → Entailment / Contradiction / Neutral + confidence
          │
          ▼
   [ { claim, matched_context, nli_label, confidence }, ... ]
```

| Layer | Technology |
|-------|-----------|
| Claim Extraction | `spaCy` sentencizer |
| Semantic Routing | `sentence-transformers` (all-MiniLM-L6-v2) |
| NLI Engine | `cross-encoder/nli-deberta-v3-small` → ONNX INT8 |
| API | `FastAPI` + `Pydantic` v2 |
| Serving | `uvicorn` / Docker |
| CI/CD | GitHub Actions |

---

## Running Tests

```bash
pytest tests/ -v
```

```
tests/test_api.py::test_evaluate_endpoint_status         PASSED
tests/test_api.py::test_evaluate_endpoint_schema         PASSED
tests/test_api.py::test_evaluate_catches_contradiction   PASSED
tests/test_api.py::test_evaluate_faithful_response       PASSED
tests/test_api.py::test_evaluate_missing_field_returns_422 PASSED
tests/test_api.py::test_evaluate_empty_strings           PASSED
tests/test_extraction.py::test_basic_extraction          PASSED
tests/test_extraction.py::test_filters_short_fragments   PASSED
tests/test_extraction.py::test_empty_string_returns_empty_list PASSED
tests/test_extraction.py::test_single_sentence           PASSED
tests/test_extraction.py::test_multisentence_paragraph   PASSED

11 passed in 16s
```

---

## 📁 Project Structure

```
rag-guard-NLI/
├── api/
│   ├── server.py           # FastAPI app & /evaluate endpoint
│   └── schemas.py          # Pydantic request/response models
├── src/
│   ├── core.py             # RAGGuardPipeline orchestrator
│   ├── claim_extraction.py # spaCy sentence splitter
│   ├── semantic_router.py  # MiniLM semantic search
│   ├── nli_engine.py       # Baseline PyTorch NLI engine
│   ├── optimized_nli.py    # ONNX INT8 NLI engine
│   └── integrations/
│       ├── langchain_evaluator.py
│       └── llamaindex_evaluator.py
├── models/
│   └── nli_onnx_quantized/ # Quantized DeBERTa ONNX model
├── benchmarks/
│   └── benchmark_evaluation.ipynb
├── tests/
│   ├── test_api.py
│   └── test_extraction.py
├── assets/                 # README images
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## 📄 License

MIT
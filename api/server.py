import time
from fastapi import FastAPI, HTTPException
from api.schemas import RAGRequest, HallucinationReport, ClaimResult
from src.core import RAGGuardPipeline

app = FastAPI(
    title="RAG-Guard NLI",
    description="Enterprise-grade Hallucination Detection for RAG Systems.",
    version="1.0.0"
)

# Load the model globally so it stays in RAM between requests
print("Starting server and loading ONNX models...")
pipeline = RAGGuardPipeline()

@app.post("/evaluate", response_model=HallucinationReport)
def evaluate_rag(request: RAGRequest):
    try:
        t0 = time.perf_counter()
        raw_results = pipeline.evaluate(
            generated_text=request.generated_text,
            source_text=request.source_text
        )
        elapsed = time.perf_counter() - t0

        return HallucinationReport(
            execution_time_seconds=round(elapsed, 4),
            total_claims_checked=len(raw_results),
            details=[
                ClaimResult(
                    claim=r["claim"],
                    nli_label=r["nli_label"],
                    confidence=r["confidence"]
                )
                for r in raw_results
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
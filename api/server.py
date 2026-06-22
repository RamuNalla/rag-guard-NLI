from fastapi import FastAPI, HTTPException
from api.schemas import RAGRequest, HallucinationReport
from src.core import RAGGuardPipeline

app = FastAPI(
    title="RAG-Guard NLI",
    description="Enterprise-grade Hallucination Detection for RAG Systems.",
    version="1.0.0"
)

# Load the model globally so it stays in RAM
print("Starting server and loading ONNX models...")
pipeline = RAGGuardPipeline()

@app.post("/evaluate", response_model=HallucinationReport)
def evaluate_rag(request: RAGRequest):
    try:
        # Run our optimized logic
        report = pipeline.evaluate(
            generated_text=request.generated_text,
            source_text=request.source_text
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
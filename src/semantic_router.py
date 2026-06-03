from sentence_transformers import SentenceTransformer, util
import torch

class SemanticRouter:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Hardware acceleration: Use Apple Silicon GPU (mps) if available, else CPU
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def get_top_k_context(self, claim, source_sentences, top_k=2):
        """Finds the most relevant source sentences for a given claim."""
        if not source_sentences:
            return ""
        
        # Encode the claim and source texts into dense vectors
        claim_emb = self.model.encode(claim, convert_to_tensor=True, device=self.device)
        source_embs = self.model.encode(source_sentences, convert_to_tensor=True, device=self.device)
        
        # Compute cosine similarities
        cosine_scores = util.cos_sim(claim_emb, source_embs)[0]
        
        # Retrieve the top_k scoring sentences
        top_k = min(top_k, len(source_sentences))
        top_results = torch.topk(cosine_scores, k=top_k)
        
        # Reconstruct the supportive context premise
        best_sentences = [source_sentences[idx] for idx in top_results.indices]
        return " ".join(best_sentences)
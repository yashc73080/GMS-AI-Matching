from typing import Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Semantic scorer
class SemanticScorer:
    def __init__(self, mode: str = "off", model_name: str = "all-MiniLM-L6-v2"):
        self.mode = mode
        self.model_name = model_name
        self._embedder = None
        if self.mode == "embed":
            try:
                self._embedder = SentenceTransformer(self.model_name)
                self.cosine_similarity = cosine_similarity
            except Exception as e:
                raise RuntimeError("Embed mode requires sentence-transformers and scikit-learn. "
                                   "Install with: pip install sentence-transformers scikit-learn")

    @staticmethod
    def _norm(s: Optional[str]) -> str:
        return "" if s is None else str(s).strip()

    def _embed_cosine01(self, a: str, b: str) -> float:
        a = self._norm(a); b = self._norm(b)
        if not a or not b:
            return 0.0
        embs = self._embedder.encode([a, b], show_progress_bar=False, normalize_embeddings=True)
        sim = float(self.cosine_similarity([embs[0]], [embs[1]])[0][0])  # [-1,1]
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))

    def goals(self, mentee_goal: str, mentor_goal: str) -> float:
        if self.mode != "embed":
            return 0.0
        return self._embed_cosine01(mentee_goal, mentor_goal)

    def interests(self, mentee_interests: str, mentor_interests: str) -> float:
        if self.mode != "embed":
            return 0.0
        return self._embed_cosine01(mentee_interests, mentor_interests)
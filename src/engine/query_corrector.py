import re
from typing import Dict, Any

class CorrectiveRAGQueryTransformer:
    """
    Self-Corrective RAG (CRAG) and Hypothetical Document Expansion (HyDE)
    engine that rewrites ambiguous or out-of-distribution queries into expanded retrieval keys.
    """
    ACRONYM_MAP = {
        "stemi": "ST-Elevation Myocardial Infarction Acute Coronary Syndrome 12-lead ECG door-to-balloon PCI",
        "acs": "Acute Coronary Syndrome STEMI NSTEMI troponin aspirin nitroglycerin",
        "qsofa": "quick Sequential Organ Failure Assessment Sepsis respiratory rate altered mentation blood pressure",
        "fhir": "Fast Healthcare Interoperability Resources HL7 R5 RESTful JSON schemas",
        "rrf": "Reciprocal Rank Fusion hybrid search dense BM25 ranking algorithm",
        "rag": "Retrieval-Augmented Generation dense embeddings BM25 reranker context"
    }

    def evaluate_and_transform(self, query: str, top_confidence: float, threshold: float = 0.70) -> Dict[str, Any]:
        needs_correction = top_confidence < threshold
        expanded_query = query
        transformed = False
        reasons = []

        if needs_correction:
            reasons.append(f"Top confidence score ({top_confidence}) is below threshold ({threshold})")
            tokens = [t.lower() for t in re.findall(r'\b\w+\b', query)]
            
            # Check for known medical / AI acronym expansions
            expansion_additions = []
            for t in tokens:
                if t in self.ACRONYM_MAP:
                    expansion_additions.append(self.ACRONYM_MAP[t])
            
            if expansion_additions:
                expanded_query = f"{query} {' '.join(expansion_additions)}"
                transformed = True
                reasons.append("Applied domain acronym HyDE expansion")
            else:
                expanded_query = f"{query} clinical guidelines diagnostic criteria protocol overview"
                transformed = True
                reasons.append("Applied generic query enrichment fallback")

        return {
            "original_query": query,
            "transformed_query": expanded_query,
            "was_transformed": transformed,
            "reasons": reasons
        }

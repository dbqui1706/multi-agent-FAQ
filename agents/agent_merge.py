from typing import Any
import logging
from google import genai
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

def run(
    faqs: list[dict],
    client: genai.Client, model_name: str,
    dedup_threshold: float
) -> list[dict]:
    """
    Eliminate duplicate meaning question in FAQs by using Gemini text-embedding-004.
    """
    if len(faqs) < 2:
        return faqs

    questions = [f["question"] for f in faqs]
    
    try:
        # batch questions to avoid API limit
        vecs = []
        for i in range(0, len(questions), 50):
            batch_questions = questions[i:i+50]
            batch_embeddings = client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=batch_questions,
            ).embeddings
            vecs.extend([e.values for e in batch_embeddings])
            time.sleep(0.3)
        
        # calculate cosine similarity
        vecs = np.array(vecs)
        sim_matrix = cosine_similarity(vecs)

        # find duplicates
        return greedy_dedup(faqs, sim_matrix, dedup_threshold)
    except Exception as e:
        logger.error("[Merge] Dedup failed: %s — returning all FAQs unfiltered", e)
        return faqs

def greedy_dedup(faqs: list[dict], sim_matrix: np.ndarray, threshold: float) -> list[int]:
    """
    Greedily find duplicates based on cosine similarity.
    """
    n = len(faqs)
    remove = set()
    for i in range(n):
        if i in remove:
            continue
        for j in range(i + 1, n):
            if j in remove:
                continue
            if sim_matrix[i, j] > threshold:
                score_i = faqs[i].get("review_score", 0)
                score_j = faqs[j].get("review_score", 0)
                remove.add(j if score_i >= score_j else i)

    return [f for idx, f in enumerate(faqs) if idx not in remove]
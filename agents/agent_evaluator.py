from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
import numpy as np  
logger = logging.getLogger(__name__)


SEMATIC_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

# ─────────────────────────────────────────────────────────────────────────────
# LLM Prompt
# ─────────────────────────────────────────────────────────────────────────────

_EVAL_SYSTEM_PROMPT = """\
Bạn là chuyên gia đánh giá chất lượng FAQ về quy chế đào tạo.

Cho một cặp Q&A và context gốc, hãy đánh giá ĐÚNG 3 chỉ số sau và trả về JSON (không markdown):

{
  "faithfulness": <số thực trong [0.0, 1.0]>,
  "answer_relevance": <số nguyên trong [1, 2, 3, 4, 5]>,
  "context_independence": <0 hoặc 1>
}

=== ĐỊNH NGHĨA ===

faithfulness [0.0 – 1.0]:
  Đo mức độ câu trả lời trung thực với context gốc.
  1.0 = mọi thông tin trong câu trả lời đều có căn cứ trong context, không bịa thêm.
  0.5 = có 1–2 chi tiết nhỏ không có trong context.
  0.0 = câu trả lời mâu thuẫn với context hoặc hoàn toàn bịa đặt.

answer_relevance [1 – 5]:
  Đo mức độ câu trả lời thực sự trả lời câu hỏi được đặt ra.
  5 = trả lời trực tiếp, đầy đủ, súc tích.
  4 = trả lời đúng nhưng có chi tiết thừa hoặc thiếu phụ.
  3 = trả lời một phần, lạc đề phụ.
  2 = hầu như không trả lời câu hỏi.
  1 = hoàn toàn không liên quan đến câu hỏi.

context_independence {0, 1}:
  Câu hỏi có thể hiểu được mà không cần đọc tài liệu gốc?
  1 = câu hỏi tự thân, nêu rõ chủ thể và ngữ cảnh.
  0 = câu hỏi dùng "nó", "điều này", "quy định trên", thiếu chủ thể, hoặc chỉ có nghĩa khi biết trước context.\
"""


def _build_eval_prompt(item: dict[str, Any]) -> str:
    return f"""{_EVAL_SYSTEM_PROMPT}

---
CÂU HỎI: {item.get("question", "")}

CÂU TRẢ LỜI: {item.get("answer", "")}

CONTEXT GỐC:
{item.get("context", "(không có context)")}
---

Kết quả đánh giá (chỉ JSON):"""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(
    faqs: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    client: Any,          # genai.Client
    model_name: str,
    llm_call_delay: float = 0.5,
) -> dict[str, Any]:
    """
    Đánh giá bộ FAQ theo 6 metrics.

    Args:
        faqs:         List FAQ dicts (output của pipeline).
        chunks:       List chunk dicts (output của agent_chunker).
        client:       genai.Client đã được khởi tạo.
        model_name:   Tên model Gemini.
        llm_call_delay: Giây delay giữa các LLM call.

    Returns:
        EvalReport dict với "summary" và "per_item".
    """
    logger.info("[Evaluator] Evaluating %d FAQs against %d chunks...", len(faqs), len(chunks))

    # ── A. Per-item LLM evaluation ────────────────────────────────────────────
    per_item_results: list[dict[str, Any]] = []

    for i, item in enumerate(faqs, 1):
        logger.info(
            "[Evaluator] [%d/%d] %s",
            i, len(faqs),
            item.get("question", "")[:70],
        )
        scores = _eval_item_llm(item, client, model_name)
        per_item_results.append({
            "id"                  : item.get("id", f"item_{i}"),
            "question"            : item.get("question", ""),
            "faithfulness"        : scores["faithfulness"],
            "answer_relevance"    : scores["answer_relevance"],
            "context_independence": scores["context_independence"],
        })
        if llm_call_delay > 0:
            time.sleep(llm_call_delay)

    # ── B. Corpus-level metrics (local) ───────────────────────────────────────
    diversity   = _compute_diversity_sematic(faqs)
    coverage    = _compute_context_coverage(faqs, chunks)
    retrieval   = _compute_retrieval_effectiveness(faqs, chunks)

    # ── C. Aggregate summary ──────────────────────────────────────────────────
    n = len(per_item_results)
    faith_avg   = round(sum(r["faithfulness"]         for r in per_item_results) / n, 3) if n else 0.0
    rel_avg     = round(sum(r["answer_relevance"]     for r in per_item_results) / n, 2) if n else 0.0
    ctx_rate    = round(sum(r["context_independence"] for r in per_item_results) / n, 3) if n else 0.0

    report: dict[str, Any] = {
        "summary": {
            "total_faqs"                : n,
            "faithfulness_avg"          : faith_avg,
            "answer_relevance_avg"      : rel_avg,
            "context_independence_rate" : ctx_rate,
            "diversity_score"           : round(diversity, 3),
            "context_coverage"          : round(coverage, 3),
            "retrieval_effectiveness"   : round(retrieval, 3),
        },
        "per_item": per_item_results,
    }

    logger.info("[Evaluator] Done. Summary: %s", report["summary"])
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Per-item LLM scoring
# ─────────────────────────────────────────────────────────────────────────────

def _eval_item_llm(
    item: dict[str, Any],
    client: Any,
    model_name: str,
) -> dict[str, Any]:
    """Gọi LLM để đánh giá 3 per-item metrics. Trả về dict với fallback an toàn."""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=_build_eval_prompt(item),
        )
        return _parse_eval_json(response.text.strip())
    except Exception as exc:
        logger.warning("[Evaluator] LLM error on %s: %s", item.get("id"), exc)
        return _fallback_scores()


def _parse_eval_json(raw: str) -> dict[str, Any]:
    """Parse JSON từ LLM, với fallback nếu lỗi."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    start   = cleaned.find("{")
    end     = cleaned.rfind("}")
    if start == -1 or end == -1:
        logger.warning("[Evaluator] No JSON found in LLM response.")
        return _fallback_scores()
    try:
        data = json.loads(cleaned[start: end + 1])
        # Clamp và type-cast để đảm bảo đúng range
        faith = float(data.get("faithfulness", 0.5))
        faith = max(0.0, min(1.0, faith))

        rel = int(round(float(data.get("answer_relevance", 3))))
        rel = max(1, min(5, rel))

        ctx_indep = int(data.get("context_independence", 1))
        ctx_indep = 1 if ctx_indep else 0

        return {
            "faithfulness"        : round(faith, 2),
            "answer_relevance"    : rel,
            "context_independence": ctx_indep,
        }
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("[Evaluator] JSON parse error: %s", exc)
        return _fallback_scores()


def _fallback_scores() -> dict[str, Any]:
    return {"faithfulness": 0.5, "answer_relevance": 3, "context_independence": 1}


# ─────────────────────────────────────────────────────────────────────────────
# Corpus-level local metrics
# ─────────────────────────────────────────────────────────────────────────────
def _compute_diversity_sematic(faqs: list[dict[str, Any]]) -> float:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    questions = [f.get("question", "") for f in faqs if f.get("question")]

    model = SentenceTransformer(SEMATIC_MODEL_NAME)
    embeddings = model.encode(
        questions,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    sim_matrix = embeddings @ embeddings.T
    n = len(questions)
    # upper triangle
    upper_indices = np.triu_indices(n, k=1)
    avg_similarity = sim_matrix[upper_indices].mean()

    diversity = 1.0 - avg_similarity
    return diversity


def _compute_diversity(faqs: list[dict[str, Any]]) -> float:
    """
    Diversity Score [0, 1]:
    Đo độ đa dạng ngữ nghĩa toàn bộ câu hỏi bằng average pairwise cosine distance.
    Score cao = câu hỏi khác nhau nhiều về nội dung.
    """
    questions = [f.get("question", "") for f in faqs if f.get("question")]
    if len(questions) < 2:
        return 1.0  # Chỉ 1 câu hỏi → tự nó là "đa dạng"

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True
        ).fit_transform(questions)
        sim_matrix = cosine_similarity(vec)

        n = len(questions)
        # upper-triangle 
        upper_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
        avg_similarity = (
            sum(sim_matrix[i, j] for i, j in upper_indices) / (len(upper_indices))
        )
        # Diversity = 1 - similarity mean
        return float(1.0 - avg_similarity)

    except ImportError:
        logger.warning("[Evaluator] sklearn not installed. Diversity = 0.0")
        return 0.0


def _compute_context_coverage(
    faqs: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> float:
    """
    Context Coverage / Recall [0, 1]:
    Tỉ lệ chunks gốc có ít nhất 1 FAQ được sinh ra từ đó.
    chunk_ids có FAQ / tổng chunk_ids.
    """
    if not chunks:
        return 0.0

    all_chunk_ids   = {c["id"] for c in chunks}
    covered_ids     = {f.get("source_chunk_id") for f in faqs if f.get("source_chunk_id")}
    covered_valid   = covered_ids & all_chunk_ids

    return len(covered_valid) / len(all_chunk_ids)


def _compute_retrieval_effectiveness(
    faqs: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> float:
    """
    Retrieval Effectiveness [0, 1]:
    Tỉ lệ FAQ có context snippet tìm lại được trong chunks gốc (context recall).
    Kiểm tra bằng cách tìm 40 ký tự đầu của context trong nội dung chunk tương ứng.
    """
    if not faqs:
        return 0.0

    # Build lookup: chunk_id → content
    chunk_map: dict[str, str] = {c["id"]: c.get("content", "") for c in chunks}

    hit   = 0
    total = 0

    for faq in faqs:
        ctx     = (faq.get("context") or "").strip()
        src_id  = faq.get("source_chunk_id", "")

        if not ctx:
            continue  # Bỏ qua FAQ không có context

        total += 1
        snippet = ctx[:60].strip()

        # Tìm snippet trong chunk nguồn
        if src_id in chunk_map and snippet in chunk_map[src_id]:
            hit += 1
            continue

        # Fallback: tìm trong tất cả chunks (context có thể span nhiều chunks)
        if any(snippet in c_text for c_text in chunk_map.values()):
            hit += 1

    return (hit / total) if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Console report helper
# ─────────────────────────────────────────────────────────────────────────────

def print_report(report: dict[str, Any]) -> None:
    """In báo cáo đánh giá ra console."""
    s = report.get("summary", {})
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"{'EVALUATION REPORT':^60}")
    print(sep)
    print(f"  Total FAQs                  : {s.get('total_faqs')}")
    print(f"  --- LLM-based (per-item) ---")
    print(f"  Faithfulness avg            : {s.get('faithfulness_avg'):.3f}  [0–1]")
    print(f"  Answer Relevance avg        : {s.get('answer_relevance_avg'):.2f}  [1–5]")
    print(f"  Context Independence rate   : {s.get('context_independence_rate'):.1%}")
    print(f"  --- Local (corpus-level) ---")
    print(f"  Diversity Score             : {s.get('diversity_score'):.3f}  [0–1]")
    print(f"  Context Coverage            : {s.get('context_coverage'):.1%}")
    print(f"  Retrieval Effectiveness     : {s.get('retrieval_effectiveness'):.1%}")
    print(sep)

    # Per-item table (top worst faithfulness)
    items = report.get("per_item", [])
    poor  = sorted(items, key=lambda x: x["faithfulness"])[:5]
    if poor:
        print(f"\n  Top 5 items có Faithfulness thấp nhất:")
        print(f"  {'ID':<25} {'Faith':>6} {'Rel':>4} {'CtxInd':>7}")
        print("  " + "-" * 46)
        for r in poor:
            print(
                f"  {r['id'][:25]:<25} "
                f"{r['faithfulness']:>6.2f} "
                f"{r['answer_relevance']:>4} "
                f"{r['context_independence']:>7}"
            )
    print()

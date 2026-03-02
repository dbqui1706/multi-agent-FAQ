import logging
import time
from typing import Any

from google import genai
from utils import parse_json_array, parse_json_object   

logger = logging.getLogger(__name__)

SEMATIC_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

_SYSTEM_PROMPT = """Bạn là chuyên gia tạo FAQ từ Quy Chế Đào Tạo Trình Độ Thạc Sĩ (Trường ĐH CNTT, ĐHQG-HCM).

Bạn được cung cấp nội dung từ NHIỀU ĐIỀU KHOẢN liên quan.
Hãy tạo 3-4 câu hỏi YÊU CẦU TỔNG HỢP thông tin từ ít nhất 2 điều khoản.

QUY TẮC:
- Câu hỏi phải TỰ THÂN (nêu đủ chủ thể, không cần đọc tài liệu gốc)
- Câu trả lời phải tổng hợp thông tin từ nhiều điều khoản, ghi rõ "Theo Điều X... Điều Y..."
- KHÔNG tạo câu hỏi chỉ liên quan đến 1 điều khoản đơn lẻ
- Câu hỏi phải có giá trị thực tế với người dùng

OUTPUT: JSON array (không markdown):
[
  {
    "question"     : "Câu hỏi tổng hợp, tự thân",
    "answer"       : "Câu trả lời tổng hợp từ nhiều Điều, có tham chiếu cụ thể",
    "context"      : "Đoạn nguyên văn từ tài liệu (tổng hợp từ các Điều liên quan, tối đa 400 từ)",
    "persona"      : "student|lecturer|admin",
    "question_type": "comparison|procedure|condition"
  }
]"""

_RETRY_SYSTEM_PROMPT = """Viết lại FAQ multi-chunk dưới đây dựa trên gợi ý sửa.
Chỉ trả về JSON object (không array, không markdown):
{"question":"...","answer":"...","context":"...","question_type":"...","persona":"..."}"""


def group_chunks(
    chunks    : list[dict[str, Any]],
    n_groups  : int = 9,
    min_size  : int = 2,
) -> list[list[dict[str, Any]]]:
    """
    Group chunks by topic using KMeans clustering on sentence embeddings.
    Returns a list of groups, each containing 2-4 chunks.
    Only keeps groups with at least min_size chunks.
    """
    valid_chunks = [c for c in chunks if c.get("term") and len(c.get("content","")) > 80]
    if len(valid_chunks) < min_size * 2:
        return []

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        import numpy as np

        contents = [c["content"] for c in valid_chunks]

        model = SentenceTransformer(SEMATIC_MODEL_NAME)
        embeddings = model.encode(contents, convert_to_numpy=True, normalize_embeddings=True)

        labels = KMeans(
            n_clusters=n_groups,
            init="k-means++",
            n_init=20,
            random_state=42,
        ).fit_predict(embeddings)

        groups = {}
        for chunk, label in zip(valid_chunks, labels):
            groups.setdefault(int(label), []).append(chunk)

        # Keep at most 4 chunks per group
        return [g[:4] for g in groups.values() if len(g) >= min_size]

    except Exception as e:
        logger.warning("[Multichunk] Grouping error: %s", e)
        return []


def run(
    chunk_groups: list[list[dict[str, Any]]],
    client      : genai.Client,
    model_name  : str,
) -> list[dict[str, Any]]:
    """Sinh cross-chunk FAQ cho từng group."""
    logger.info("[Multichunk] %d groups", len(chunk_groups))
    all_faqs: list[dict[str, Any]] = []

    for i, group in enumerate(chunk_groups, 1):
        if len(group) < 2:
            continue

        logger.info("[Multichunk] [%d/%d] group: %s",
                    i, len(chunk_groups),
                    " + ".join(c.get("term","")[:30] for c in group))

        prompt = _build_prompt(group)
        try:
            resp = client.models.generate_content(model=model_name, contents=prompt)
            qa_pairs = parse_json_array(resp.text.strip())
        except Exception as e:
            logger.warning("[Multichunk] group %d error: %s", i, e)
            qa_pairs = []

        for j, pair in enumerate(qa_pairs, 1):
            q = pair.get("question", "").strip()
            a = pair.get("answer", "").strip()
            if not q or not a:
                continue

            chunk_ids = [c["id"] for c in group]
            sources   = " + ".join(
                c.get("term", c.get("chapter", ""))[:40] for c in group
            )
            all_faqs.append({
                "id"              : f"multichunk_{i:03d}_qa_{j:02d}",
                "source"          : sources,
                "source_chunk_id" : chunk_ids[0],
                "source_chunk_ids": chunk_ids,
                "page_numbers"    : sorted({p for c in group for p in c.get("page_numbers", [])}),
                "question"        : q,
                "answer"          : a,
                "context"         : pair.get("context", ""),
                "persona"         : pair.get("persona", "student"),
                "question_type"   : pair.get("question_type", "comparison"),
                "source_agent"    : "multichunk",
                "is_retry"        : False,
            })

        time.sleep(0.5)

    logger.info("[Multichunk] Generated %d cross-chunk FAQs", len(all_faqs))
    return all_faqs


def rewrite(
    item      : dict[str, Any],
    client    : genai.Client,
    model_name: str,
) -> dict[str, Any] | None:
    """Viết lại 1 multichunk FAQ dựa trên improvement_hint."""
    hint = item.get("improvement_hint", "")
    if not hint:
        return None
    prompt = f"""{_RETRY_SYSTEM_PROMPT}

HINT SỬA: {hint}
CÂU HỎI GỐC: {item.get('question', '')}
CÂU TRẢ LỜI GỐC: {item.get('answer', '')}
CONTEXT: {item.get('context', '')}

JSON object:"""
    try:
        resp = client.models.generate_content(model=model_name, contents=prompt)
        obj  = parse_json_object(resp.text.strip())
        if obj and obj.get("question") and obj.get("answer"):
            updated = dict(item)
            updated.update({
                "question" : obj["question"],
                "answer"   : obj["answer"],
                "context"  : obj.get("context", item["context"]),
                "is_retry" : True,
            })
            return updated
    except Exception as e:
        logger.warning("[Multichunk] rewrite error: %s", e)
    return None


# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(group: list[dict[str, Any]]) -> str:
    blocks = []
    for chunk in group:
        term    = chunk.get("term") or chunk.get("chapter") or "N/A"
        chapter = chunk.get("chapter") or "N/A"
        blocks.append(f"=== {chapter} — {term} ===\n{chunk.get('content','')}")
    combined = "\n\n".join(blocks)
    return f"""{_SYSTEM_PROMPT}

---
{combined}
---
JSON array (3-4 câu hỏi tổng hợp):"""
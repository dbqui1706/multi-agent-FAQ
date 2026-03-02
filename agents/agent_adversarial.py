import json
import logging
import re
import time
from typing import Any

from google import genai
from utils import parse_json_array, parse_json_object

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Bạn là chuyên gia tạo câu hỏi kiểm tra độ robust cho FAQ Quy Chế Đào Tạo Thạc Sĩ.

Với đoạn văn bản được cung cấp, tạo 1-3 câu hỏi "khó" theo các loại sau (chỉ tạo những loại CÓ THỂ áp dụng):

NEGATIVE  — Câu hỏi mà câu trả lời đúng là "KHÔNG ĐƯỢC" hoặc "KHÔNG ĐỦ ĐIỀU KIỆN"
            Ví dụ: "Học viên có thể bảo vệ luận văn hoàn toàn trực tuyến không?"
            → Đáp: Không — phải có chủ tịch HĐ và thư ký có mặt tại Trường.

BOUNDARY  — Câu hỏi về trường hợp nằm đúng ngưỡng điều kiện
            Ví dụ: "Học viên đã học đúng 30 tín chỉ có được đăng ký song ngành chưa?"
            → Đáp: Phải hoàn thành TỐI THIỂU 30 tín chỉ → đúng 30 tín chỉ là đủ.

CONFLICT  — Câu hỏi khi 2 điều kiện có vẻ mâu thuẫn hoặc cần ưu tiên rõ ràng
            Ví dụ: "Học viên đang bị kỷ luật cảnh cáo có được chuyển sang CT ứng dụng không?"

QUY TẮC BẮT BUỘC:
- Câu hỏi phải TỰ THÂN (nêu đủ chủ thể, không dùng "nó", "điều này")
- Câu trả lời phải rõ ràng, có tham chiếu Điều/Khoản cụ thể
- Nếu không tìm thấy loại nào phù hợp → trả về array rỗng []

OUTPUT: JSON array (không markdown):
[
  {
    "question"     : "...",
    "answer"       : "...",
    "context"      : "Đoạn nguyên văn liên quan (tối đa 200 từ)",
    "question_type": "negative|boundary|conflict",
    "persona"      : "student|lecturer|admin"
  }
]"""

_RETRY_SYSTEM_PROMPT = """Viết lại FAQ adversarial dưới đây dựa trên gợi ý sửa.
Chỉ trả về JSON object (không array, không markdown):
{"question":"...","answer":"...","context":"...","question_type":"...","persona":"..."}"""

def run(
    chunks    : list[dict[str, Any]],
    client    : genai.Client,
    model_name: str,
) -> list[dict[str, Any]]:
    """Sinh adversarial FAQ cho toàn bộ chunks."""
    logger.info("[Adversarial] %d chunks", len(chunks))
    all_faqs: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks, 1):
        content   = chunk.get("content", "").strip()
        extracted = chunk.get("extracted_info", {})

        if len(content) < 100 or (isinstance(extracted, dict) and extracted.get("_skip")):
            continue

        logger.info("[Adversarial] [%d/%d] %s", i, len(chunks),
                    chunk.get("term", "")[:60])

        prompt = _build_prompt(chunk, content, extracted)
        try:
            resp     = client.models.generate_content(model=model_name, contents=prompt)
            qa_pairs = parse_json_array(resp.text.strip())
        except Exception as e:
            logger.warning("[Adversarial] chunk %s error: %s", chunk["id"], e)
            qa_pairs = []

        for j, pair in enumerate(qa_pairs, 1):
            q = pair.get("question", "").strip()
            a = pair.get("answer", "").strip()
            if not q or not a:
                continue
            chapter = chunk.get("chapter") or "N/A"
            term    = chunk.get("term") or chunk.get("chapter") or "N/A"
            all_faqs.append({
                "id"              : f"{chunk['id']}_adv_{j:02d}",
                "source"          : f"{chapter} — {term}",
                "source_chunk_id" : chunk["id"],
                "source_chunk_ids": [chunk["id"]],
                "page_numbers"    : chunk.get("page_numbers", []),
                "question"        : q,
                "answer"          : a,
                "context"         : pair.get("context", content[:500]),
                "persona"         : pair.get("persona", "student"),
                "question_type"   : pair.get("question_type", "negative"),
                "source_agent"    : "adversarial",
                "is_retry"        : False,
            })

        time.sleep(0.5)

    logger.info("[Adversarial] Generated %d adversarial FAQs", len(all_faqs))
    return all_faqs

def rewrite(
    item      : dict[str, Any],
    client    : genai.Client,
    model_name: str,
) -> dict[str, Any] | None:
    """Viết lại 1 adversarial FAQ dựa trên improvement_hint."""
    hint = item.get("improvement_hint", "")
    if not hint:
        return None
    prompt = f"""{_RETRY_SYSTEM_PROMPT}

HINT SỬA: {hint}
CÂU HỎI GỐC: {item.get('question', '')}
CÂU TRẢ LỜI GỐC: {item.get('answer', '')}
CONTEXT: {item.get('context', '')}
QUESTION_TYPE: {item.get('question_type', 'negative')}

JSON object:"""
    try:
        resp = client.models.generate_content(model=model_name, contents=prompt)
        obj  = parse_json_object(resp.text.strip())
        if obj and obj.get("question") and obj.get("answer"):
            updated = dict(item)
            updated.update({
                "question"     : obj["question"],
                "answer"       : obj["answer"],
                "context"      : obj.get("context", item["context"]),
                "question_type": obj.get("question_type", item["question_type"]),
                "is_retry"     : True,
            })
            return updated
    except Exception as e:
        logger.warning("[Adversarial] rewrite error: %s", e)
    return None

def _build_prompt(chunk: dict, content: str, extracted: dict) -> str:
    term    = chunk.get("term") or chunk.get("chapter") or "N/A"
    chapter = chunk.get("chapter") or "N/A"

    edge_block = ""
    if isinstance(extracted, dict) and extracted.get("edge_cases"):
        edge_block = "\nTRƯỜNG HỢP ĐẶC BIỆT (ưu tiên dùng cho BOUNDARY/CONFLICT):\n" + \
                     "\n".join(f"  - {e}" for e in extracted["edge_cases"])

    return f"""{_SYSTEM_PROMPT}

---
NGUỒN: {chapter} — {term}
NỘI DUNG:{edge_block}

{content}
---
JSON array ([] nếu không có loại nào phù hợp):"""
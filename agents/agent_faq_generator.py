"""
Agent: FAQ Generator — Taxonomy + Answer Planning CoT
======================================================
Cải tiến so với v1:
  [NEW-1] Taxonomy 6 loại câu hỏi bắt buộc
  [NEW-2] Answer Planning CoT — LLM lập outline trước khi viết
  [NEW-3] field source_agent = "faq_generator" để Reviewer routing
  [NEW-4] field is_retry + retry_hint để xử lý retry
"""

import json
import logging
import re
import time
from typing import Any

from google import genai
from utils import parse_json_array, parse_json_object

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Bạn là chuyên gia tạo FAQ từ Quy Chế Đào Tạo Trình Độ Thạc Sĩ (Trường ĐH CNTT, ĐHQG-HCM).

=== QUY TẮC BẮT BUỘC ===

[1] CÂU HỎI PHẢI TỰ THÂN (Context Independence):
    - Người đọc CHƯA ĐỌC tài liệu vẫn hiểu câu hỏi
    - TUYỆT ĐỐI KHÔNG dùng: "nó", "điều này", "quy định trên", "như đã nêu"
    - Luôn nêu rõ chủ thể cụ thể

[2] ANSWER PLANNING CoT — Trước khi viết câu trả lời, lập outline:
    • Kết luận trực tiếp (con số / điều kiện chính)
    • Điều kiện đi kèm (nếu có)
    • Ngoại lệ / trường hợp đặc biệt (nếu có)
    • Hậu quả nếu vi phạm (nếu relevant)
    • Tham chiếu (Điều X, Khoản Y)
    Sau đó viết answer đầy đủ từ outline — KHÔNG bỏ sót điểm nào.

[3] TAXONOMY — Mỗi chunk phải có ít nhất 3 loại câu hỏi trong số:
    - definition:   "... là gì / được hiểu như thế nào?"
    - condition:    "Điều kiện để ... là gì?"
    - procedure:    "Quy trình / thủ tục ... như thế nào?"
    - exception:    "Trường hợp nào ... được / không được?"
    - consequence:  "Điều gì xảy ra nếu ... / Vi phạm ... bị xử lý thế nào?"
    - comparison:   "Sự khác nhau giữa ... và ... là gì?"

[4] PERSONA — mỗi chunk nên có câu hỏi từ ít nhất 2 nhóm:
    - student: học viên đang học / sắp tốt nghiệp
    - lecturer: giảng viên / người hướng dẫn
    - admin: cán bộ quản lý

[5] TRÁNH TRÙNG LẶP với EXISTING_QUESTIONS

=== OUTPUT FORMAT ===
JSON array, KHÔNG có markdown:
[
  {
    "question"     : "Câu hỏi tự thân, rõ ràng",
    "answer"       : "Câu trả lời đầy đủ từ CoT outline",
    "context"      : "Đoạn nguyên văn từ tài liệu (tối đa 300 từ)",
    "persona"      : "student|lecturer|admin",
    "question_type": "definition|condition|procedure|exception|consequence|comparison"
  }
]"""

_RETRY_SYSTEM_PROMPT = """Bạn là chuyên gia chỉnh sửa FAQ về Quy Chế Đào Tạo Thạc Sĩ.
Viết lại câu hỏi và câu trả lời dưới đây dựa trên gợi ý sửa.
Giữ nguyên context và persona. Chỉ trả về JSON object (không array, không markdown):
{
  "question"     : "...",
  "answer"       : "...",
  "context"      : "...",
  "persona"      : "...",
  "question_type": "..."
}"""


def run(
    chunks         : list[dict[str, Any]],
    client         : genai.Client,
    model_name     : str,
    existing_questions: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Sinh FAQ từ danh sách chunks."""
    logger.info("[FAQ Generator] %d chunks", len(chunks))
    all_faqs: list[dict[str, Any]] = []
    tracked_questions = list(existing_questions or [])

    for i, chunk in enumerate(chunks, 1):
        content   = chunk.get("content", "").strip()
        extracted = chunk.get("extracted_info", {})

        if len(content) < 80 or (isinstance(extracted, dict) and extracted.get("_skip")):
            continue

        num_qa = _estimate_qa_count(content, extracted)
        recent = tracked_questions[-15:]
        prompt = _build_prompt(chunk, content, extracted, num_qa, recent)

        logger.info("[FAQ Generator] [%d/%d] %s", i, len(chunks),
                    chunk.get("term", chunk.get("chapter", ""))[:60])
        try:
            resp     = client.models.generate_content(model=model_name, contents=prompt)
            qa_pairs = parse_json_array(resp.text.strip())
        except Exception as e:
            logger.warning("[FAQ Generator] chunk %s error: %s", chunk["id"], e)
            qa_pairs = []

        for j, pair in enumerate(qa_pairs, 1):
            q = pair.get("question", "").strip()
            a = pair.get("answer", "").strip()
            if not q or not a:
                continue
            ctx = pair.get("context", "").strip() or _extract_best_context(q, content)
            item = _make_faq_item(chunk, j, q, a, ctx, pair)
            all_faqs.append(item)
            tracked_questions.append(q)

        time.sleep(0.5)

    logger.info("[FAQ Generator] Generated %d FAQs", len(all_faqs))
    return all_faqs


def rewrite(
    item      : dict[str, Any],
    client    : genai.Client,
    model_name: str,
) -> dict[str, Any] | None:
    """Viết lại 1 FAQ item dựa trên improvement_hint từ Reviewer."""
    hint = item.get("improvement_hint", "")
    if not hint:
        return None

    prompt = f"""{_RETRY_SYSTEM_PROMPT}

---
NGUỒN: {item.get('source', '')}
HINT SỬA: {hint}

CÂU HỎI GỐC: {item.get('question', '')}
CÂU TRẢ LỜI GỐC: {item.get('answer', '')}
CONTEXT: {item.get('context', '')}
PERSONA: {item.get('persona', 'student')}
---
JSON (chỉ object, không array):"""

    try:
        resp = client.models.generate_content(model=model_name, contents=prompt)
        obj  = parse_json_object(resp.text.strip())
        if obj and obj.get("question") and obj.get("answer"):
            updated = dict(item)
            updated.update({
                "question"     : obj["question"],
                "answer"       : obj["answer"],
                "context"      : obj.get("context", item["context"]),
                "persona"      : obj.get("persona", item["persona"]),
                "question_type": obj.get("question_type", item.get("question_type")),
                "is_retry"     : True,
            })
            return updated
    except Exception as e:
        logger.warning("[FAQ Generator] rewrite error: %s", e)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_faq_item(chunk, j, q, a, ctx, pair) -> dict:
    chapter = chunk.get("chapter") or "N/A"
    term    = chunk.get("term") or chunk.get("chapter") or "N/A"
    return {
        "id"             : f"{chunk['id']}_faq_{j:02d}",
        "source"         : f"{chapter} — {term}",
        "source_chunk_id": chunk["id"],
        "source_chunk_ids": [chunk["id"]],
        "page_numbers"   : chunk.get("page_numbers", []),
        "question"       : q,
        "answer"         : a,
        "context"        : ctx,
        "persona"        : pair.get("persona", "student"),
        "question_type"  : pair.get("question_type", "condition"),
        "source_agent"   : "faq_generator",
        "is_retry"       : False,
    }


def _build_prompt(chunk, content, extracted, num_qa, existing_questions) -> str:
    term    = chunk.get("term") or chunk.get("chapter") or "N/A"
    chapter = chunk.get("chapter") or "N/A"

    ext_block = ""
    if isinstance(extracted, dict) and not extracted.get("_skip"):
        parts = []
        if extracted.get("key_rules"):
            parts.append("• Quy định chính:\n  - " + "\n  - ".join(extracted["key_rules"]))
        if extracted.get("numbers_deadlines"):
            parts.append("• Số liệu/thời hạn:\n  - " + "\n  - ".join(extracted["numbers_deadlines"]))
        if extracted.get("edge_cases"):
            parts.append("• Trường hợp đặc biệt:\n  - " + "\n  - ".join(extracted["edge_cases"]))
        if extracted.get("suggested_questions"):
            parts.append("• Gợi ý câu hỏi:\n  - " + "\n  - ".join(extracted["suggested_questions"]))
        ext_block = "\n".join(parts)

    existing_block = ""
    if existing_questions:
        existing_block = "\n\nEXISTING_QUESTIONS (KHÔNG tạo câu hỏi tương đương):\n" + \
                         "\n".join(f"  - {q}" for q in existing_questions)

    return f"""{_SYSTEM_PROMPT}

---
NGUỒN: {chapter} — {term}

NỘI DUNG GỐC:
{content}

THÔNG TIN TRÍCH XUẤT:
{ext_block or "(không có)"}
{existing_block}
---
Tạo ĐÚNG {num_qa} cặp Q&A (JSON array):"""


def _estimate_qa_count(content: str, extracted: dict) -> int:
    base = 2
    if len(content) > 800:  base += 1
    if len(content) > 1500: base += 1
    if isinstance(extracted, dict):
        base += min(len(extracted.get("edge_cases", [])) // 2, 1)
        base += min(len(extracted.get("key_rules", [])) // 3, 1)
    return min(base, 5)


def _extract_best_context(question: str, content: str, max_words: int = 250) -> str:
    stop = {"là","có","và","của","trong","được","không","để","theo","về","tại",
            "với","các","một","hay","như","thế","nào","gì","khi","nếu","mà","thì"}
    q_words = set(re.findall(r"\b\w{3,}\b", question.lower())) - stop
    paragraphs = re.split(r"\n{2,}|\n(?=[a-z]\.|[0-9]+\.)", content)
    if len(paragraphs) == 1:
        lines = content.split("\n")
        paragraphs = ["\n".join(lines[max(0,i-1):i+2]) for i in range(0, len(lines), 2)]
    best_para, best_score = "", 0.0
    for para in paragraphs:
        if len(para.strip()) < 20:
            continue
        p_words = set(re.findall(r"\b\w{3,}\b", para.lower())) - stop
        if not p_words:
            continue
        score = len(q_words & p_words) / (len(q_words) + 0.1)
        if score > best_score:
            best_score, best_para = score, para.strip()
    if not best_para:
        best_para = "\n\n".join(paragraphs[:3])
    return " ".join(best_para.split()[:max_words])
"""
Agent 3: FAQ Generator — FIXED + IMPROVED VERSION
===================================================
Fix so với bản gốc:
  [FIX-1] Bug "source = chapter - chapter" → đúng "chapter - term"
  [FIX-2] Prompt enforce Context Independence (câu hỏi tự thân)
  [FIX-3] Diversity control: inject danh sách câu hỏi đã tạo để tránh trùng
  [FIX-4] context fallback: tìm đoạn văn liên quan nhất thay vì 500 ký tự đầu
  [FIX-5] Tận dụng extracted_info có cấu trúc từ Agent 2 (IMP-1)
  [IMP-1] Prompt phân tầng persona: sinh viên / giảng viên / cán bộ quản lý
  [IMP-2] Số lượng Q&A động theo độ dài/độ phức tạp chunk
"""

import logging
import json
import re
import time
from typing import Any

from google import genai

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — với đầy đủ ràng buộc chất lượng
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """Bạn là chuyên gia tạo FAQ từ Quy Chế Đào Tạo Trình Độ Thạc Sĩ (Trường ĐH CNTT, ĐHQG-HCM).

=== QUY TẮC BẮT BUỘC ===

[1] CÂU HỎI PHẢI TỰ THÂN (Context Independence):
    - Người đọc CHƯA ĐỌC tài liệu vẫn hiểu câu hỏi
    - TUYỆT ĐỐI KHÔNG dùng: "nó", "điều này", "quy định trên", "như đã nêu",
      "các điều kiện đó", "theo khoản trên", "trong trường hợp này"
    - Luôn nêu rõ chủ thể: thay "Học viên có được..." bằng "Học viên thạc sĩ tại Trường ĐH CNTT có được..."

[2] CÂU TRẢ LỜI PHẢI ĐẦY ĐỦ ĐIỀU KIỆN:
    - Nêu đủ các điều kiện tiên quyết, ngoại lệ, trường hợp đặc biệt
    - Giữ nguyên số liệu cụ thể (%, tín chỉ, ngày, điểm số)
    - KHÔNG suy diễn ngoài văn bản gốc

[3] CONTEXT PHẢI LÀ NGUYÊN VĂN:
    - Chép nguyên đoạn văn bản gốc liên quan trực tiếp, KHÔNG paraphrase
    - Tối đa 300 từ, ưu tiên đoạn chứa con số / điều kiện cụ thể

[4] TRÁNH TRÙNG LẶP:
    - Không tạo câu hỏi tương đương với các câu hỏi đã có trong danh sách EXISTING_QUESTIONS
    - Ưu tiên góc nhìn khác nhau: quy trình, điều kiện, thời hạn, ngoại lệ, hậu quả vi phạm

[5] PHÂN TẦNG PERSONA — mỗi chunk nên có câu hỏi từ ít nhất 2 nhóm:
    - 🎓 Học viên (đang học, sắp tốt nghiệp, vi phạm)
    - 👨‍🏫 Giảng viên / Người hướng dẫn
    - 🏫 Cán bộ quản lý (ĐVQL, ĐVCM)

=== ĐỊNH DẠNG OUTPUT ===
Trả về JSON array hợp lệ, KHÔNG có markdown:
[
  {
    "question": "Câu hỏi tự thân, rõ ràng, nêu đủ chủ thể",
    "answer": "Câu trả lời đầy đủ điều kiện, giữ nguyên số liệu",
    "context": "Đoạn nguyên văn từ tài liệu gốc (tối đa 300 từ)",
    "persona": "student|lecturer|admin"
  }
]"""


def run(
    chunks: list[dict[str, Any]],
    client: genai.Client,
    model_name: str,
) -> list[dict[str, Any]]:
    """
    Generate FAQ pairs cho mỗi chunk.

    Returns:
        List of FAQ dicts: id, source, source_chunk_id, page_numbers,
        question, answer, context, persona.
    """
    logger.info("[FAQ Generator] Generating FAQs for %d chunks...", len(chunks))
    all_faqs: list[dict[str, Any]] = []

    # [FIX-3] Diversity control: track câu hỏi đã tạo toàn pipeline
    existing_questions: list[str] = []

    for i, chunk in enumerate(chunks, 1):
        content   = chunk.get("content", "").strip()
        extracted = chunk.get("extracted_info", {})

        # Skip nếu quá ngắn hoặc bị đánh dấu skip
        if len(content) < 80 or (isinstance(extracted, dict) and extracted.get("_skip")):
            logger.debug("[FAQ Gen] Skipping chunk %s", chunk["id"])
            continue

        term    = chunk.get("term") or chunk.get("chapter") or "N/A"
        chapter = chunk.get("chapter") or "N/A"

        logger.info("[FAQ Generator] [%d/%d] %s", i, len(chunks), term)

        # [IMP-2] Số Q&A động theo độ dài content
        num_qa = _estimate_qa_count(content, extracted)

        # [FIX-3] Chỉ truyền 15 câu hỏi gần nhất để tránh prompt quá dài
        recent_questions = existing_questions[-15:] if existing_questions else []

        prompt = _build_prompt(chunk, content, extracted, num_qa, recent_questions)

        try:
            response  = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            raw       = response.text.strip()
            qa_pairs  = _parse_json_response(raw)
        except Exception as exc:
            logger.warning("[FAQ Gen] Error on chunk %s: %s", chunk["id"], exc)
            qa_pairs = []

        for j, pair in enumerate(qa_pairs, 1):
            question = pair.get("question", "").strip()
            answer   = pair.get("answer", "").strip()

            if not question or not answer:
                continue

            # [FIX-4] Tìm context snippet phù hợp nhất
            context = pair.get("context", "").strip()
            if not context or len(context) < 30:
                context = _extract_best_context(question, content)

            faq_item = {
                "id"             : f"{chunk['id']}_qa_{j:02d}",
                # [FIX-1] BUG FIX: chapter - term (không phải chapter - chapter)
                "source"         : f"{chapter} - {term}",
                "source_chunk_id": chunk["id"],
                "page_numbers"   : chunk.get("page_numbers", []),
                "question"       : question,
                "answer"         : answer,
                "context"        : context,
                "persona"        : pair.get("persona", "student"),
            }

            all_faqs.append(faq_item)
            existing_questions.append(question)

        time.sleep(0.6)

    logger.info("[FAQ Generator] Done. %d Q&A pairs generated.", len(all_faqs))
    return all_faqs


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(
    chunk: dict,
    content: str,
    extracted: dict,
    num_qa: int,
    existing_questions: list[str],
) -> str:
    """Xây dựng prompt đầy đủ cho FAQ Generator."""

    term    = chunk.get("term") or chunk.get("chapter") or "N/A"
    chapter = chunk.get("chapter") or "N/A"

    # Format extracted_info có cấu trúc từ Agent 2
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
            parts.append("• Gợi ý câu hỏi từ Extractor:\n  - " + "\n  - ".join(extracted["suggested_questions"]))
        ext_block = "\n".join(parts)

    # Format existing questions để tránh trùng
    existing_block = ""
    if existing_questions:
        existing_block = (
            "\n\nEXISTING_QUESTIONS (KHÔNG tạo câu hỏi tương đương):\n"
            + "\n".join(f"  - {q}" for q in existing_questions)
        )

    return f"""{_SYSTEM_PROMPT}

---
NGUỒN: {chapter} — {term}

NỘI DUNG GỐC:
{content}

THÔNG TIN TRÍCH XUẤT (ưu tiên tạo Q&A về các điểm này):
{ext_block if ext_block else "(không có)"}
{existing_block}
---

Tạo ĐÚNG {num_qa} cặp Q&A (JSON array):"""


def _estimate_qa_count(content: str, extracted: dict) -> int:
    """
    [IMP-2] Ước tính số Q&A phù hợp dựa trên:
    - Độ dài content
    - Số edge cases
    - Số quy định chính
    """
    base = 2
    if len(content) > 800:
        base += 1
    if len(content) > 1500:
        base += 1

    if isinstance(extracted, dict):
        edge_count = len(extracted.get("edge_cases", []))
        rule_count = len(extracted.get("key_rules", []))
        base += min(edge_count // 2, 1)
        base += min(rule_count // 3, 1)

    return min(base, 5)  # Tối đa 5 Q&A/chunk để duy trì quality


def _extract_best_context(question: str, content: str, max_words: int = 250) -> str:
    """
    [FIX-4] Tìm đoạn trong content liên quan nhất đến câu hỏi
    dựa trên keyword overlap, thay vì lấy 500 ký tự đầu.
    """
    # Lấy keywords từ câu hỏi (loại stop words cơ bản)
    stop = {"là", "có", "và", "của", "trong", "được", "không", "để",
            "theo", "về", "tại", "với", "các", "một", "hay", "như",
            "thế", "nào", "gì", "khi", "nếu", "mà", "thì", "bao",
            "nhiêu", "bằng", "hoặc", "đó", "này"}

    q_words = set(re.findall(r"\b\w{3,}\b", question.lower())) - stop

    # Chia content thành các đoạn (theo dòng trống hoặc khoản)
    paragraphs = re.split(r"\n{2,}|\n(?=[a-z]\.|[0-9]+\.)", content)
    if len(paragraphs) == 1:
        # Nếu không chia được, dùng sliding window 3 dòng
        lines = content.split("\n")
        paragraphs = [
            "\n".join(lines[max(0, i-1) : i+2])
            for i in range(0, len(lines), 2)
        ]

    # Tính overlap score cho mỗi đoạn
    best_para, best_score = "", 0.0
    for para in paragraphs:
        if len(para.strip()) < 20:
            continue
        p_words = set(re.findall(r"\b\w{3,}\b", para.lower())) - stop
        if not p_words:
            continue
        overlap = len(q_words & p_words)
        score   = overlap / (len(q_words) + 0.1)
        if score > best_score:
            best_score, best_para = score, para.strip()

    if not best_para:
        # Fallback: 3 đoạn đầu
        best_para = "\n\n".join(paragraphs[:3])

    # Cắt theo giới hạn từ
    words = best_para.split()
    return " ".join(words[:max_words])


def _parse_json_response(raw: str) -> list[dict]:
    """Extract và parse JSON array từ response của model."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    start   = cleaned.find("[")
    end     = cleaned.rfind("]")
    if start == -1 or end == -1:
        logger.warning("[FAQ Gen] No JSON array found.")
        return []
    try:
        data = json.loads(cleaned[start : end + 1])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as exc:
        logger.warning("[FAQ Gen] JSON parse error: %s", exc)
        return []

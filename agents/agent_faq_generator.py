"""
Agent 3: FAQ Generator
Takes enriched chunks (with extracted_info) and uses Gemini to generate
2-5 Q&A pairs per chunk. Each Q&A includes the relevant context passage
from the original chunk content.
"""

import logging
import json
import re
import time
from typing import Any

from google import genai

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Bạn là chuyên gia tạo câu hỏi và câu trả lời (FAQ) từ văn bản pháp lý giáo dục.

Nhiệm vụ: Dựa trên nội dung và thông tin đã trích xuất, tạo từ 2 đến 5 cặp Q&A hữu ích.

YÊU CẦU ĐỊNH DẠNG - Trả về một JSON array hợp lệ, mỗi item gồm:
{
  "question": "Câu hỏi rõ ràng, cụ thể",
  "answer": "Câu trả lời đầy đủ, chính xác dựa trên văn bản",
  "context": "Đoạn trích nguyên văn từ tài liệu liên quan trực tiếp đến Q&A này (tối đa 300 từ)"
}

QUAN TRỌNG:
- Câu hỏi phải thiết thực, là những điều sinh viên / giảng viên thực sự cần biết
- Câu trả lời phải dựa chính xác trên văn bản, không suy diễn
- Context phải là đoạn trích NGUYÊN VĂN từ tài liệu gốc (không paraphrase)
- CHỈ trả về JSON array, không thêm markdown hoặc text khác"""


def run(
    chunks: list[dict[str, Any]],
    client: genai.Client,
    model_name: str,
) -> list[dict[str, Any]]:
    """
    Generate FAQ pairs for each enriched chunk.

    Args:
        chunks:     List of enriched chunk dicts (with `extracted_info`).
        client:     Configured google.genai Client instance.
        model_name: Gemini model name string.

    Returns:
        List of FAQ item dicts: id, source, question, answer, context.
    """
    logger.info("[FAQ Generator] Generating FAQs for %d chunks...", len(chunks))
    all_faqs: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "").strip()
        extracted = chunk.get("extracted_info", "").strip()

        if len(content) < 80 or "(Nội dung quá ngắn" in extracted:
            logger.debug("[FAQ Gen] Skipping chunk %s (too short)", chunk["id"])
            continue

        logger.info(
            "[FAQ Generator] [%d/%d] Generating for: %s",
            i,
            len(chunks),
            chunk["term"],
        )

        prompt = f"""{_SYSTEM_PROMPT}

---
NGUỒN: {chunk['chapter']} - {chunk['term']}

NỘI DUNG GỐC:
{content}

THÔNG TIN ĐÃ TRÍCH XUẤT:
{extracted}
---

JSON array Q&A:"""

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            raw = response.text.strip()
            qa_pairs = _parse_json_response(raw)
        except Exception as exc:
            logger.warning("[FAQ Gen] Error on chunk %s: %s", chunk["id"], exc)
            qa_pairs = []

        for j, pair in enumerate(qa_pairs, 1):
            faq_item = {
                "id": f"{chunk['id']}_qa_{j:02d}",
                "source": f"{chunk['chapter']} - {chunk['chapter']}",
                "source_chunk_id": chunk["id"],
                "page_numbers": chunk.get("page_numbers", []),
                "question": pair.get("question", "").strip(),
                "answer": pair.get("answer", "").strip(),
                "context": pair.get("context", content[:500]).strip(),
            }
            # Skip malformed entries
            if faq_item["question"] and faq_item["answer"]:
                all_faqs.append(faq_item)

        time.sleep(0.6)

    logger.info("[FAQ Generator] Done. %d Q&A pairs generated.", len(all_faqs))
    return all_faqs


# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> list[dict]:
    """Extract and parse a JSON array from the model's raw text response."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Try to find JSON array boundaries
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1:
        logger.warning("[FAQ Gen] No JSON array found in response.")
        return []

    json_str = cleaned[start : end + 1]
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError as exc:
        logger.warning("[FAQ Gen] JSON parse error: %s", exc)

    return []

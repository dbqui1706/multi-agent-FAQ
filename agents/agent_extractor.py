"""
Agent 2: Extractor
For each chunk produced by the Chunker, calls the Gemini API to extract:
- Key regulations and conditions
- Important numbers/dates/requirements
- Main topics covered

Adds an `extracted_info` field to every chunk dict.
"""

import logging
import time
from typing import Any

from google import genai

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích văn bản pháp lý giáo dục Việt Nam.
Nhiệm vụ: Đọc đoạn văn bản từ Quy Chế Đào Tạo Trình Độ Thạc Sĩ và trích xuất các thông tin QUAN TRỌNG nhất.

Trả về dưới dạng văn bản thuần (không dùng markdown header), bao gồm:
1. Các quy định chính
2. Điều kiện, yêu cầu cụ thể (số liệu, thời hạn, tỷ lệ...)
3. Đối tượng áp dụng
4. Các điểm đặc biệt cần chú ý

Hãy ngắn gọn, súc tích, giữ nguyên thuật ngữ chuyên ngành."""


def run(
    chunks: list[dict[str, Any]],
    client: genai.Client,
    model_name: str,
) -> list[dict[str, Any]]:
    """
    Enrich each chunk with extracted key information.

    Args:
        chunks:     List of chunk dicts from Agent 1 (Chunker).
        client:     Configured google.genai Client instance.
        model_name: Gemini model name string.

    Returns:
        Same list of chunks, each with an added `extracted_info` field.
    """
    logger.info("[Extractor] Processing %d chunks...", len(chunks))
    enriched: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "").strip()

        # Skip very short / empty chunks
        if len(content) < 50:
            chunk = dict(chunk)
            chunk["extracted_info"] = "(Nội dung quá ngắn, bỏ qua)"
            enriched.append(chunk)
            logger.debug("[Extractor] Chunk %s skipped (too short)", chunk["id"])
            continue

        logger.info(
            "[Extractor] [%d/%d] Extracting: %s",
            i,
            len(chunks),
            chunk["term"],
        )

        prompt = f"""{_SYSTEM_PROMPT}

---
ĐOẠN VĂN BẢN ({chunk['chapter']} - {chunk['term']}):
{content}
---

Trích xuất thông tin quan trọng:"""

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            extracted = response.text.strip()
        except Exception as exc:
            logger.warning("[Extractor] Error on chunk %s: %s", chunk["id"], exc)
            extracted = f"(Lỗi khi trích xuất: {exc})"

        chunk = dict(chunk)
        chunk["extracted_info"] = extracted
        enriched.append(chunk)

        # Avoid hitting rate limits
        time.sleep(0.5)

    logger.info("[Extractor] Done. %d chunks enriched.", len(enriched))
    return enriched

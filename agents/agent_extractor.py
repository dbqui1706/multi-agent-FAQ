import json
import logging
import re
import time
from typing import Any

from google import genai

logger = logging.getLogger(__name__)

# "cross_references": [
#     "điểm X khoản Y Điều Z của Quy chế này",
#     "điểm A, B khoản C Điều D của Quy chế này",
#     "khoản X, Điều Y của Quy chế này",
#     ...
#   ],
#   "self_references": [
#     "điểm a, b khoản 1 Điều này",
#     "tại khoản X và khoản Y Điều này",
#     ...
#   ]

# Prompt 
_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích văn bản pháp lý giáo dục Việt Nam.
Nhiệm vụ: Đọc đoạn văn từ Quy Chế Đào Tạo Trình Độ Thạc Sĩ và trích xuất thông tin.

Trả về JSON hợp lệ với cấu trúc sau (không có markdown):
{
  "key_rules": ["Quy định chính 1", "Quy định chính 2"],
  "numbers_deadlines": ["15 ngày trước khi...", "tối đa 30 tín chỉ...", "5,5 điểm..."],
  "subjects": ["đối tượng áp dụng 1", "đối tượng áp dụng 2"],
  "edge_cases": ["trường hợp đặc biệt 1", "ngoại lệ 2"],
  "suggested_questions": [
    "Câu hỏi thiết thực mà sinh viên/GV có thể hỏi về đoạn này 1",
    "Câu hỏi thiết thực 2"
  ]
}

Giữ nguyên thuật ngữ chuyên ngành. Ngắn gọn, súc tích."""


def run(
    chunks: list[dict[str, Any]],
    client: genai.Client,
    model_name: str,
) -> list[dict[str, Any]]:
    """
    Enrich each chunk with structured extracted information.

    Returns:
        Same list with added `extracted_info` field (dict, not plain text).
    """
    logger.info("[Extractor] Processing %d chunks...", len(chunks))
    enriched: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "").strip()

        if len(content) < 50:
            chunk = dict(chunk)
            chunk["extracted_info"] = {
                "key_rules": [],
                "numbers_deadlines": [],
                "subjects": [],
                "edge_cases": [],
                "suggested_questions": [],
                "_skip": True,
            }
            enriched.append(chunk)
            logger.debug("[Extractor] Chunk %s skipped (too short)", chunk["id"])
            continue

        term    = chunk.get("term") or chunk.get("chapter") or "N/A"
        chapter = chunk.get("chapter") or "N/A"

        logger.info(
            "[Extractor] [%d/%d] Extracting: %s",
            i, len(chunks), term,
        )

        prompt = f"""Nguồn: {chapter} — {term}

Nội dung:
{content}

Trích xuất thông tin (JSON):"""

        try:
            response  = client.models.generate_content(
                model=model_name,
                contents=f"{_SYSTEM_PROMPT}\n\n{prompt}",
            )
            raw       = response.text.strip()
            extracted = _parse_json(raw)
        except Exception as exc:
            logger.warning("[Extractor] Error on chunk %s: %s", chunk["id"], exc)
            extracted = _empty_extraction(error=str(exc))

        chunk = dict(chunk)
        chunk["extracted_info"] = extracted
        enriched.append(chunk)

        time.sleep(0.5)

    logger.info("[Extractor] Done. %d chunks enriched.", len(enriched))
    return enriched


# ─────────────────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """[IMP-4] Parse JSON với fallback an toàn."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    start   = cleaned.find("{")
    end     = cleaned.rfind("}")
    if start == -1 or end == -1:
        return _empty_extraction(error="No JSON object found")
    try:
        data = json.loads(cleaned[start : end + 1])
        # Đảm bảo tất cả key bắt buộc tồn tại
        for key in ("key_rules", "numbers_deadlines", "subjects",
                    "edge_cases", "suggested_questions"):
            if key not in data:
                data[key] = []
        return data
    except json.JSONDecodeError as exc:
        return _empty_extraction(error=str(exc))


def _empty_extraction(error: str = "") -> dict:
    return {
        "key_rules": [],
        "numbers_deadlines": [],
        "subjects": [],
        "edge_cases": [],
        "suggested_questions": [],
        "_error": error,
    }

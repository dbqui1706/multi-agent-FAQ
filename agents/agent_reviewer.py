import logging
import json
import re
import time
from typing import Any

from google import genai

logger = logging.getLogger(__name__)

_APPROVAL_THRESHOLD     = 6
_CONTEXT_INDEP_REQUIRED = True  # Nếu True: context_dependent → tự động reject


_SYSTEM_PROMPT = """Bạn là chuyên gia kiểm duyệt FAQ về Quy Chế Đào Tạo Thạc Sĩ.

Đánh giá cặp Q&A theo 5 tiêu chí. Trả về JSON (không có markdown):

{
  "scores": {
    "accuracy": <0-10>,
    "relevance": <0-10>,
    "clarity": <0-10>,
    "completeness": <0-10>,
    "context_independence": <0 hoặc 10>
  },
  "context_independence_ok": <true/false>,
  "notes": "<nhận xét ngắn gọn, tối đa 2 câu>",
  "improvement_hint": "<gợi ý sửa nếu reject, để trống nếu approve>",
  "approved": <true/false>
}

=== TIÊU CHÍ CHẤM ĐIỂM ===

accuracy (0-10, trọng số 35%):
  - 9-10: Hoàn toàn đúng với context, giữ đủ số liệu/điều kiện
  - 7-8:  Đúng nhưng thiếu 1-2 điều kiện phụ
  - 5-6:  Có sai sót nhỏ hoặc thiếu thông tin quan trọng
  - 0-4:  Sai sự thật hoặc suy diễn không có căn cứ

relevance (0-10, trọng số 25%):
  - 9-10: Câu hỏi thiết thực, đúng nhu cầu thực tế của sinh viên/GV
  - 7-8:  Liên quan nhưng không phải câu hỏi thường gặp
  - 5-6:  Ít liên quan hoặc quá học thuật
  - 0-4:  Không liên quan

clarity (0-10, trọng số 20%):
  - 9-10: Câu hỏi và trả lời rõ ràng, dễ hiểu ngay lần đầu
  - 7-8:  Tương đối rõ ràng
  - 5-6:  Cần đọc lại nhiều lần
  - 0-4:  Mơ hồ, khó hiểu

completeness (0-10, trọng số 10%):
  - 9-10: Trả lời đầy đủ, bao gồm ngoại lệ và trường hợp đặc biệt
  - 7-8:  Trả lời đủ ý chính
  - 5-6:  Thiếu thông tin đáng kể
  - 0-4:  Quá sơ sài

context_independence (0 hoặc 10, trọng số 10%):
  - 10: Câu hỏi tự thân — người chưa đọc tài liệu vẫn hiểu ngay
  - 0:  Dùng "nó", "điều này", "quy định trên", "như đã nêu", thiếu chủ thể,
        hoặc dạng "Điều kiện đó là gì?" / "Trường hợp nào được áp dụng?"

approved = true khi:
  (1) Điểm tổng có trọng số >= 6.0 VÀ
  (2) context_independence_ok = true (bắt buộc)"""


def run(
    faq_items: list[dict[str, Any]],
    client: genai.Client,
    model_name: str,
) -> list[dict[str, Any]]:
    """
    Review từng FAQ item. Trả về chỉ các item được duyệt.

    Returns:
        Filtered list với review metadata: review_score, review_notes,
        review_breakdown, improvement_hint, is_approved.
    """
    logger.info("[Reviewer] Reviewing %d FAQ items...", len(faq_items))
    reviewed: list[dict[str, Any]] = []

    approved_count = 0
    retry_count = 0
    reject_count = 0

    for i, item in enumerate(faq_items, 1):
        logger.info("[Reviewer] [%d/%d] %s", i, len(faq_items), item.get("question", "")[:70])

        prompt = _build_prompt(item)

        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            review = _parse_review_json(response.text.strip())
        except Exception as exc:
            logger.warning("[Reviewer] Error on item %s: %s", item["id"], exc)
            review = _error_fallback(str(exc))

        weighted_score = _compute_weighted_score(review.get("scores", {}))
        ctx_ok   = review.get("context_independence_ok", True)
        approved = (weighted_score >= _APPROVAL_THRESHOLD and (ctx_ok or not _CONTEXT_INDEP_REQUIRED))

        item = dict(item)
        item["review_score"]     = round(weighted_score, 2)
        item["review_breakdown"] = review.get("scores", {})
        item["review_notes"]     = review.get("notes", "")
        item["improvement_hint"] = review.get("improvement_hint", "")
        item["is_approved"]      = approved

        reviewed.append(item)
        if approved:
            approved_count += 1
        elif review.get("is_retry"):
            reject_count += 1
        else:
            retry_count += 1 # first fail

        time.sleep(0.5)

    logger.info(
       "[Reviewer] approved=%d  retry=%d  reject=%d / %d total",
        approved_count, retry_count, reject_count, len(faq_items)
    )
    return reviewed


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(item: dict) -> str:
    return f"""{_SYSTEM_PROMPT}

---
NGUỒN: {item.get('source', 'N/A')}
PERSONA: {item.get('persona', 'student')}

CÂU HỎI: {item.get('question', '')}

CÂU TRẢ LỜI: {item.get('answer', '')}

CONTEXT GỐC:
{item.get('context', '')}
---

JSON:"""


def _compute_weighted_score(scores: dict) -> float:
    """[FIX-2] Điểm tổng có trọng số thực sự."""
    weights = {
        "accuracy"            : 0.35,
        "relevance"           : 0.25,
        "clarity"             : 0.20,
        "completeness"        : 0.10,
        "context_independence": 0.10,
    }
    return sum(scores.get(k, 5.0) * w for k, w in weights.items())


def _parse_review_json(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    start   = cleaned.find("{")
    end     = cleaned.rfind("}")
    if start == -1 or end == -1:
        return _error_fallback("No JSON object found")
    try:
        data = json.loads(cleaned[start : end + 1])
        if "scores" not in data:
            data["scores"] = {k: 5 for k in
                ("accuracy", "relevance", "clarity",
                 "completeness", "context_independence")}
        if "context_independence_ok" not in data:
            data["context_independence_ok"] = (
                data["scores"].get("context_independence", 10) > 0
            )
        return data
    except json.JSONDecodeError as exc:
        return _error_fallback(str(exc))


def _error_fallback(reason: str) -> dict:
    """[FIX-1] Default reject khi review lỗi."""
    return {
        "scores": {
            "accuracy": 4, "relevance": 5, "clarity": 5,
            "completeness": 4, "context_independence": 5,
        },
        "context_independence_ok": True,
        "notes": f"Review error: {reason[:100]}",
        "improvement_hint": "",
        "approved": False,
    }

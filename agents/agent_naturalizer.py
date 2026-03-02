import json
import logging
import re
import time
from typing import Any

from google import genai

logger = logging.getLogger(__name__)

_PERSONA_PROFILES = {
    "student": (
        "Sinh viên đại học đang cân nhắc học thạc sĩ hoặc học viên mới nhập học. "
        "Hỏi đơn giản, chưa biết thuật ngữ chuyên ngành, thường hỏi theo kiểu 'có được không', 'cần gì'."
    ),
    "lecturer": (
        "Giảng viên / người hướng dẫn luận văn. "
        "Biết thuật ngữ, hỏi về nghĩa vụ, quyền hạn và quy trình của mình."
    ),
    "admin": (
        "Cán bộ quản lý đào tạo. "
        "Hỏi về quy trình xét duyệt, thẩm quyền, và các trường hợp đặc biệt cần xử lý."
    ),
}

_SYSTEM_PROMPT = """Viết lại câu hỏi sau theo ngôn ngữ tự nhiên mà người dùng thực tế sẽ gõ vào chatbot hoặc Google.

NGUYÊN TẮC:
- Giữ nguyên NỘI DUNG và Ý NGHĨA của câu hỏi gốc
- Dùng ngôn ngữ tự nhiên, đời thường — KHÔNG dùng văn phong văn bản pháp lý
- Câu hỏi vẫn phải TỰ THÂN (nêu đủ chủ thể, không cần đọc tài liệu mới hiểu)
- Ngắn gọn hơn nếu có thể

CHỈ trả về câu hỏi đã viết lại — KHÔNG giải thích, KHÔNG markdown."""


def run(
    faq_items : list[dict[str, Any]],
    client    : genai.Client,
    model_name: str,
) -> list[dict[str, Any]]:
    """Viết lại question cho từng FAQ item theo persona."""
    logger.info("[Naturalizer] %d items", len(faq_items))
    result = []

    for i, item in enumerate(faq_items, 1):
        persona = item.get("persona", "student")
        persona_desc = _PERSONA_PROFILES.get(persona, _PERSONA_PROFILES["student"])

        logger.info("[Naturalizer] [%d/%d] %s",
                    i, len(faq_items), item.get("question", "")[:60])

        prompt = f"""{_SYSTEM_PROMPT}

PERSONA: {persona} — {persona_desc}

CÂU HỎI GỐC: {item['question']}

Câu hỏi viết lại:"""

        try:
            resp     = client.models.generate_content(model=model_name, contents=prompt)
            new_q    = resp.text.strip().strip('"').strip("'")
            # Sanity check: nếu kết quả quá ngắn hoặc quá dài thì giữ nguyên
            if 10 < len(new_q) < 300 and "?" in new_q:
                updated = dict(item)
                updated["question_original"] = item["question"]
                updated["question"]          = new_q
                updated["source_agent"]      = "naturalizer"
                result.append(updated)
            else:
                logger.debug("[Naturalizer] kept original (bad rewrite): %s", new_q[:60])
                result.append(dict(item))
        except Exception as e:
            logger.warning("[Naturalizer] item %s error: %s", item.get("id"), e)
            result.append(dict(item))

        time.sleep(0.3)

    logger.info("[Naturalizer] Done.")
    return result


def rewrite(
    item      : dict[str, Any],
    client    : genai.Client,
    model_name: str,
) -> dict[str, Any] | None:
    """Viết lại câu hỏi lần 2 dựa trên improvement_hint từ Reviewer."""
    hint = item.get("improvement_hint", "")
    if not hint:
        return None

    prompt = f"""Viết lại câu hỏi FAQ dưới đây. Chỉ trả về câu hỏi mới — không giải thích.

LỖI CẦN SỬA: {hint}
CÂU HỎI GỐC: {item.get('question', '')}

Câu hỏi viết lại:"""

    try:
        resp  = client.models.generate_content(model=model_name, contents=prompt)
        new_q = resp.text.strip().strip('"').strip("'")
        if 10 < len(new_q) < 300 and "?" in new_q:
            updated = dict(item)
            updated["question"] = new_q
            updated["is_retry"] = True
            return updated
    except Exception as e:
        logger.warning("[Naturalizer] rewrite error: %s", e)
    return None
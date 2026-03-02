import json
import logging
import re

from pathlib import Path
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output-graph"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_json(data, filename: str) -> None:
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    n = len(data) if isinstance(data, list) else 1

def load_json(filename: str):
    path = OUTPUT_DIR / filename
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def parse_json_array(raw: str) -> list[dict]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(cleaned[start:end+1])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def parse_json_object(raw: str) -> dict | None:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(cleaned[start:end+1])
    except json.JSONDecodeError:
        return None

def save_markdown(faqs: list[dict]) -> None:
    path  = OUTPUT_DIR / "faq_final.md"
    lines = [
        "# FAQ — Quy Chế Đào Tạo Trình Độ Thạc Sĩ",
        "",
        f"> 🤖 Tạo tự động bởi Sequential Multi-Agent Pipeline  ",
        f"> 📅 Ngày tạo: {datetime.now().strftime('%d/%m/%Y %H:%M')}  ",
        f"> 📊 Tổng số Q&A: **{len(faqs)}**",
        "",
        "---",
        "",
    ]

    grouped: dict[str, list[dict]] = defaultdict(list)
    for faq in faqs:
        grouped[faq.get("source", "Khác")].append(faq)

    PERSONA_EMOJI = {
        "student": "🎓",
        "lecturer": "👨‍🏫",
        "admin": "🏫",
    }

    for source, items in grouped.items():
        lines.append(f"## {source}")
        lines.append("")
        for faq in items:
            persona_label = PERSONA_EMOJI.get(faq.get("persona", ""), "")
            lines.append(f"### {persona_label} ❓ {faq['question']}")
            lines.append("")
            lines.append(f"**Trả lời:** {faq['answer']}")
            lines.append("")

            ctx = faq.get("context", "").strip()
            if ctx:
                lines.append("> **📄 Context (nguyên văn tài liệu):**")
                for ctx_line in ctx.split("\n"):
                    lines.append(f"> {ctx_line}")
            lines.append("")

            # [FIX-3] Hiển thị review breakdown đầy đủ
            breakdown = faq.get("review_breakdown", {})
            score     = faq.get("review_score", "-")
            notes     = faq.get("review_notes", "")
            hint      = faq.get("improvement_hint", "")
            pages     = faq.get("page_numbers", [])

            meta_parts = [f"⭐ Score: **{score}/10**"]
            if breakdown:
                bd_str = " | ".join(
                    f"{k}: {v}"
                    for k, v in breakdown.items()
                )
                meta_parts.append(f"📋 [{bd_str}]")
            if notes:
                meta_parts.append(f"📝 {notes}")
            if pages:
                meta_parts.append(f"📖 Trang: {', '.join(str(p) for p in pages)}")

            lines.append(f"<sub>{' &nbsp;|&nbsp; '.join(meta_parts)}</sub>")
            lines.append("")
            lines.append("---")
            lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Saved: %s  (%d items)", path.name, len(faqs))

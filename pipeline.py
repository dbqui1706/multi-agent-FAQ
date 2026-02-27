"""
pipeline.py — Sequential Multi-Agent FAQ Pipeline — FIXED VERSION
==================================================================
Fix so với bản gốc:
  [FIX-1] Step 1 và 2 được bật lại đúng cách, không còn load file cũ
  [FIX-2] Bổ sung STEP 4.5: Deduplication (loại câu hỏi quá giống nhau)
  [FIX-3] _save_markdown() hiển thị review_breakdown + improvement_hint
  [FIX-4] Pipeline có thể resume từ bất kỳ step nào qua biến RESUME_FROM_STEP
  [IMP-1] Stats cuối pipeline chi tiết hơn (rejection reasons, persona dist)

Stages:
  1. Chunker       → Split PDF thành Chương/Điều chunks
  2. Extractor     → Extract key info có cấu trúc (JSON) từ mỗi chunk
  3. FAQ Generator → Sinh Q&A với context independence + diversity control
  4. Reviewer      → Chấm 5 tiêu chí, reject context-dependent questions
  4.5. Dedup       → Loại câu hỏi trùng lặp (cosine similarity > 0.85)
  5. Output        → Lưu JSON + Markdown
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
PDF_PATH   = BASE_DIR / "data" / "QUY CHẾ ĐÀO TẠO TRÌNH ĐỘ THẠC SĨ.pdf"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY    = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# [FIX-4] Resume control: đặt số step muốn bắt đầu lại (1 = chạy từ đầu)
# Ví dụ: RESUME_FROM_STEP=3 → load step_2_extracted.json, chạy từ step 3
RESUME_FROM_STEP = int(os.getenv("RESUME_FROM_STEP", "1"))

# Dedup similarity threshold — cặp câu hỏi có similarity > threshold sẽ bị loại
DEDUP_THRESHOLD = float(os.getenv("DEDUP_THRESHOLD", "0.85"))

# ── Logging ───────────────────────────────────────────────────────────────────
import io
_utf8_stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(_utf8_stdout),
        logging.FileHandler(OUTPUT_DIR / "pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _save_json(data, filename: str) -> None:
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    n = len(data) if isinstance(data, list) else 1
    logger.info("Saved: %s  (%d items)", path.name, n)


def _load_json(filename: str):
    path = OUTPUT_DIR / filename
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _banner(step, name: str) -> None:
    sep = "─" * 62
    logger.info("\n%s\n  STEP %s — %s\n%s", sep, step, name, sep)


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication (Step 4.5)
# ─────────────────────────────────────────────────────────────────────────────

def _dedup_faqs(
    faqs: list[dict],
    threshold: float = 0.85,
) -> list[dict]:
    """
    [FIX-2] Loại bỏ câu hỏi quá giống nhau dựa trên TF-IDF cosine similarity.
    Giữ lại câu hỏi đầu tiên khi 2 câu hỏi similarity > threshold.

    Returns:
        Danh sách FAQ đã dedup.
    """
    if len(faqs) < 2:
        return faqs

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        logger.warning("[Dedup] sklearn not available, skipping deduplication.")
        return faqs

    questions = [f["question"] for f in faqs]
    vec       = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True
    ).fit_transform(questions)
    sim_matrix = cosine_similarity(vec)

    n         = len(faqs)
    to_remove = set()

    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            if sim_matrix[i, j] > threshold:
                # Giữ câu hỏi có review_score cao hơn
                score_i = faqs[i].get("review_score", 0)
                score_j = faqs[j].get("review_score", 0)
                remove_idx = j if score_i >= score_j else i
                to_remove.add(remove_idx)
                logger.debug(
                    "[Dedup] Removing duplicate (sim=%.3f): %s",
                    sim_matrix[i, j],
                    faqs[remove_idx]["question"][:60],
                )

    deduped = [f for idx, f in enumerate(faqs) if idx not in to_remove]
    logger.info(
        "[Dedup] %d → %d FAQs (removed %d duplicates, threshold=%.2f)",
        n, len(deduped), len(to_remove), threshold,
    )
    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    if not API_KEY or API_KEY == "your_gemini_api_key_here":
        logger.error("GEMINI_API_KEY chưa được thiết lập trong file .env!")
        sys.exit(1)

    if not PDF_PATH.exists():
        logger.error("Không tìm thấy file PDF: %s", PDF_PATH)
        sys.exit(1)

    client         = genai.Client(api_key=API_KEY)
    pipeline_start = time.time()

    logger.info("=" * 62)
    logger.info("  SEQUENTIAL MULTI-AGENT FAQ PIPELINE (FIXED)")
    logger.info("  PDF:    %s", PDF_PATH.name)
    logger.info("  Model:  %s", MODEL_NAME)
    logger.info("  Resume: step %d", RESUME_FROM_STEP)
    logger.info("  Start:  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 62)

    # ── STEP 1: CHUNKER ───────────────────────────────────────────────────────
    _banner(1, "CHUNKER — Phân tích cấu trúc PDF")
    if RESUME_FROM_STEP <= 1:
        from agents import agent_chunker
        chunks = agent_chunker.run(str(PDF_PATH))
        _save_json(chunks, "step_1_chunks.json")
        logger.info("✓ Chunker: %d chunks", len(chunks))
    else:
        chunks = _load_json("step_1_chunks.json")
        logger.info("↩ Loaded step_1_chunks.json (%d chunks)", len(chunks))

    # ── STEP 2: EXTRACTOR ─────────────────────────────────────────────────────
    _banner(2, "EXTRACTOR — Trích xuất thông tin có cấu trúc")
    if RESUME_FROM_STEP <= 2:
        from agents import agent_extractor
        enriched_chunks = agent_extractor.run(chunks, client, MODEL_NAME)
        _save_json(enriched_chunks, "step_2_extracted.json")
        logger.info("✓ Extractor: %d chunks enriched", len(enriched_chunks))
    else:
        enriched_chunks = _load_json("step_2_extracted.json")
        logger.info("↩ Loaded step_2_extracted.json (%d chunks)", len(enriched_chunks))

    # ── STEP 3: FAQ GENERATOR ─────────────────────────────────────────────────
    _banner(3, "FAQ GENERATOR — Sinh Q&A với diversity control")
    if RESUME_FROM_STEP <= 3:
        from agents import agent_faq_generator
        raw_faqs = agent_faq_generator.run(enriched_chunks, client, MODEL_NAME)
        _save_json(raw_faqs, "step_3_faqs.json")
        logger.info("✓ FAQ Generator: %d Q&A pairs", len(raw_faqs))
    else:
        raw_faqs = _load_json("step_3_faqs.json")
        logger.info("↩ Loaded step_3_faqs.json (%d items)", len(raw_faqs))

    # ── STEP 4: REVIEWER ──────────────────────────────────────────────────────
    _banner(4, "REVIEWER — Chấm 5 tiêu chí, reject context-dependent")
    if RESUME_FROM_STEP <= 4:
        from agents import agent_reviewer
        reviewed_faqs = agent_reviewer.run(raw_faqs, client, MODEL_NAME)
        _save_json(reviewed_faqs, "step_4_reviewed.json")
        logger.info(
            "✓ Reviewer: %d/%d approved (%.0f%%)",
            len(reviewed_faqs), len(raw_faqs),
            (len(reviewed_faqs) / len(raw_faqs) * 100) if raw_faqs else 0,
        )
    else:
        reviewed_faqs = _load_json("step_4_reviewed.json")
        logger.info("↩ Loaded step_4_reviewed.json (%d items)", len(reviewed_faqs))

    # ── STEP 4.5: DEDUPLICATION ───────────────────────────────────────────────
    _banner("4.5", "DEDUPLICATION — Loại câu hỏi trùng lặp")
    deduped_faqs = _dedup_faqs(reviewed_faqs, threshold=DEDUP_THRESHOLD)
    _save_json(deduped_faqs, "step_45_deduped.json")

    # ── STEP 5: OUTPUT ────────────────────────────────────────────────────────
    _banner(5, "OUTPUT — Xuất kết quả cuối")
    final_faqs = deduped_faqs
    _save_json(final_faqs, "faq_final.json")
    _save_markdown(final_faqs)

    elapsed = time.time() - pipeline_start
    _print_final_stats(final_faqs, raw_faqs, elapsed)


# ─────────────────────────────────────────────────────────────────────────────
# Output formatters
# ─────────────────────────────────────────────────────────────────────────────

def _print_final_stats(
    final_faqs: list[dict],
    raw_faqs: list[dict],
    elapsed: float,
) -> None:
    """[IMP-1] Thống kê cuối pipeline chi tiết."""
    from collections import Counter

    persona_dist = Counter(f.get("persona", "unknown") for f in final_faqs)
    score_avg    = (
        sum(f.get("review_score", 0) for f in final_faqs) / len(final_faqs)
        if final_faqs else 0
    )

    logger.info("\n%s", "=" * 62)
    logger.info("  PIPELINE HOÀN THÀNH")
    logger.info("  Thời gian:    %.1f giây", elapsed)
    logger.info("  Q&A raw:      %d", len(raw_faqs))
    logger.info("  Q&A final:    %d (sau review + dedup)", len(final_faqs))
    logger.info("  Review score avg: %.2f/10", score_avg)
    logger.info("  Persona distribution: %s", dict(persona_dist))
    logger.info("  Output files:")
    logger.info("    faq_final.json      — FAQ cuối (JSON)")
    logger.info("    faq_final.md        — FAQ cuối (Markdown)")
    logger.info("    step_45_deduped.json — Sau dedup")
    logger.info("    step_4_reviewed.json — Sau review")
    logger.info("%s\n", "=" * 62)


def _save_markdown(faqs: list[dict]) -> None:
    """
    [FIX-3] Format markdown với đầy đủ review_breakdown và improvement_hint.
    """
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

    from collections import defaultdict
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


if __name__ == "__main__":
    run_pipeline()

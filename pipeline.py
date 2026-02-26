"""
pipeline.py — Sequential Multi-Agent FAQ Pipeline Orchestrator
==============================================================
Stages (run in order):
  1. Chunker      → Split PDF into Chương/Điều chunks
  2. Extractor    → Extract key info from each chunk (Gemini)
  3. FAQ Generator→ Generate Q&A pairs with context (Gemini)
  4. Reviewer     → Score & filter Q&A pairs (Gemini)
  5. Output       → Save to JSON + Markdown

Intermediate outputs are saved to output/step_*.json for easy debugging.
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

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "data" / "QUY CHẾ ĐÀO TẠO TRÌNH ĐỘ THẠC SĨ.pdf"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Env & Config ──────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── Logging setup ─────────────────────────────────────────────────────────────
import io
_utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
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


def _save_json(data, filename: str) -> None:
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s  (%d items)", path.name, len(data) if isinstance(data, list) else 1)


def _banner(step: int, name: str) -> None:
    sep = "─" * 60
    logger.info("\n%s\n  STEP %d — %s\n%s", sep, step, name, sep)


def run_pipeline() -> None:
    # ── Validate config ──────────────────────────────────────────────────────
    if not API_KEY or API_KEY == "your_gemini_api_key_here":
        logger.error(
            "GEMINI_API_KEY chưa được thiết lập!\n"
            "Mở file .env và thêm API key của bạn vào."
        )
        sys.exit(1)

    if not PDF_PATH.exists():
        logger.error("Không tìm thấy file PDF: %s", PDF_PATH)
        sys.exit(1)

    # Initialize new google.genai client
    client = genai.Client(api_key=API_KEY)

    pipeline_start = time.time()
    logger.info("=" * 60)
    logger.info("  SEQUENTIAL MULTI-AGENT FAQ PIPELINE")
    logger.info("  PDF: %s", PDF_PATH.name)
    logger.info("  Model: %s", MODEL_NAME)
    logger.info("  Bắt đầu: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    # ── STEP 1: CHUNKER ──────────────────────────────────────────────────────
    _banner(1, "CHUNKER — Phân tích cấu trúc PDF")
    # from agents import agent_chunker
    # chunks = agent_chunker.run(str(PDF_PATH))
    # _save_json(chunks, "step_1_chunks.json")
    # logger.info("✓ Chunker: %d chunks tạo thành công", len(chunks))
    # Load file chunker json
    # with open(OUTPUT_DIR / "step_1_chunks.json", "r", encoding="utf-8") as f:
    #     chunks = json.load(f)

    # ── STEP 2: EXTRACTOR ────────────────────────────────────────────────────
    _banner(2, "EXTRACTOR — Trich xuat thong tin quan trong")
    # from agents import agent_extractor
    # enriched_chunks = agent_extractor.run(chunks, client, MODEL_NAME)
    # _save_json(enriched_chunks, "step_2_extracted.json")
    # logger.info("Extractor: %d chunks da duoc lam giau", len(enriched_chunks))
    with open(OUTPUT_DIR / "step_2_extracted.json", "r", encoding="utf-8") as f:
        enriched_chunks = json.load(f)

    # ── STEP 3: FAQ GENERATOR ────────────────────────────────────────────────
    _banner(3, "FAQ GENERATOR — Tao Q&A voi context")
    from agents import agent_faq_generator
    raw_faqs = agent_faq_generator.run(enriched_chunks, client, MODEL_NAME)
    _save_json(raw_faqs, "step_3_faqs.json")
    logger.info("FAQ Generator: %d Q&A pairs da tao", len(raw_faqs))
    with open(OUTPUT_DIR / "step_3_faqs.json", "r", encoding="utf-8") as f:
        raw_faqs = json.load(f)

    # # ── STEP 4: REVIEWER ─────────────────────────────────────────────────────
    _banner(4, "REVIEWER — Kiem tra chat luong Q&A")
    from agents import agent_reviewer
    approved_faqs = agent_reviewer.run(raw_faqs, client, MODEL_NAME)
    _save_json(approved_faqs, "step_4_reviewed.json")
    logger.info(
        "Reviewer: %d/%d Q&A duoc duyet (%.0f%%)",
        len(approved_faqs),
        len(raw_faqs),
        (len(approved_faqs) / len(raw_faqs) * 100) if raw_faqs else 0,
    )

    # # ── STEP 5: OUTPUT ───────────────────────────────────────────────────────
    # _banner(5, "OUTPUT — Xuat ket qua cuoi")
    _save_json(approved_faqs, "faq_final.json")
    _save_markdown(approved_faqs)

    #
    elapsed = time.time() - pipeline_start
    logger.info("\n%s", "=" * 60)
    logger.info("  HOAN THANH! Thoi gian: %.1f giay", elapsed)
    logger.info("  Output files trong thu muc: output/")
    logger.info("    faq_final.json             - FAQ output (JSON)")
    logger.info("    faq_final.md               - FAQ output (Markdown)")
    logger.info("%s\n", "=" * 60)


def _save_markdown(faqs: list[dict]) -> None:
    """Format and save approved FAQs as a readable Markdown file."""
    path = OUTPUT_DIR / "faq_final.md"
    lines = [
        "# FAQ — Quy Chế Đào Tạo Trình Độ Thạc Sĩ",
        "",
        f"> Tạo tự động bởi Sequential Multi-Agent Pipeline  ",
        f"> Ngày tạo: {datetime.now().strftime('%d/%m/%Y %H:%M')}  ",
        f"> Tổng số Q&A: **{len(faqs)}**",
        "",
        "---",
        "",
    ]

    # Group by source
    from collections import defaultdict
    grouped: dict[str, list[dict]] = defaultdict(list)
    for faq in faqs:
        grouped[faq.get("source", "Khác")].append(faq)

    for source, items in grouped.items():
        lines.append(f"## {source}")
        lines.append("")
        for faq in items:
            lines.append(f"### ❓ {faq['question']}")
            lines.append("")
            lines.append(f"**Trả lời:** {faq['answer']}")
            lines.append("")

            # Context block
            ctx = faq.get("context", "").strip()
            if ctx:
                lines.append("> **📄 Context (đoạn văn gốc liên quan):**")
                for ctx_line in ctx.split("\n"):
                    lines.append(f"> {ctx_line}")
            lines.append("")

            # Review metadata
            score = faq.get("review_score", "-")
            notes = faq.get("review_notes", "")
            pages = faq.get("page_numbers", [])
            meta_parts = [f"⭐ Review score: **{score}/10**"]
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


def _save_evaluation_markdown(report: dict) -> None:
    """Render the evaluation report as a human-readable Markdown file."""
    path = OUTPUT_DIR / "evaluation_report.md"
    summary = report.get("summary", {})
    coverage = report.get("coverage_detail", {})
    items = report.get("per_item", [])

    def _bar(score: float, max_val: float = 1.0, width: int = 20) -> str:
        """Simple ASCII progress bar."""
        ratio = min(score / max_val, 1.0)
        filled = round(ratio * width)
        return "█" * filled + "░" * (width - filled) + f"  {score:.3f}"

    lines = [
        "# Evaluation Report — FAQ Quy Chế Đào Tạo Trình Độ Thạc Sĩ",
        "",
        f"> Ngày đánh giá: {datetime.now().strftime('%d/%m/%Y %H:%M')}  ",
        f"> Tổng số Q&A được đánh giá: **{summary.get('total_faq_items', len(items))}**  ",
        f"> Phương pháp: LLM-as-a-Judge (Gemini) + Statistical",
        "",
        "---",
        "",
        "## 📊 Overall Scorecard",
        "",
        "| Metric | Score | Scale | Bar |",
        "|--------|-------|-------|-----|",
        f"| **Faithfulness Score** | `{summary.get('faithfulness_score', 0):.3f}` | [0, 1] | {_bar(summary.get('faithfulness_score', 0))} |",
        f"| **Answer Relevance** | `{summary.get('answer_relevance_avg', 0):.2f}` | [1, 5] | {_bar(summary.get('answer_relevance_avg', 0), max_val=5)} |",
        f"| **Context Independence Rate** | `{summary.get('context_independence_rate', 0):.1%}` | {{0,1}} | {_bar(summary.get('context_independence_rate', 0))} |",
        f"| **Diversity Score** | `{summary.get('diversity_score', 0):.3f}` | [0, 1] | {_bar(summary.get('diversity_score', 0))} |",
        f"| **Context Coverage / Recall** | `{summary.get('context_coverage', 0):.3f}` | [0, 1] | {_bar(summary.get('context_coverage', 0))} |",
        f"| **Retrieval Effectiveness** | `{summary.get('retrieval_effectiveness_avg', 0):.3f}` | [0, 1] | {_bar(summary.get('retrieval_effectiveness_avg', 0))} |",
        "",
        "---",
        "",
        "## 📋 Metric Definitions",
        "",
        "| Metric | Mô tả | Phương pháp |",
        "|--------|-------|-------------|",
        "| **Faithfulness Score** [0,1] | Câu trả lời có bám sát context gốc không? | LLM-as-a-Judge |",
        "| **Answer Relevance** [1-5] | Câu trả lời có thực sự trả lời câu hỏi không? | LLM-as-a-Judge |",
        "| **Context Independence** {0,1} | Câu hỏi có tự hiểu được không cần context bổ sung? | LLM-as-a-Judge |",
        "| **Diversity Score** [0,1] | Độ đa dạng câu hỏi (TTR + bigram + prefix) | Statistical |",
        "| **Context Coverage** [0,1] | FAQ có phủ đủ chủ đề của tài liệu gốc không? | LLM-as-a-Judge |",
        "| **Retrieval Effectiveness** [0,1] | Context passage có hỗ trợ chính xác câu trả lời? | LLM-as-a-Judge |",
        "",
        "---",
        "",
        "## 📈 Distributions",
        "",
        "### Answer Relevance Distribution (1–5)",
        "",
    ]

    ar_dist = summary.get("answer_relevance_distribution", {})
    if ar_dist:
        lines.append("| Score | Count | Fraction | Bar |")
        lines.append("|-------|-------|----------|-----|")
        for score_label, val in sorted(ar_dist.items(), key=lambda x: x[0]):
            frac = val.get("fraction", 0)
            bar = "█" * round(frac * 20)
            lines.append(f"| {score_label} | {val.get('count', 0)} | {frac:.1%} | {bar} |")
        lines.append("")

    lines += [
        "",
        "### Faithfulness Distribution",
        "",
    ]
    f_dist = summary.get("faithfulness_distribution", {})
    if f_dist:
        lines.append("| Range | Count | Fraction | Bar |")
        lines.append("|-------|-------|----------|-----|")
        for label, val in f_dist.items():
            frac = val.get("fraction", 0)
            bar = "█" * round(frac * 20)
            lines.append(f"| {label} | {val.get('count', 0)} | {frac:.1%} | {bar} |")
        lines.append("")

    lines += [
        "",
        "---",
        "",
        "## 🗺️ Context Coverage Detail",
        "",
        f"**Coverage Score:** `{coverage.get('context_coverage', 0):.3f}`  ",
        f"**Simple Source Ratio:** `{coverage.get('simple_source_ratio', 0):.3f}`  ",
        "",
        f"**Nhận xét:** {coverage.get('coverage_notes', '')}",
        "",
    ]

    uncovered = coverage.get("uncovered_topics", [])
    if uncovered:
        lines.append("**Chủ đề chưa được phủ:**")
        lines.append("")
        for t in uncovered[:20]:
            lines.append(f"- {t}")
        lines.append("")

    lines += [
        "---",
        "",
        "## 🔍 Per-Item Evaluation Details",
        "",
        "| # | Question (truncated) | Faith. | Relev. | Ctx.Ind. | Retr.Eff. | Notes |",
        "|---|----------------------|--------|--------|----------|-----------|-------|",
    ]

    for i, item in enumerate(items, 1):
        ev = item.get("eval", {})
        q = item.get("question", "")[:55].replace("|", "/")
        lines.append(
            f"| {i} | {q}… | "
            f"`{ev.get('faithfulness', 0):.2f}` | "
            f"`{ev.get('answer_relevance', '-')}` | "
            f"`{ev.get('context_independence', '-')}` | "
            f"`{ev.get('retrieval_effectiveness', 0):.2f}` | "
            f"{ev.get('notes', '')[:60]} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Saved: %s", path.name)


if __name__ == "__main__":
    run_pipeline()

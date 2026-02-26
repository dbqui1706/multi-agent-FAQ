"""
Agent: Docling Chunker
======================
Dùng Docling (pypdfium2 backend) để load PDF, gộp toàn bộ text
từ các pages, sau đó chia thành chunks dựa theo cấu trúc văn bản
pháp lý Việt Nam:

  Chương (CHƯƠNG I, II, ...) → chapter context (carry-forward)
  Điều   (Điều 1., Điều 2., ...) → term (article heading)
  Khoản  (1., 2., a., b., ...) → nằm trong content của Điều

Output: List[dict] với schema:
  {
    "id"          : "chunk_001",
    "chapter"     : "CHƯƠNG I — QUY ĐỊNH CHUNG",   # carry-forward
    "term"        : "Điều 1. Phạm vi điều chỉnh",  # None nếu chunk Chương
    "content"     : "<toàn bộ nội dung đoạn>",
    "page_numbers": [1, 2]
  }
"""

import re
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────────────────────────────────────

# CHƯƠNG I / CHƯƠNG II / CHƯƠNG 1 ...  ở đầu dòng
_CHUONG_RE = re.compile(
    r"^CHƯƠNG\s+[IVXLCDM\d]+[.\s]*",
    re.IGNORECASE | re.UNICODE,
)

# Điều X. Tên điều — primary heading (có dấu chấm sau số)
# Cho phép OCR artifact: "Điều 1 3. ..." → "Điều 13."
_DIEU_RE = re.compile(
    r"^Điều\s+\d[\d\s]*\d\.|^Điều\s+\d+\.",
    re.UNICODE,
)

# Pattern tách đoạn: Chương hoặc Điều primary — ở đầu dòng
_SPLIT_PATTERN = re.compile(
    r"(?m)^(?=CHƯƠNG\s+[IVXLCDM\d]+|Điều\s+\d[\d\s]*\.)",
    re.UNICODE,
)

# Pattern loại bỏ dòng cuối chứa số trang và 1 ký tự xuống dòng "\n"
_PAGE_NUMBER_RE = re.compile(r"^\d+\n$", re.UNICODE)

# Pattern loại bỏ dòng cuối chứa số trang và 1 ký tự xuống dòng "\n"
_PAGE_FOOTER_RE = re.compile(r'(?m)^[\s]*\d{1,4}[\s]*$', re.UNICODE)
_PAGE_FOOTER_RE2 = re.compile(r'(?m)\n[\s]*\d{1,4}[\s]*\Z', re.UNICODE)

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(pdf_path: str) -> list[dict[str, Any]]:
    """
    Load PDF bằng pypdfium2 (Docling dependency), chia thành chunks
    theo Chương / Điều.

    Args:
        pdf_path: Đường dẫn tuyệt đối đến file PDF.

    Returns:
        List[dict] — mỗi dict có: id, chapter, term, content, page_numbers
    """
    logger.info("[DoclingChunker] Loading PDF: %s", pdf_path)

    # ── 1. Load PDF dùng pypdfium2 (Docling's text-extraction backend) ────────
    pages = _load_pdf_pages(pdf_path)
    logger.info("[DoclingChunker] Tổng số trang: %d", len(pages))

    # ── 2. Gộp thành full_text ────────────────────────────────────────────────
    full_text = "".join(text for _, text in pages)

    # ── 3. Chia chunk theo Chương / Điều ──────────────────────────────────────
    chunks = _split_into_chunks(full_text, pages)
    logger.info("[DoclingChunker] Tổng chunks: %d", len(chunks))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# PDF text extraction (pypdfium2 — Docling dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _load_pdf_pages(pdf_path: str) -> list[tuple[int, str]]:
    """
    Dùng pypdfium2 để extract text từng trang PDF.
    pypdfium2 là dependency trực tiếp của Docling và được cài kèm.

    Returns:
        List of (page_number, page_text), page_number bắt đầu từ 1.
    """
    import pypdfium2 as pdfium  # noqa: PLC0415

    pdf = pdfium.PdfDocument(pdf_path)
    pages: list[tuple[int, str]] = []

    for i in range(len(pdf)):
        page = pdf[i]
        textpage = page.get_textpage()
        text = textpage.get_text_range()
        textpage.close()
        page.close()
        # Chuẩn hóa line endings — pdfium dùng \r\n trên Windows
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if text.strip():
            pages.append((i + 1, text))

    pdf.close()
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# Chunking logic
# ─────────────────────────────────────────────────────────────────────────────

def _split_into_chunks(
    full_text: str,
    pages: list[tuple[int, str]],
) -> list[dict[str, Any]]:
    """Chia full_text thành chunks và gán chapter/term/page_numbers."""
    segments = _SPLIT_PATTERN.split(full_text)
    chunks: list[dict[str, Any]] = []
    chunk_id = 0
    current_chapter: str | None = None

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        first_line = seg.split("\n")[0].strip()

        if _CHUONG_RE.match(first_line):
            # Chunk Chương
            current_chapter = _build_chapter_title(first_line, seg)
            term = None

        elif _DIEU_RE.match(first_line):
            # Chunk Điều
            term = _normalize_dieu_title(first_line)[:200]

        else:
            # Phần đầu tài liệu / Phụ lục / v.v.
            term = None

        seg_pages = _find_pages(seg[:80], pages)
        chunk_id += 1

        chunks.append(
            {
                "id": f"chunk_{chunk_id:03d}",
                "chapter": current_chapter,
                "term": term,
                "content": seg,
                "page_numbers": seg_pages,
            }
        )

    return chunks


def _build_chapter_title(first_line: str, seg: str) -> str:
    lines = [ln.strip() for ln in seg.split("\n") if ln.strip()]
    if len(lines) >= 2 and not _DIEU_RE.match(lines[1]):
        return f"{lines[0]} — {lines[1]}"[:200]
    return first_line[:200]


def _normalize_dieu_title(title: str) -> str:
    """
    Chuẩn hóa OCR artifacts trong tiêu đề Điều.
    Ví dụ: "Điều 1 3. Tên điều" → "Điều 13. Tên điều"
    """
    m = re.match(r"^(Điều\s+)([\d\s]+?)(\..*)", title, re.UNICODE)
    if m:
        prefix, num_raw, rest = m.groups()
        num_clean = "".join(num_raw.split())
        return f"{prefix}{num_clean}{rest}"
    return title


def _find_pages(snippet: str, pages: list[tuple[int, str]]) -> list[int]:
    found: set[int] = set()
    snippet_clean = snippet.strip()[:60]
    for page_no, page_text in pages:
        if snippet_clean and snippet_clean in page_text:
            found.add(page_no)
    return sorted(found)

def clean_page_numbers(text: str) -> str:
    text = _PAGE_FOOTER_RE.sub('', text)
    text = _PAGE_FOOTER_RE2.sub('', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()
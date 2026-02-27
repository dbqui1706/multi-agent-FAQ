import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────────────────────────────────────

_CHAPTER_RE = re.compile(
    r"^CHƯƠNG\s+[IVXLCDM\d]+[.\s]*",
    re.IGNORECASE | re.UNICODE,
)

_TERM_RE = re.compile(
    r"^Điều\s+\d[\d\s]*\d\.|^Điều\s+\d+\.",
    re.UNICODE,
)

# Pass-1: split theo CHƯƠNG
_CHAPTER_SPLIT_RE = re.compile(
    r"(?mi)^(?=CHƯƠNG\s+[IVXLCDM\d]+[.\s]*)",
    re.UNICODE,
)

# Pass-2: split theo Điều bên trong mỗi Chương
_TERM_SPLIT_RE = re.compile(
    r"(?m)^(?=Điều\s+\d[\d\s]*\.)",
    re.UNICODE,
)

# Page footer patterns
_PAGE_FOOTER_RE  = re.compile(r"(?m)^[\s]*\d{1,4}[\s]*$", re.UNICODE)
_PAGE_FOOTER_RE2 = re.compile(r"(?m)\n[\s]*\d{1,4}[\s]*\Z", re.UNICODE)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(pdf_path: str) -> list[dict[str, Any]]:
    """
    Load PDF by pypdfium2, split into chunks by Chapter / Term.

    Returns:
        List[dict] — each dict: id, chapter, term, content, page_numbers
    """
    logger.info("[Chunker] Loading PDF: %s", pdf_path)

    pages = _load_pdf_pages(pdf_path)
    logger.info("[Chunker] Total pages: %d", len(pages))

    full_text = "".join(text for _, text in pages)
    chunks = _split_into_chunks(full_text, pages)

    logger.info("[Chunker] Total chunks: %d", len(chunks))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# PDF text extraction
# ─────────────────────────────────────────────────────────────────────────────

def _load_pdf_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Use pypdfium2 to extract text from each PDF page."""
    import pypdfium2 as pdfium

    pdf   = pdfium.PdfDocument(pdf_path)
    pages: list[tuple[int, str]] = []

    for i in range(len(pdf)):
        page     = pdf[i]
        textpage = page.get_textpage()
        text     = textpage.get_text_range()
        textpage.close()
        page.close()
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
    """
    Split full_text into chunks by 2-pass:
      Pass 1 — split by CHAPTER
      Pass 2 — with each Chapter, loop to split by TERM
    All segments are cleaned by _clean_page_numbers() before processing.
    """
    chunks:  list[dict[str, Any]] = []
    chunk_id = 0
    current_chapter: str | None   = None

    # ── Pass 1: split by CHAPTER 
    chapter_segments = _CHAPTER_SPLIT_RE.split(full_text)

    for chap_seg in chapter_segments:
        chap_seg = chap_seg.strip()
        if not chap_seg:
            continue

        chap_seg = _clean_page_numbers(chap_seg)
        if not chap_seg:
            continue

        first_line = chap_seg.split("\n", 1)[0].strip()
        is_chapter = bool(_CHAPTER_RE.match(first_line))

        if is_chapter:
            # update chapter context
            current_chapter = _build_chapter_title(first_line, chap_seg)

            # split CHAPTER header and inner text
            dieu_match  = _TERM_SPLIT_RE.search(chap_seg)
            if dieu_match:
                chap_header = chap_seg[: dieu_match.start()].strip()
                inner_text  = chap_seg[dieu_match.start():]
            else:
                # Chapter without any TERM
                chap_header = chap_seg
                inner_text  = ""

            # Chunk header Chapter (title + short description, term = None)
            if chap_header:
                chunk_id += 1
                chunks.append({
                    "id"          : f"chunk_{chunk_id:03d}",
                    "chapter"     : current_chapter,
                    "term"        : None,                               
                    "content"     : chap_header,
                    "page_numbers": _find_pages_fuzzy(chap_header, pages),
                })

        else:
            # Part before Chapter I: introduction, document title, etc.
            current_chapter = None
            inner_text      = chap_seg

        # ── Pass 2: Loop each TERM in Chapter (or the first part of the document) 
        dieu_segments = _TERM_SPLIT_RE.split(inner_text)

        for dieu_seg in dieu_segments:
            dieu_seg = dieu_seg.strip()
            if not dieu_seg:
                continue

            # Clean each TERM separately
            dieu_seg = _clean_page_numbers(dieu_seg)
            if not dieu_seg:
                continue

            first_line_dieu = dieu_seg.split("\n", 1)[0].strip()

            if _TERM_RE.match(first_line_dieu):
                term = _normalize_dieu_title(first_line_dieu)[:200]
            else:
                # Remainder not a TERM (appendix, list of abbreviations, etc.)
                term = first_line_dieu[:100] if first_line_dieu else None

            chunk_id += 1
            chunks.append({
                "id"          : f"chunk_{chunk_id:03d}",
                "chapter"     : current_chapter,
                "term"        : term,
                "content"     : dieu_seg,
                "page_numbers": _find_pages_fuzzy(dieu_seg, pages),   
            })

    # Filter out empty or too short chunks (< 20 characters)
    chunks = [c for c in chunks if c["content"].strip() and len(c["content"]) > 20]
    return chunks


def _build_chapter_title(first_line: str, seg: str) -> str:
    """
    Get the Chapter title from the next line, regardless of whether it matches
    _TERM_RE — because the Chapter title never starts with "Điều".
    Only skip if the 2nd line is a real TERM title.
    """
    lines = [ln.strip() for ln in seg.split("\n") if ln.strip()]
    if len(lines) >= 2:
        second = lines[1]
        # The 2nd line is the chapter title if it is not a TERM title
        if not _TERM_RE.match(second):
            return f"{lines[0]} — {second}"[:200]
    return first_line[:200]


def _normalize_dieu_title(title: str) -> str:
    """Chuẩn hóa OCR artifacts: "Điều 1 3." → "Điều 13." """
    m = re.match(r"^(Điều\s+)([\d\s]+?)(\..*)", title, re.UNICODE)
    if m:
        prefix, num_raw, rest = m.groups()
        num_clean = "".join(num_raw.split())
        return f"{prefix}{num_clean}{rest}"
    return title


def _clean_page_numbers(text: str) -> str:
    """
    Remove PDF page numbers appearing as standalone lines.
    Called directly on each segment before processing.
    """
    text = _PAGE_FOOTER_RE.sub("", text)
    text = _PAGE_FOOTER_RE2.sub("", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _find_pages_fuzzy(seg: str, pages: list[tuple[int, str]]) -> list[int]:
    """
    Fuzzy page matching:
    - Try exact match with 3 different snippets (start / middle / end)
    - If not found, use trigram overlap to select the nearest page
    - Avoid page_numbers = [] due to 1 wrong OCR character
    """
    found: set[int] = set()
    seg_stripped = seg.strip()

    # Get 3 representative snippets
    snippets = _get_representative_snippets(seg_stripped)

    # Step 1: Exact match
    for snip in snippets:
        if not snip:
            continue
        for page_no, page_text in pages:
            if snip in page_text:
                found.add(page_no)

    if found:
        return sorted(found)

    # Step 2: Fuzzy — use trigram similarity if exact match fails
    best_page, best_score = -1, 0.0
    query_trigrams = _trigrams(seg_stripped[:200])

    if query_trigrams:
        for page_no, page_text in pages:
            page_trigrams = _trigrams(page_text[:500])
            if not page_trigrams:
                continue
            overlap = len(query_trigrams & page_trigrams)
            score = overlap / max(len(query_trigrams), len(page_trigrams))
            if score > best_score:
                best_score, best_page = score, page_no

        if best_score > 0.15 and best_page != -1:
            found.add(best_page)

    return sorted(found)


def _get_representative_snippets(text: str) -> list[str]:
    """Get 3 representative snippets: start, middle, end of segment."""
    n = len(text)
    snippets = []
    for start in [0, n // 3, 2 * n // 3]:
        snip = text[start : start + 50].strip()
        if len(snip) >= 20:
            snippets.append(snip)
    return snippets


def _trigrams(text: str) -> set[str]:
    """Create character trigram set from text."""
    clean = re.sub(r"\s+", " ", text).strip()
    return {clean[i : i + 3] for i in range(len(clean) - 2)}
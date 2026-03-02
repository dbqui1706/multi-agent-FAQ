import os
import logging
import time
import sys
import io

from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from IPython.display import display, Image
from graph.builder import build_graph

# Config
load_dotenv()
BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "data" / "QUY CHẾ ĐÀO TẠO TRÌNH ĐỘ THẠC SĨ.pdf"
API_KEY    = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

OUTPUT_LOG = BASE_DIR / "logs" / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Logging
_utf8_stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(_utf8_stdout),
        logging.FileHandler(OUTPUT_LOG, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    client = genai.Client(api_key=API_KEY)
    graph = build_graph(client, MODEL_NAME)

    config = {"configurable": {"thread_id": "faq-pipeline-1"}}

    # init state
    initial_state = {
        "COVERAGE_THRESHOLD": 0.92,
        "DEDUPLICATION_THRESHOLD": 0.92,
        "pdf_path"       : str(PDF_PATH),
        "raw_faqs"      : [],
        "approved_faqs" : [],
        "gap_fill_done" : False,
    }

    start = time.time()
    logger.info("=" * 60)
    logger.info("  FAQ PIPELINE v2 — LangGraph")
    logger.info("  PDF   : %s", PDF_PATH)
    logger.info("  Model : %s", MODEL_NAME)
    logger.info("  Start : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)
    
    # Save image of graph
    display(Image(graph.get_graph().draw_mermaid_png()))

    # Run graph
    graph.invoke(initial_state, config=config)

    # Final state
    final_state = graph.get_state(config)
    stats = final_state.values.get("stats", {})
    logger.info("Pipeline finished. Stats: %s", stats)
    
    elapsed = time.time() - start
    approved = final_state.get("approved_faqs", [])
    summary  = final_state.get("eval_summary", {})

    logger.info("=" * 60)
    logger.info("  COMPLETED — %.1fs", elapsed)
    logger.info("  FAQ COUNT : %d items", len(approved))
    logger.info("  COVERAGE  : %.1f%%", final_state.get("coverage", 0) * 100)
    if summary:
        logger.info("  FAITHFULNESS  : %.3f", summary.get("faithfulness_avg", 0))
        logger.info("  ANSWER RELEVANCE  : %.2f", summary.get("answer_relevance_avg", 0))
        logger.info("  DIVERSITY     : %.3f", summary.get("diversity_score", 0))
        logger.info("  CONTEXT COVERAGE  : %.3f", summary.get("context_coverage_avg", 0))
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
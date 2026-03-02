from typing import Any, Annotated
from typing_extensions import TypedDict
import operator

class GraphState(TypedDict):
    # Thresholds
    COVERAGE_THRESHOLD: float
    DEDUPLICATION_THRESHOLD: float
    

    # Input
    pdf_path: str
    model_name: str
    api_key: str

    # stage 1: Chunker
    chunks: list[dict[str, Any]]
    
    # stage 2: Extractor
    enriched_chunks: list[dict[str, Any]]
    
    # stage 3: Generators (parallel - Adversarial, FAQ Generator, Multichunk Generator)
    chunk_groups: list[list[dict[str, Any]]] # multichunk groups
    raw_faqs: Annotated[list[dict[str, Any]], operator.add]

    # stage 4: Merge + Sematic deduplication
    synthesis_faqs: list[dict[str, Any]]
    deduped_faqs: list[dict[str, Any]]
     
    # stage 5: Naturalizer
    naturalized_faqs: list[dict[str, Any]]
    
    # stage 6: Reviewer
    reviewed_faqs: list[dict[str, Any]]
    
    # stage 7: Coverage Gate
    coverage: float
    gap_fill_done: bool

    approved_faqs: Annotated[list[dict[str, Any]], operator.add]

    # stage 8: Evaluator
    eval_summary: dict[str, Any]


    errors: list[str]
    stats: dict[str, Any]


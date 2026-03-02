import logging
from typing import Any
from google import genai

from graph.state import GraphState
import agents.agent_adversarial as agent_adversarial
from agents import agent_chunker, agent_extractor
from agents import agent_faq_generator, agent_reviewer
from agents import agent_multichunk_synthesis
from agents import agent_merge, agent_naturalizer, agent_evaluator
from utils import save_json, save_markdown

logger = logging.getLogger(__name__)

def make_nodes(client: genai.Client, model_name: str):
    
    def node_chunker(state: GraphState):
        """Node 1: Chunk PDF"""
        logger.info("═══ [Node] Chunker ═══")
        try:
            chunks = agent_chunker.run(state["pdf_path"])
            save_json(chunks, "chunks.json")
            logger.info("[Node] Chunker: %d chunks", len(chunks))
            return {"chunks": chunks}
        except Exception as e:
            logger.error("[Node] Chunker failed: %s", e)
            return {"errors": state.get("errors", []) + [f"chunker: {e}"]}
    
    def node_extractor(state: GraphState) -> dict:
        """Node 2: Extractor — also used for gap-fill pass"""
        logger.info("═══ [Node] Extractor ═══")
        try:
            chunks_to_process = state.get("chunks", [])
            approved_faqs     = state.get("approved_faqs", [])
            is_gap_fill       = bool(approved_faqs)

            if is_gap_fill:
                covered_ids  = {f["source_chunk_id"] for f in approved_faqs}
                existing_enriched = state.get("enriched_chunks", []) or chunks_to_process
                chunks_to_process = [
                   c for c in existing_enriched
                   if c["id"] not in covered_ids and c.get("term")
                ]
                logger.info("Extractor (gap-fill): %d uncovered chunks", len(chunks_to_process))
                if not chunks_to_process:
                    logger.info("Extractor (gap-fill): all chunks covered, skipping")
                    return {"gap_fill_done": True}

            enriched = agent_extractor.run(
                chunks_to_process, client, model_name
            )
            logger.info("[Node] Extractor: %d enriched chunks", len(enriched))

            # Merge with previously enriched chunks (gap-fill reuses old ones)
            if is_gap_fill:
                old_enriched = state.get("enriched_chunks", [])
                existing_ids = {c["id"] for c in old_enriched}
                merged = old_enriched + [c for c in enriched if c["id"] not in existing_ids]
            else:
                merged = enriched

            # Group chunks
            groups = agent_multichunk_synthesis.group_chunks(enriched)
            logger.info("[Node] Extractor: %d chunk groups for multi-chunk", len(groups))

            result = {
                "enriched_chunks": merged,
                "chunk_groups"   : groups,
                "raw_faqs"       : [],
                "gap_fill_done"  : False,
            }
            save_json(result, "extractor_result.json")
            return result

        except Exception as e:
            logger.error("[Node] Extractor failed: %s", e)
            return {"errors": state.get("errors", []) + [f"extractor: {e}"]}

    def node_faq_generator(state: GraphState) -> dict:
        """Node 3: FAQ Generator"""
        logger.info("═══ [Node] FAQ Generator ═══")
        try:
            existing_questions = [f["question"] for f in state.get("approved_faqs", [])]
            faqs = agent_faq_generator.run(
                state["enriched_chunks"], client, model_name, existing_questions
            )
            logger.info("[Node] Generated %d FAQs", len(faqs))
            save_json(faqs, "raw_faqs.json")
            return {"raw_faqs": faqs}
        except Exception as e:
            logger.error("[Node] FAQ Generator failed: %s", e)
            return {"errors": state.get("errors", []) + [f"faq_gen: {e}"]}

    def node_adversarial(state: GraphState) -> dict:
        """Node 4: Adversarial"""
        logger.info("═══ [Node] Adversarial ═══")
        try:
            adversarial_faqs = agent_adversarial.run(
                state["enriched_chunks"], client, model_name
            )
            logger.info("[Node] Generated %d adversarial FAQs", len(adversarial_faqs))
            save_json(adversarial_faqs, "adversarial_faqs.json")
            return {"raw_faqs": adversarial_faqs}
        except Exception as e:
            logger.error("[Node] Adversarial failed: %s", e)
            return {"errors": state.get("errors", []) + [f"adversarial: {e}"]}

    def node_multichunk_synthesis(state: GraphState) -> dict:
        """Node 5: Multi-chunk Synthesis"""
        logger.info("═══ [Node] Multi-chunk Synthesis ═══")
        try:
            chunk_groups = state.get("chunk_groups", [])
            synthesis_faqs = agent_multichunk_synthesis.run(
                chunk_groups,
                client,
                model_name
            )
            logger.info("[Node] Generated %d synthesis FAQs", len(synthesis_faqs))
            save_json(synthesis_faqs, "synthesis_faqs.json")
            return {"raw_faqs": synthesis_faqs}
        except Exception as e:
            logger.error("[Node] Multi-chunk Synthesis failed: %s", e)
            return {"errors": state.get("errors", []) + [f"synthesis: {e}"]}

    def node_merge_dedup(state: GraphState) -> dict:
        """Node 6: Merge + Dedup"""
        logger.info("═══ [Node] Merge + Dedup ═══")
        try:
            raw = state.get("raw_faqs", [])
            logger.info("Merge: %d raw FAQs from 3 generators", len(raw))
            dedup_threshold = state.get("DEDUPLICATION_THRESHOLD", 0.72)
            deduped = agent_merge.run(
                raw, client, model_name, dedup_threshold
            )
            logger.info("Dedup: %d → %d (removed %d)", len(raw), len(deduped), len(raw)-len(deduped))
            save_json(deduped, "deduped_faqs.json")
            return {"deduped_faqs": deduped}
        except Exception as e:
            logger.error("[Node] Merge + Dedup failed: %s", e)
            return {"errors": state.get("errors", []) + [f"merge_dedup: {e}"]}

    def node_naturalizer(state: GraphState) -> dict:
        """Node 7: Naturalizer"""
        logger.info("═══ [Node] Naturalizer ═══")
        try:
            items = state.get("deduped_faqs", [])
            naturalized = agent_naturalizer.run(
                items, client, model_name
            )
            logger.info("Naturalized: %d items", len(naturalized))
            save_json(naturalized, "naturalized_faqs.json")
            return {"naturalized_faqs": naturalized}
        except Exception as e:
            logger.error("[Node] Naturalizer failed: %s", e)
            return {"errors": state.get("errors", []) + [f"naturalizer: {e}"]}

    def node_reviewer(state: GraphState) -> dict:
        """Node 8: Reviewer"""
        logger.info("═══ [Node] Reviewer ═══")
        try:
            items = state.get("naturalized_faqs", [])
            reviewed = agent_reviewer.run(items, client, model_name)
            approved = [r for r in reviewed if r["is_approved"]]
            logger.info("Reviewer: %d/%d approved", len(approved), len(reviewed))
            save_json(reviewed, "reviewed_faqs.json")
            return {
                "reviewed_faqs": reviewed,
                "approved_faqs": approved,
            }
        except Exception as e:
            logger.error("[Node] Reviewer failed: %s", e)
            return {"errors": state.get("errors", []) + [f"reviewer: {e}"]}

    def node_retry(state: GraphState) -> dict:
        """Node 9: Retry"""
        logger.info("═══ [Node] Retry ═══")
        try:
            reviewed = state.get("reviewed_faqs", [])
            to_retry = [r for r in reviewed if not r["is_approved"] and not r.get("is_retry")]
            logger.info("Retry: %d items", len(to_retry))

            # rewrite
            rewritten = []
            # Define rewrite function for each agent
            rewrite_fn = {
                    "faq_generator": agent_faq_generator.rewrite,
                    "adversarial"  : agent_adversarial.rewrite,
                    "multichunk"   : agent_multichunk_synthesis.rewrite,
                    "naturalizer"  : agent_naturalizer.rewrite,
            }
            for item in to_retry:
                agent = item.get("source_agent", "faq_generator")
                updated = rewrite_fn.get(agent, agent_faq_generator.rewrite)(item, client, model_name)
                if updated:
                    rewritten.append(updated)
                    logger.debug("Retry: rewrote %s via %s", item.get("id"), agent)

            logger.info("Retry: %d items rewritten → back to Reviewer", len(rewritten))
            return {"naturalized_faqs": rewritten}
        except Exception as e:
            logger.error("[Node] Retry failed: %s", e)
            return {"errors": state.get("errors", []) + [f"retry: {e}"]}

    def node_coverage_gate(state: GraphState) -> dict:
        """Node 10: Coverage Gate"""
        logger.info("═══ [Node] Coverage Gate ═══")
        try:
            approved = state.get("approved_faqs", [])
            enriched_chunks = state.get("enriched_chunks", [])
            
            valid_chunk_ids = {c['id'] for c in enriched_chunks if c.get("term")}
            covered_ids = {f["source_chunk_id"] for f in approved}

            coverage = len(covered_ids & valid_chunk_ids) / max(len(valid_chunk_ids), 1)
            
            logger.info(
                "Coverage: %.1f%% (%d/%d chunks covered)",
                coverage * 100, len(covered_ids & valid_chunk_ids), len(valid_chunk_ids),
            )
            
            return {"coverage": coverage}
        except Exception as e:
            logger.error("[Node] Coverage Gate failed: %s", e)
            return {"errors": state.get("errors", []) + [f"coverage_gate: {e}"]}
            
    def node_evaluator(state: GraphState) -> dict:
        """Node 11: Evaluator"""
        logger.info("═══ [Node] Evaluator ═══")
        try:
            approved = state.get("approved_faqs", [])
            
            report = agent_evaluator.run(
                approved, state.get("enriched_chunks", []), client, model_name
            )
            save_json(report, "evaluator_report.json")
            logger.info("   [Node] Saved evaluator report")

            summary = report.to_dict()["summary"]
            logger.info(
                "Evaluation: faith=%.3f | relev=%.2f | indep=%.0f%% | div=%.3f | cov=%.3f | prec=%.3f",
                summary.get("faithfulness_avg", 0),
                summary.get("answer_relevance_avg", 0),
                summary.get("context_independence_rate", 0) * 100,
                summary.get("diversity_score", 0),
                summary.get("context_coverage", 0),
                summary.get("retrieval_effectiveness", 0),
            )

            return {"eval_summary": summary}
        except Exception as e:
            logger.error("[Node] Evaluator failed: %s", e)
            return {"errors": state.get("errors", []) + [f"evaluator: {e}"]}

    def node_output(state: GraphState) -> dict:
        """Node 12: Output — save JSON + Markdown"""
        logger.info("═══ [Node] Output ══")
        final = state.get("approved_faqs", [])
        save_json(final, "faq_final.json")
        save_markdown(final)
        logger.info("   [Node] Saved %d FAQs", len(final))

        stats = {
            "total_raw": len(state.get("raw_faqs", [])),
            "total_final": len(final),
            "errors": state.get("errors", []),
        }
        logger.info("[Node] Pipeline done. %d FAQs saved.", len(final))
        return {"stats": stats}

    return {
        "chunker": node_chunker,
        "extractor": node_extractor,
        "faq_generator": node_faq_generator,
        "adversarial_generator": node_adversarial,
        "multichunk_generator": node_multichunk_synthesis,
        "merge_dedup": node_merge_dedup,
        "naturalizer": node_naturalizer,
        "reviewer": node_reviewer,
        "node_retry": node_retry,
        "coverage_gate": node_coverage_gate,
        "evaluator": node_evaluator,
        "output": node_output,
    }
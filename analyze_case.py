"""
End-to-end CLI for the Judge case-analysis pipeline.

Usage:
    uv run analyze_case.py Evidence/fir.pdf Evidence/witness_statement.docx
    uv run analyze_case.py Evidence/*.pdf --output report.json

On first run (or whenever ./chroma_db is empty) this will also ingest the
three national statutes in NationalDocs/ into the shared vector store, so
the legal-reasoning stage has something to retrieve against.
"""

import argparse
import glob
import json
import logging
import os

from dotenv import load_dotenv

from src import JudgeFacade, RAGFacade

NATIONAL_DOCS_DIR = os.path.join(os.path.dirname(__file__), "NationalDocs")


def ensure_national_law_ingested(judge: JudgeFacade) -> None:
    if judge.national_law_chunk_count() > 0:
        return

    statute_paths = sorted(glob.glob(os.path.join(NATIONAL_DOCS_DIR, "*.pdf")))
    if not statute_paths:
        logging.warning("No statute PDFs found in %s — legal reasoning will have nothing to cite.", NATIONAL_DOCS_DIR)
        return

    logging.info("Vector store is empty. Ingesting %d national statute(s)...", len(statute_paths))
    total_chunks = judge.ingest_national_law(statute_paths)
    logging.info("Ingested %d statute chunk(s).", total_chunks)


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run the full Judge case-analysis pipeline on evidence documents.")
    parser.add_argument("evidence", nargs="+", help="Paths to evidence PDFs/DOCX/images (glob-expanded).")
    parser.add_argument("--output", "-o", default=None, help="Optional path to write the full JSON report.")
    parser.add_argument("--top-k", type=int, default=6, help="How many statute chunks to retrieve for reasoning.")
    args = parser.parse_args()

    evidence_paths = []
    for pattern in args.evidence:
        matches = glob.glob(pattern)
        evidence_paths.extend(matches if matches else [pattern])

    judge = JudgeFacade(rag_facade=RAGFacade(), statute_top_k=args.top_k)
    ensure_national_law_ingested(judge)

    report = judge.analyze_case(evidence_paths)

    print("\n================ PREDICATES ================")
    for i, p in enumerate(report.predicates):
        print(f"[{i}] {p.as_fact_string()}  (confidence={p.confidence:.2f}, source={p.source_document})")

    print("\n================ TIMELINE ====================")
    for event in report.timeline:
        marker = event.parsed_time.isoformat() if event.parsed_time else f"(unparsed: {event.raw_time})"
        print(f"{marker} — {event.description}")

    print("\n================ CONTRADICTIONS ==============")
    if not report.contradictions:
        print("(none detected)")
    for c in report.contradictions:
        print(f"[{c.severity.upper()}/{c.contradiction_type}] {c.description}  -> predicates {c.predicate_indices}")

    print("\n================ LEGAL ANALYSIS ==============")
    print(f"Case summary: {report.legal_analysis.case_summary}\n")
    print(f"Legal query used for retrieval: {report.legal_analysis.legal_query}\n")
    if not report.legal_analysis.applicable_provisions:
        print("No applicable provisions could be grounded in the retrieved statute excerpts.")
    for provision in report.legal_analysis.applicable_provisions:
        print(f"- {provision.act} — {provision.section} ({provision.title or 'untitled'})")
        print(f"    reasoning: {provision.reasoning}")
        print(f"    confidence: {provision.confidence:.2f}")
    if report.legal_analysis.caveats:
        print("\nCaveats:")
        for caveat in report.legal_analysis.caveats:
            print(f"  - {caveat}")

    if args.output:
        with open(args.output, "w") as f:
            f.write(report.model_dump_json(indent=2))
        print(f"\nFull report written to {args.output}")


if __name__ == "__main__":
    main()

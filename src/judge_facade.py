import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from .factory.parse_factory import DocumentLoaderFactory
from .models import Predicate, CaseReport
from .chunkers import RecursiveChunker
from .processor import EvidenceProcessor
from .llm import LLMFactory
from .extraction import PredicateExtractor
from .analysis import TimelineBuilder, ContradictionDetector
from .legal import LegalQueryBuilder, LegalReasoner
from .rag_facade import RAGFacade

logger = logging.getLogger(__name__)


class JudgeFacade:
    """
    Top-level orchestrator for the case-analysis pipeline, built on
    LangChain throughout:

        evidence PDFs/DOCX/images
            -> LangChain document loaders (parallel, per document)
            -> RecursiveCharacterTextSplitter (small, bounded chunks)
            -> LCEL predicate-extraction chain, run via `.batch()`
               (parallel, per chunk)
            -> deterministic timeline construction (plain Python)
            -> LCEL contradiction-detection chain, run via `.batch()`
               (parallel, per predicate-batch)
            -> LCEL legal-query-formulation chain
            -> retrieval against the ingested national statutes
               (langchain_chroma.Chroma, via RAGFacade)
            -> LCEL legal-reasoning chain, grounded in the retrieved
               excerpts
            -> CaseReport

    Every LLM-calling stage is a `prompt | llm.with_structured_output(Schema)`
    LCEL chain; this class only wires them together. Each stage is
    independently testable and swappable (see src/extraction, src/analysis,
    src/legal, src/llm).
    """

    def __init__(
        self,
        rag_facade: Optional[RAGFacade] = None,
        llm: Optional[BaseChatModel] = None,
        evidence_chunk_size: int = 600,
        evidence_chunk_overlap: int = 80,
        max_parse_workers: int = 4,
        max_extraction_concurrency: int = 6,
        statute_top_k: int = 6,
    ):
        self._rag_facade = rag_facade or RAGFacade()
        self._llm = llm or LLMFactory.create()
        self._loader_factory = DocumentLoaderFactory()

        # Deliberately smaller chunks than the national-law chunker: the
        # smaller the window handed to the extraction LLM, the less room
        # it has to hallucinate facts that aren't actually in that window.
        evidence_chunker = RecursiveChunker(
            chunk_size=evidence_chunk_size,
            chunk_overlap=evidence_chunk_overlap,
        )
        self._evidence_processor = EvidenceProcessor(evidence_chunker)

        self._predicate_extractor = PredicateExtractor(self._llm, max_concurrency=max_extraction_concurrency)
        self._timeline_builder = TimelineBuilder()
        self._contradiction_detector = ContradictionDetector(self._llm)
        self._query_builder = LegalQueryBuilder(self._llm)
        self._legal_reasoner = LegalReasoner(self._llm)

        self._max_parse_workers = max_parse_workers
        self._statute_top_k = statute_top_k

    # ------------------------------------------------------------------
    # National law corpus (ingested once, queried per case)
    # ------------------------------------------------------------------
    def ingest_national_law(self, file_paths: List[str]) -> int:
        """Ingest one or more statute PDFs into the shared vector store."""
        total = 0
        for path in file_paths:
            total += self._rag_facade.ingest_document(path)
        return total

    def national_law_chunk_count(self) -> int:
        """How many statute chunks are currently in the vector store."""
        return self._rag_facade.count()

    # ------------------------------------------------------------------
    # Evidence ingestion + parsing (parallel across documents)
    # ------------------------------------------------------------------
    def _load_evidence_chunks(self, evidence_paths: List[str]) -> List[Document]:
        all_chunks: List[Document] = []

        with ThreadPoolExecutor(max_workers=self._max_parse_workers) as executor:
            future_to_path = {
                executor.submit(self._load_and_chunk_one, path): path for path in evidence_paths
            }
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    all_chunks.extend(future.result())
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to parse/chunk evidence document %s: %s", path, exc)

        return all_chunks

    def _load_and_chunk_one(self, file_path: str) -> List[Document]:
        filename = Path(file_path).name
        pages = self._loader_factory.load(file_path)
        return self._evidence_processor.process(pages, filename)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def analyze_case(self, evidence_paths: List[str]) -> CaseReport:
        if not evidence_paths:
            raise ValueError("analyze_case requires at least one evidence document path.")

        logger.info("Parsing %d evidence document(s)...", len(evidence_paths))
        chunks = self._load_evidence_chunks(evidence_paths)
        logger.info("Produced %d evidence chunk(s). Extracting predicates...", len(chunks))

        predicates: List[Predicate] = self._predicate_extractor.extract_from_chunks(chunks)
        logger.info("Extracted %d predicate(s). Building timeline...", len(predicates))

        timeline = self._timeline_builder.build(predicates)

        logger.info("Detecting contradictions...")
        contradictions = self._contradiction_detector.detect(predicates)

        logger.info("Formulating legal query...")
        query_output = self._query_builder.build(predicates, timeline, contradictions)

        logger.info("Retrieving relevant statute sections for: %s", query_output.legal_query)
        retrieved_chunks = self._rag_facade.query(query_output.legal_query, top_k=self._statute_top_k)

        logger.info("Running final legal reasoning...")
        legal_analysis = self._legal_reasoner.reason(
            case_summary=query_output.case_summary,
            legal_query=query_output.legal_query,
            predicates=predicates,
            timeline=timeline,
            contradictions=contradictions,
            retrieved_chunks=retrieved_chunks,
        )

        return CaseReport(
            evidence_documents=[Path(p).name for p in evidence_paths],
            predicates=predicates,
            timeline=timeline,
            contradictions=contradictions,
            legal_analysis=legal_analysis,
        )

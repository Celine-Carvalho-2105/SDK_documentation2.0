"""
pipeline.py
Central orchestrator — coordinates all agents through the documentation pipeline.
"""

import logging
from typing import List, Dict, Callable, Optional

from utils.llm_client import LLMClient
from rag.retriever import VectorStore, chunk_file
from agents.analyzer import AnalyzerAgent
from agents.doc_generator import DocGeneratorAgent
from agents.example_generator import ExampleGeneratorAgent
from agents.validator import ValidatorAgent
from output.formatter import format_markdown, format_html, format_json

logger = logging.getLogger(__name__)


class DocumentationPipeline:
    """
    Orchestrates the full documentation generation pipeline:
    Ingest → Chunk → Embed → Analyze → Generate → Validate → Format
    """

    def __init__(self, groq_api_key: str):
        self.llm = LLMClient(api_key=groq_api_key)
        self.vector_store = VectorStore()

        # Initialize agents
        self.analyzer = AnalyzerAgent(self.llm, self.vector_store)
        self.doc_generator = DocGeneratorAgent(self.llm, self.vector_store)
        self.example_generator = ExampleGeneratorAgent(self.llm, self.vector_store)
        self.validator = ValidatorAgent(self.llm)

    def run(
        self,
        files: List[Dict],
        project_name: str,
        output_format: str = "markdown",
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> str:
        """
        Execute the full pipeline.

        Args:
            files: List of file dicts with path, content, extension
            project_name: Name of the project
            output_format: 'markdown', 'html', or 'json'
            progress_callback: fn(message: str, percent: int)

        Returns:
            Generated documentation string in requested format
        """

        def update(msg: str, pct: int):
            logger.info(f"[{pct}%] {msg}")
            if progress_callback:
                progress_callback(msg, pct)

        # ── Step 1: Chunk all files ─────────────────────────────────────────
        update("Chunking source files for analysis...", 5)
        all_chunks = []
        for f in files:
            chunks = chunk_file(f)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No content could be extracted from the provided files.")

        update(f"Created {len(all_chunks)} content chunks from {len(files)} files.", 10)

        # ── Step 2: Build vector store (RAG) ───────────────────────────────
        update("Building embedding index (RAG)...", 15)

        def embed_progress(done, total):
            pct = 15 + int((done / total) * 15)
            update(f"Embedding chunks: {done}/{total}", pct)

        self.vector_store.build(all_chunks, progress_callback=embed_progress)
        update("Vector index ready.", 30)

        # ── Step 3: Analyze project ─────────────────────────────────────────
        update("Analyzer Agent: Understanding project structure...", 35)
        analysis = self.analyzer.analyze(files, project_name)
        update(f"Analysis complete: {analysis.get('architecture', '?')} project detected.", 42)

        # ── Step 4: Generate documentation sections ─────────────────────────
        sections: Dict[str, str] = {}

        update("Doc Generator: Writing project overview...", 45)
        sections["overview"] = self.doc_generator.generate_overview(analysis)

        update("Doc Generator: Writing installation guide...", 52)
        sections["installation"] = self.doc_generator.generate_installation(analysis, files)

        update("Doc Generator: Writing usage guide...", 58)
        sections["usage_guide"] = self.doc_generator.generate_usage_guide(analysis)

        update("Doc Generator: Writing API reference...", 64)
        sections["api_docs"] = self.doc_generator.generate_api_docs(analysis)

        update("Doc Generator: Writing architecture docs...", 70)
        sections["architecture"] = self.doc_generator.generate_architecture_doc(analysis)

        # ── Step 5: Example generator ───────────────────────────────────────
        update("Example Generator: Creating usage examples...", 76)
        sections["examples"] = self.example_generator.generate_examples(analysis)

        # ── Step 6: Optional configuration docs ────────────────────────────
        update("Doc Generator: Writing configuration docs...", 80)
        config_doc = self.doc_generator.generate_configuration_doc(analysis, files)
        if config_doc:
            sections["configuration"] = config_doc

        # ── Step 7: Validator agent ─────────────────────────────────────────
        update("Validator: Reviewing and improving documentation quality...", 84)
        sections = self.validator.validate_and_improve(sections, analysis)

        # ── Step 8: Add supplementary sections ─────────────────────────────
        update("Adding contributing guide and changelog...", 90)
        sections["contributing"] = self.validator.generate_contributing_guide(analysis)
        sections["changelog"] = self.validator.generate_changelog_stub(analysis)

        # ── Step 9: Format output ───────────────────────────────────────────
        update(f"Formatting output as {output_format.upper()}...", 95)
        fmt = output_format.lower()
        if fmt == "html":
            result = format_html(sections, analysis)
        elif fmt == "json":
            result = format_json(sections, analysis)
        else:
            result = format_markdown(sections, analysis)

        update("Documentation generation complete! ✅", 100)
        return result
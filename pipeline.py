"""
pipeline.py
Central orchestrator for the documentation pipeline.
"""

import logging
from typing import Callable, Dict, List, Optional

from agents.analyzer import AnalyzerAgent
from agents.doc_generator import DocGeneratorAgent
from output.formatter import format_html, format_json, format_markdown
from rag.retriever import VectorStore, chunk_file
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class DocumentationPipeline:
    """
    Orchestrates a rate-limit-friendly documentation pipeline:
    Ingest -> Chunk -> Embed -> Analyze -> Generate -> Format
    """

    def __init__(self, groq_api_key: str):
        self.llm = LLMClient(api_key=groq_api_key)
        self.vector_store = VectorStore()

        self.analyzer = AnalyzerAgent(self.llm, self.vector_store)
        self.doc_generator = DocGeneratorAgent(self.llm, self.vector_store)

    def run(
        self,
        files: List[Dict],
        project_name: str,
        output_format: str = "markdown",
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> str:
        """
        Execute the pipeline.

        Args:
            files: List of file dicts with path, content, extension
            project_name: Name of the project
            output_format: 'markdown', 'html', or 'json'
            progress_callback: fn(message: str, percent: int)

        Returns:
            Generated documentation string in requested format.
        """

        def update(msg: str, pct: int):
            logger.info("[%s%%] %s", pct, msg)
            if progress_callback:
                progress_callback(msg, pct)

        update("Chunking source files for analysis...", 5)
        all_chunks = []
        for file_info in files:
            all_chunks.extend(chunk_file(file_info))

        if not all_chunks:
            raise ValueError("No content could be extracted from the provided files.")

        update(f"Created {len(all_chunks)} content chunks from {len(files)} files.", 10)

        update("Building embedding index (RAG)...", 15)

        def embed_progress(done, total):
            pct = 15 + int((done / total) * 15)
            update(f"Embedding chunks: {done}/{total}", pct)

        self.vector_store.build(all_chunks, progress_callback=embed_progress)
        update("Vector index ready.", 30)

        update("Analyzer Agent: Understanding project structure...", 35)
        analysis = self.analyzer.analyze(files, project_name)
        update(f"Analysis complete: {analysis.get('architecture', '?')} project detected.", 42)

        sections: Dict[str, str] = {}

        update("Doc Generator: Writing project overview...", 48)
        sections["overview"] = self.doc_generator.generate_overview(analysis)

        update("Doc Generator: Writing installation guide...", 58)
        sections["installation"] = self.doc_generator.generate_installation(analysis, files)

        update("Doc Generator: Writing usage guide...", 68)
        sections["usage_guide"] = self.doc_generator.generate_usage_guide(analysis)

        update("Doc Generator: Writing architecture docs...", 78)
        sections["architecture"] = self.doc_generator.generate_architecture_doc(analysis)

        # API docs are useful but token-heavy. Keep them, but make failures non-fatal.
        update("Doc Generator: Writing compact API reference...", 86)
        try:
            sections["api_docs"] = self.doc_generator.generate_api_docs(analysis)
        except RuntimeError as exc:
            logger.warning("Skipping API docs after LLM error: %s", exc)
            sections["api_docs"] = _api_docs_fallback(analysis)

        update("Adding local contributing guide and changelog...", 92)
        sections["contributing"] = _contributing_fallback(analysis)
        sections["changelog"] = _changelog_fallback(analysis)

        update(f"Formatting output as {output_format.upper()}...", 96)
        fmt = output_format.lower()
        if fmt == "html":
            result = format_html(sections, analysis)
        elif fmt == "json":
            result = format_json(sections, analysis)
        else:
            result = format_markdown(sections, analysis)

        update("Documentation generation complete!", 100)
        return result


def _api_docs_fallback(analysis: Dict) -> str:
    components = analysis.get("key_components", [])
    lines = ["## API Reference", ""]

    if not components:
        lines.append("<!-- not determinable from code -->")
        return "\n".join(lines)

    lines.extend(["| Component | File | Description |", "|-----------|------|-------------|"])
    for component in components[:8]:
        lines.append(
            "| {name} | `{file}` | {description} |".format(
                name=component.get("name", "unknown"),
                file=component.get("file", "unknown"),
                description=component.get("description", "<!-- not determinable from code -->"),
            )
        )

    return "\n".join(lines)


def _contributing_fallback(analysis: Dict) -> str:
    project_name = analysis.get("project_name", "this project")
    language = analysis.get("language", "the project language")

    return f"""## Contributing

Contributions to {project_name} are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make focused changes that match the existing {language} style.
4. Run the project's tests or smoke checks before submitting.
5. Open a pull request with a short summary of the change.
"""


def _changelog_fallback(analysis: Dict) -> str:
    project_name = analysis.get("project_name", "the project")
    description = analysis.get("description", "Initial generated documentation.")

    return f"""## Changelog

### v1.0.0 - Initial Release
- Initial documentation for {project_name}.
- {description}

> Update this changelog with real version history before release.
"""

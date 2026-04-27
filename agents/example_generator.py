"""
agents/example_generator.py
Generates realistic usage examples for the documented project.
"""

import logging
from typing import Dict, List
from utils.llm_client import LLMClient
from rag.retriever import VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert developer who writes clear, realistic, and runnable code examples.
Examples must be complete, correct, and actually demonstrate the library/tool being documented.
Use the actual API from the code — don't invent methods."""


class ExampleGeneratorAgent:
    """Creates practical code examples for the documentation."""

    def __init__(self, llm: LLMClient, vector_store: VectorStore):
        self.llm = llm
        self.vs = vector_store

    def generate_examples(self, analysis: Dict) -> str:
        # Retrieve code that shows how things are called
        ctx = self.vs.retrieve(
            f"example usage import function call {analysis.get('project_name', '')}",
            top_k=8,
        )
        context_text = "\n\n---\n\n".join(
            f"[{c['source']}]\n{c['text']}" for c in ctx
        )

        lang = analysis.get("language", "python").lower()
        code_lang = _detect_code_lang(lang)

        prompt = f"""Create a comprehensive Examples section for "{analysis['project_name']}".

PROJECT INFO:
- Language: {lang}
- Framework: {analysis.get('framework', 'none')}
- Features: {', '.join(analysis.get('features', [])[:6])}

ACTUAL CODE TO BASE EXAMPLES ON:
{context_text}

Write in Markdown:
## Examples

Create 3-5 realistic, complete examples that show different use cases.
Each example must:
1. Have a descriptive title as ### heading
2. Explain what it demonstrates in 1-2 sentences  
3. Show complete, runnable {code_lang} code
4. Add brief comments in the code explaining key lines

Examples should progress from basic to advanced.
Base examples on the ACTUAL code context provided — don't invent fake APIs."""

        return self.llm.simple_prompt(prompt, system=SYSTEM_PROMPT, max_tokens=1500)


def _detect_code_lang(lang: str) -> str:
    mapping = {
        ".py": "python", "python": "python",
        ".js": "javascript", "javascript": "javascript",
        ".ts": "typescript", "typescript": "typescript",
        ".go": "go", "go": "go",
        ".rs": "rust", "rust": "rust",
        ".java": "java", "java": "java",
        ".rb": "ruby", "ruby": "ruby",
        ".cpp": "cpp", "c++": "cpp",
    }
    return mapping.get(lang, lang)

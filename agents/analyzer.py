"""
agents/analyzer.py
Analyzes the codebase structure and produces a high-level understanding.
"""

import logging
from typing import List, Dict, Optional
from utils.llm_client import LLMClient
from rag.retriever import VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior software architect. Your job is to analyze source code 
and provide clear, accurate technical summaries. Be concise but thorough. 
Focus on: purpose, architecture patterns, key components, dependencies, and entry points."""


class AnalyzerAgent:
    """Understands the project structure and produces an analysis report."""

    def __init__(self, llm: LLMClient, vector_store: VectorStore):
        self.llm = llm
        self.vs = vector_store

    def _build_file_tree(self, files: List[Dict]) -> str:
        paths = sorted(f["path"] for f in files)
        return "\n".join(f"  {p}" for p in paths[:100])

    def _get_key_files(self, files: List[Dict]) -> List[Dict]:
        """Prioritize entry points, configs, and READMEs."""
        priority = []
        secondary = []
        for f in files:
            name = f["path"].lower()
            if any(kw in name for kw in ["readme", "main", "app", "index", "setup", "config", "__init__"]):
                priority.append(f)
            else:
                secondary.append(f)
        return (priority + secondary)[:12]

    def analyze(self, files: List[Dict], project_name: str) -> Dict:
        """
        Run analysis and return a structured analysis dict.
        """
        file_tree = self._build_file_tree(files)
        key_files = self._get_key_files(files)

        # Build a summary of key file contents
        file_summaries = []
        for f in key_files:
            preview = f["content"][:1500]
            file_summaries.append(f"### {f['path']}\n```\n{preview}\n```")

        key_content = "\n\n".join(file_summaries)

        prompt = f"""Analyze this software project called "{project_name}".

FILE STRUCTURE:
{file_tree}

KEY FILES:
{key_content}

Provide a comprehensive analysis in this JSON structure (respond with ONLY valid JSON):
{{
  "project_name": "{project_name}",
  "description": "One paragraph describing what this project does",
  "language": "Primary programming language(s)",
  "framework": "Main framework or library if any",
  "architecture": "Architecture pattern (MVC, microservices, library, CLI, etc.)",
  "entry_points": ["list of main entry point files"],
  "key_components": [
    {{"name": "ComponentName", "description": "what it does", "file": "path/to/file"}}
  ],
  "dependencies": ["list of key external dependencies detected"],
  "features": ["list of main features or capabilities"],
  "complexity": "simple|moderate|complex",
  "documentation_strategy": "How to best document this project"
}}"""

        raw = self.llm.simple_prompt(prompt, system=SYSTEM_PROMPT, max_tokens=1200)

        # Parse JSON robustly
        import json, re
        # Strip markdown code blocks if present
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        try:
            analysis = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON object
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    analysis = json.loads(match.group())
                except Exception:
                    analysis = self._fallback_analysis(project_name, files)
            else:
                analysis = self._fallback_analysis(project_name, files)

        # Ensure project_name is set
        analysis["project_name"] = project_name
        analysis["total_files"] = len(files)
        analysis["file_tree"] = file_tree

        logger.info(f"Analysis complete: {analysis.get('architecture', 'unknown')} project")
        return analysis

    def _fallback_analysis(self, project_name: str, files: List[Dict]) -> Dict:
        """Minimal fallback if LLM parsing fails."""
        exts = {}
        for f in files:
            ext = f.get("extension", "unknown")
            exts[ext] = exts.get(ext, 0) + 1
        primary_lang = max(exts, key=exts.get) if exts else "unknown"

        return {
            "project_name": project_name,
            "description": f"A software project with {len(files)} files.",
            "language": primary_lang,
            "framework": "unknown",
            "architecture": "unknown",
            "entry_points": [],
            "key_components": [],
            "dependencies": [],
            "features": [],
            "complexity": "moderate",
            "documentation_strategy": "Generate documentation based on file contents.",
        }

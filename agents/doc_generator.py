"""
agents/doc_generator.py
Generates documentation sections using LLM + RAG context.
"""

import logging
from typing import Dict, List, Optional

from rag.retriever import VectorStore
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a strict, precise SDK documentation engine.

RULES - FOLLOW EXACTLY:
- Output ONLY Markdown. No prose introductions, no meta-commentary.
- Every claim MUST be derived from the provided code context. Do NOT invent APIs, parameters, or behaviors.
- If a detail is not present in the context, write: `<!-- not determinable from code -->` and move on.
- Never write generic sentences like "This module provides functionality" or "This is useful for developers".
- Be concise. Use bullet points and tables. Avoid paragraphs longer than 2 sentences.
- Use exact function/class/variable names as they appear in the source code.
- Do not hallucinate default values, types, or return types. Only state what the code confirms.
- Format every code example as a fenced code block with the correct language tag."""


class DocGeneratorAgent:
    """Generates comprehensive documentation from analysis and RAG context."""

    def __init__(self, llm: LLMClient, vector_store: VectorStore):
        self.llm = llm
        self.vs = vector_store

    def _retrieve_context(self, query: str, top_k: int = 6) -> str:
        chunks = self.vs.retrieve(query, top_k=top_k)
        if not chunks:
            return "No additional context available."

        parts = []
        for chunk in chunks:
            parts.append(f"[{chunk['source']}]\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def generate_overview(self, analysis: Dict) -> str:
        ctx = self._retrieve_context(
            f"entry point main module imports exports {analysis.get('project_name', '')}",
            top_k=5,
        )

        prompt = f"""Produce a concise SDK-style project overview for `{analysis['project_name']}`.

PROJECT METADATA:
{_fmt_analysis(analysis)}

SOURCE CONTEXT:
{ctx}

OUTPUT RULES:
- Do NOT write a blog post. Write reference documentation.
- Do NOT fabricate features. Only list what is confirmed by the metadata and context above.
- Use exact names from the code (module names, class names, CLI commands, etc.).

REQUIRED STRUCTURE - output exactly these sections, in this order:

# {analysis['project_name']}

> One sentence: what this project does and who it is for. Derived strictly from the code.

## Overview
- 3-6 bullet points describing concrete capabilities. Each bullet must reference a real module, class, or function from the context.

## Key Features
| Feature | Description |
|---------|-------------|
(Populate only from confirmed code context. Max 6 rows.)

## Tech Stack
| Component | Technology | Role |
|-----------|------------|------|
(Use exact dependency names from requirements/imports. Max 8 rows.)

## Architecture
- Bullet-point description of top-level modules and their responsibilities.
- One bullet per module/package. Use the exact directory/file names.

Do not add any section not listed above."""

        return self.llm.simple_prompt(prompt, system=SYSTEM_PROMPT, max_tokens=1200)

    def generate_installation(self, analysis: Dict, files: List[Dict]) -> str:
        setup_content = ""
        setup_keywords = [
            "requirements",
            "setup.py",
            "pyproject",
            "package.json",
            "go.mod",
            "cargo.toml",
            "gemfile",
        ]

        for file_info in files:
            if any(kw in file_info["path"].lower() for kw in setup_keywords):
                setup_content += (
                    f"\n### {file_info['path']}\n"
                    f"```\n{file_info['content'][:800]}\n```\n"
                )

        ctx = self._retrieve_context(
            "installation setup dependencies environment variables config",
            top_k=4,
        )

        prompt = f"""Generate a precise Installation & Setup reference for `{analysis['project_name']}`.

DETECTED LANGUAGE: {analysis.get('language', 'unknown')}
DETECTED FRAMEWORK: {analysis.get('framework', 'unknown')}
DETECTED DEPENDENCIES: {', '.join(analysis.get('dependencies', [])[:15]) or 'none detected'}

SETUP FILES FROM REPOSITORY:
{setup_content or '(No setup files detected in repository.)'}

ADDITIONAL CONTEXT:
{ctx}

OUTPUT RULES:
- Use only information derived from the setup files and context above.
- Do NOT invent install commands. Use the exact package names from the dependency files.
- If environment variables are referenced in the code, list each one explicitly.
- If no .env usage is detected, omit the Environment Variables section entirely.

REQUIRED STRUCTURE:

## Installation

### Prerequisites
- List runtime/tool prerequisites (e.g., Python >= 3.9, Node.js >= 18). Base version requirements on what the setup files specify.

### Install Dependencies
```(language)
(exact install commands derived from setup files)
```

### Environment Configuration
(Include ONLY if .env / os.getenv / environment variables are detected in context.)

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|

### Verify Installation
```(language)
(minimal command to confirm the install works - e.g., running --version or a smoke-test import)
```

Do not add any section not listed above."""

        return self.llm.simple_prompt(prompt, system=SYSTEM_PROMPT, max_tokens=1500)

    def generate_api_docs(self, analysis: Dict) -> str:
        ctx = self._retrieve_context(
            f"class def function method return type parameter signature decorator "
            f"{analysis.get('project_name', '')}",
            top_k=12,
        )

        prompt = f"""Generate complete SDK-style API Reference documentation for `{analysis['project_name']}`.

SOURCE CODE CONTEXT (ground truth - document ONLY what appears here):
{ctx}

OUTPUT RULES:
- Structure output STRICTLY as: File -> Class -> Method/Function.
- Document every class, function, and method visible in the context. Do not skip any.
- Do NOT document private members (names starting with `_`) unless they are explicitly exposed in a public API.
- Do NOT hallucinate parameter names, types, default values, or return types.
  If a type annotation is absent from the code, write the type as `unknown`.
- If a docstring exists in the code, quote it verbatim under the item. Do not paraphrase.
- If a detail cannot be determined from the context, write: `<!-- not determinable from code -->`.
- Every method/function entry MUST include a minimal usage example derived from actual call sites in the context,
  or constructed from the real signature if no call site is present.

REQUIRED FORMAT - follow this exact heading hierarchy:

## API Reference

---

## File: `path/to/file.py`

### Class: `ClassName`

> (One sentence from the class docstring or inferred strictly from its `__init__` and method signatures.)

**Constructor**
```python
ClassName(param1: type, param2: type = default)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| param1 | type | - | (from docstring or code) |

---

#### Method: `method_name()`

```python
def method_name(self, param: type) -> return_type
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| param | type | - | (from docstring or code) |

**Returns:** `return_type` - description.

**Raises:** `ExceptionType` - condition. (Omit this row if no exceptions are raised or documented.)

**Example**
```python
# minimal working example using real names from the code
instance = ClassName(...)
result = instance.method_name(...)
```

---

### Function: `function_name()`

(Same structure as Method above, omitting `self`.)

---

(Repeat File / Class / Function blocks for every file present in the context.)"""

        return self.llm.simple_prompt(prompt, system=SYSTEM_PROMPT, max_tokens=2048)

    def generate_architecture_doc(self, analysis: Dict) -> str:
        ctx = self._retrieve_context(
            "module imports pipeline flow data transformation class inheritance dependency injection",
            top_k=6,
        )

        prompt = f"""Generate a precise Architecture & Design reference for `{analysis['project_name']}`.

PROJECT METADATA:
{_fmt_analysis(analysis)}

SOURCE CONTEXT:
{ctx}

FILE TREE:
{analysis.get('file_tree', '(not available)')}

OUTPUT RULES:
- Do NOT write prose paragraphs. Use bullet points, tables, and short descriptions.
- Every module listed must correspond to an actual file/directory in the file tree or context.
- Do not invent design patterns. Only name a pattern if it is clearly implemented in the code.
- Data flow steps must reference real function/method names from the context.

REQUIRED STRUCTURE:

## Architecture & Design

### Module Responsibilities
| Module / File | Responsibility |
|---------------|----------------|
(One row per top-level file or package. Use exact names from the file tree.)

### Component Dependencies
- List which modules import from which. Use `A -> B` notation to mean "A depends on B".
- Derive strictly from import statements visible in the context.

### Data Flow
1. (Step 1: entry point - exact function/class name)
2. (Step 2: transformation - exact function/class name)
3. (Continue for each major step visible in the code.)

### Design Patterns
| Pattern | Where Applied | Evidence from Code |
|---------|---------------|--------------------|
(Only include patterns that are directly observable in the context. Omit table if none are confirmed.)

### Directory Structure
(Reproduce the file tree exactly as provided above. Do not modify it.)

Do not add any section not listed above."""

        return self.llm.simple_prompt(prompt, system=SYSTEM_PROMPT, max_tokens=2000)

    def generate_usage_guide(self, analysis: Dict) -> str:
        ctx = self._retrieve_context(
            f"example usage instantiation call invoke import main entry __main__ "
            f"{analysis.get('project_name', '')}",
            top_k=8,
        )

        prompt = f"""Generate a concise Usage Guide for `{analysis['project_name']}`.

PROJECT METADATA:
{_fmt_analysis(analysis)}

SOURCE CONTEXT (use real names, signatures, and patterns from here):
{ctx}

OUTPUT RULES:
- Every code example MUST use real class/function/variable names from the context.
- Do NOT fabricate method calls or import paths. If you cannot confirm a call from the context, omit it.
- Quick Start must be the minimal sequence of steps to produce a working result.
- Each use-case example must focus on a distinct capability confirmed by the source.
- Do not write motivational copy or filler sentences.

REQUIRED STRUCTURE:

## Usage Guide

### Quick Start
```(language)
# Minimal working example - every line must be derivable from the source context.
```

### Common Use Cases

#### (Use Case 1 Title - name it after the specific capability)
```(language)
# Example code using real APIs from the context
```
- (1-2 bullets explaining what this example does and any non-obvious behavior)

#### (Use Case 2 Title)
```(language)
# Example code
```
- (bullets)

(Add up to 4 use cases total. Only include a use case if it is confirmed by the context.)

### Configuration Options
(Include ONLY if configurable parameters/env vars are detected in the context.)

| Option | Type | Default | Effect |
|--------|------|---------|--------|

### Common Patterns & Pitfalls
- (Bullet list of real gotchas, ordering requirements, or non-obvious behaviors visible in the code.)
- (If none are determinable from the context, omit this section entirely.)

Do not add any section not listed above."""

        return self.llm.simple_prompt(prompt, system=SYSTEM_PROMPT, max_tokens=1500)

    def generate_configuration_doc(self, analysis: Dict, files: List[Dict]) -> Optional[str]:
        config_files = [
            file_info
            for file_info in files
            if any(
                kw in file_info["path"].lower()
                for kw in [".env", "config", "settings", ".yaml", ".yml", ".toml", ".ini"]
            )
        ]
        if not config_files:
            return None

        config_snippets = "\n".join(
            f"### {file_info['path']}\n```\n{file_info['content'][:600]}\n```"
            for file_info in config_files[:5]
        )

        prompt = f"""Generate a Configuration Reference for `{analysis.get('project_name', 'this project')}`.

CONFIGURATION FILES FROM REPOSITORY:
{config_snippets}

OUTPUT RULES:
- Document ONLY configuration keys/variables that appear in the files above.
- Do NOT invent keys or default values. If a default is not set in the file, write `-` in the Default column.
- If a key's purpose is not clear from its name or surrounding comments, write `<!-- purpose not determinable from code -->`.
- Group variables by the file they appear in.

REQUIRED STRUCTURE:

## Configuration

### `(filename)`

| Key | Type | Default | Required | Description |
|-----|------|---------|----------|-------------|
(One row per config key found in this file.)

(Repeat the table block for each config file.)

### Notes
- (Any cross-cutting constraints visible in the config files - e.g., mutual exclusivity, ordering, format requirements.)
- (Omit this section if no such constraints are present.)

Do not add any section not listed above."""

        return self.llm.simple_prompt(prompt, system=SYSTEM_PROMPT, max_tokens=1500)


def _fmt_analysis(analysis: Dict) -> str:
    lines = [
        f"Project: {analysis.get('project_name', 'unknown')}",
        f"Description: {analysis.get('description', '')}",
        f"Language: {analysis.get('language', '')}",
        f"Framework: {analysis.get('framework', '')}",
        f"Architecture: {analysis.get('architecture', '')}",
        f"Features: {', '.join(analysis.get('features', [])[:8])}",
        f"Key Components: {', '.join(c.get('name', '') for c in analysis.get('key_components', [])[:6])}",
        f"Complexity: {analysis.get('complexity', '')}",
    ]
    return "\n".join(lines)

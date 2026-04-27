"""
agents/validator.py
Validates and improves generated documentation quality.
"""

import logging
import re
from typing import Dict
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior technical documentation editor. 
Your job is to review documentation and improve it for:
- Accuracy and completeness
- Clarity and readability  
- Proper Markdown formatting
- Consistent style and tone
- Removal of placeholder text like [TODO] or [INSERT]
Return the improved documentation only, no meta-commentary."""


class ValidatorAgent:
    """Reviews and improves documentation quality."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def validate_and_improve(self, docs: Dict[str, str], analysis: Dict) -> Dict[str, str]:
        """
        Check each section for quality issues and improve where needed.
        Returns improved sections dict.
        """
        improved = {}
        project_name = analysis.get("project_name", "the project")

        for section_name, content in docs.items():
            if not content or len(content.strip()) < 50:
                improved[section_name] = content
                continue

            issues = self._detect_issues(content)
            if issues:
                logger.info(f"Improving section '{section_name}': {issues}")
                improved[section_name] = self._improve_section(
                    section_name, content, issues, project_name
                )
            else:
                improved[section_name] = content

        return improved

    def _detect_issues(self, content: str) -> list:
        """Detect quality issues in a documentation section."""
        issues = []

        # Check for placeholder text
        placeholders = re.findall(
            r"\[(?:TODO|FIXME|INSERT|PLACEHOLDER|TBD|ADD)\b.*?\]",
            content,
            re.IGNORECASE,
        )
        if placeholders:
            issues.append(f"Contains placeholders: {placeholders[:3]}")

        # Check for very short sections
        if len(content.strip()) < 100:
            issues.append("Content is too brief")

        # Check for broken markdown (unclosed code blocks)
        code_block_count = content.count("```")
        if code_block_count % 2 != 0:
            issues.append("Unclosed code block detected")

        # Check for repetitive content
        lines = content.split("\n")
        non_empty = [l.strip() for l in lines if l.strip()]
        if len(non_empty) != len(set(non_empty)) and len(non_empty) > 5:
            issues.append("Possible duplicate lines detected")

        return issues

    def _improve_section(
        self, section: str, content: str, issues: list, project_name: str
    ) -> str:
        issues_str = "; ".join(issues)
        prompt = f"""Review and improve this documentation section for "{project_name}".

SECTION: {section}
ISSUES FOUND: {issues_str}

CURRENT CONTENT:
{content}

Instructions:
- Fix all identified issues
- Fix any unclosed code blocks
- Remove placeholder text and replace with meaningful content
- Ensure Markdown is properly formatted
- Keep all accurate information intact
- Do NOT add fictional information — if you don't know something, omit it

Return ONLY the improved Markdown content."""

        try:
            return self.llm.simple_prompt(prompt, system=SYSTEM_PROMPT, max_tokens=1500)
        except Exception as e:
            logger.warning(f"Validation improvement failed for '{section}': {e}")
            return content  # Return original if improvement fails

    def generate_changelog_stub(self, analysis: Dict) -> str:
        """Generate a basic changelog/changelog stub."""
        return f"""## Changelog

### v1.0.0 — Initial Release
- Initial release of {analysis.get('project_name', 'the project')}
- {analysis.get('description', 'Core functionality implemented.')}

> This changelog was auto-generated. Update with actual version history.
"""

    def generate_contributing_guide(self, analysis: Dict) -> str:
        lang = analysis.get("language", "Python")
        prompt = f"""Generate a Contributing Guide for "{analysis.get('project_name', 'this project')}" 
(a {lang} project with {analysis.get('complexity', 'moderate')} complexity).

Write in Markdown with:
## Contributing

Sections:
1. How to report issues
2. Development setup
3. Pull request process
4. Code style guidelines (appropriate for {lang})
5. Testing guidelines

Keep it concise and welcoming."""

        try:
            return self.llm.simple_prompt(prompt, system=SYSTEM_PROMPT, max_tokens=700)
        except Exception:
            return f"""## Contributing

We welcome contributions to {analysis.get('project_name', 'this project')}!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes with tests
4. Submit a pull request

Please follow existing code style and include tests for new features.
"""

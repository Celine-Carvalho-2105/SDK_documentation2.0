"""
ingestion/ingestor.py
Handles all input types: ZIP files, individual files, Git repositories.
"""

import os
import zipfile
import tempfile
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import chardet

logger = logging.getLogger(__name__)

# File extensions considered as source code / documentation
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".md", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml", ".ini",
    ".cfg", ".xml", ".html", ".css", ".sh", ".bash", ".dockerfile",
    ".tf", ".sql", ".r", ".m", ".lua", ".dart", ".vue", ".svelte",
}

SKIP_DIRS = {
    ".git", ".svn", "__pycache__", "node_modules", ".venv", "venv",
    "env", ".env", "dist", "build", ".idea", ".vscode", "coverage",
    ".pytest_cache", ".mypy_cache", "*.egg-info",
}

MAX_FILE_SIZE_BYTES = 500_000  # 500 KB per file


def _safe_read_file(path: str) -> Optional[str]:
    """Read a file safely, detecting encoding."""
    try:
        file_size = os.path.getsize(path)
        if file_size > MAX_FILE_SIZE_BYTES:
            return f"[File too large to include: {file_size} bytes]"
        if file_size == 0:
            return None

        with open(path, "rb") as f:
            raw = f.read()

        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
        try:
            return raw.decode(encoding, errors="replace")
        except Exception:
            return raw.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return None


def _should_skip_dir(dir_name: str) -> bool:
    return dir_name in SKIP_DIRS or dir_name.startswith(".")


def _collect_files_from_dir(root: str) -> List[Dict]:
    """Walk directory and collect supported files."""
    collected = []
    root_path = Path(root)

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune directories in-place
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]

        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            full_path = os.path.join(dirpath, filename)
            rel_path = str(Path(full_path).relative_to(root_path))
            content = _safe_read_file(full_path)

            if content:
                collected.append({
                    "path": rel_path,
                    "content": content,
                    "extension": ext,
                    "size": os.path.getsize(full_path),
                })

    return collected


class Ingestor:
    """Central ingestion handler for all input types."""

    def __init__(self):
        self._temp_dirs: List[str] = []

    def _make_temp_dir(self) -> str:
        d = tempfile.mkdtemp(prefix="docgen_")
        self._temp_dirs.append(d)
        return d

    def cleanup(self):
        """Remove all temporary directories."""
        for d in self._temp_dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
        self._temp_dirs.clear()

    def ingest_zip(self, zip_bytes: bytes) -> Tuple[List[Dict], str]:
        """Extract ZIP and collect files."""
        tmp = self._make_temp_dir()
        zip_path = os.path.join(tmp, "upload.zip")

        with open(zip_path, "wb") as f:
            f.write(zip_bytes)

        extract_dir = os.path.join(tmp, "extracted")
        os.makedirs(extract_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile as e:
            raise ValueError(f"Invalid ZIP file: {e}")

        files = _collect_files_from_dir(extract_dir)
        if not files:
            raise ValueError("ZIP file contained no supported source files.")

        project_name = "uploaded_project"
        return files, project_name

    def ingest_files(self, uploaded_files: List) -> Tuple[List[Dict], str]:
        """Ingest individually uploaded Streamlit file objects."""
        collected = []
        for uf in uploaded_files:
            ext = Path(uf.name).suffix.lower()
            content_bytes = uf.read()
            if not content_bytes:
                continue

            detected = chardet.detect(content_bytes)
            encoding = detected.get("encoding") or "utf-8"
            try:
                content = content_bytes.decode(encoding, errors="replace")
            except Exception:
                content = content_bytes.decode("utf-8", errors="replace")

            collected.append({
                "path": uf.name,
                "content": content,
                "extension": ext,
                "size": len(content_bytes),
            })

        if not collected:
            raise ValueError("No readable files were uploaded.")

        return collected, "uploaded_files"

    def ingest_git(
        self,
        repo_url: str,
        branch: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Tuple[List[Dict], str]:
        """Clone a Git repository and collect files."""
        import git

        tmp = self._make_temp_dir()
        clone_dir = os.path.join(tmp, "repo")

        # Inject token for private repos
        effective_url = repo_url
        if token:
            # Support https://github.com/user/repo -> https://token@github.com/user/repo
            if "https://" in repo_url:
                effective_url = repo_url.replace("https://", f"https://{token}@")

        clone_kwargs = {"to_path": clone_dir, "depth": 1}
        if branch:
            clone_kwargs["branch"] = branch

        try:
            git.Repo.clone_from(effective_url, **clone_kwargs)
        except git.GitCommandError as e:
            err_msg = str(e)
            if "Authentication failed" in err_msg or "could not read" in err_msg.lower():
                raise ValueError(
                    "Git authentication failed. Provide a valid token for private repos."
                )
            elif "not found" in err_msg.lower() or "Repository not found" in err_msg:
                raise ValueError(f"Repository not found: {repo_url}")
            elif "Remote branch" in err_msg:
                raise ValueError(f"Branch '{branch}' not found in repository.")
            else:
                raise ValueError(f"Git clone failed: {err_msg}")

        files = _collect_files_from_dir(clone_dir)
        if not files:
            raise ValueError("Repository contained no supported source files.")

        # Derive project name from URL
        repo_name = repo_url.rstrip("/").split("/")[-1]
        repo_name = repo_name.replace(".git", "")

        return files, repo_name
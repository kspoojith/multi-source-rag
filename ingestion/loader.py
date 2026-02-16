"""
Document Loader
================
Reads raw files from the data/raw/ directory and converts them into
standardized Document objects.

Design decisions:
- We use a dataclass (not dict) for type safety and IDE support
- Each document carries metadata: source filename, detected topic, language
- The loader is format-agnostic — it dispatches to the right reader
  based on file extension
- We intentionally load full files (not chunked) because chunking
  is a separate concern handled by chunker.py
"""

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    Represents a single loaded document before chunking.

    Attributes:
        content:  The full text of the document
        metadata: Dict with keys like 'source', 'topic', 'language'
        doc_id:   Unique identifier (defaults to filename)
    """
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = self.metadata.get("source", "unknown")


# ─── Topic Detection (simple heuristic) ──────────────────────────────────────
# Maps keywords found in filenames/content to topic labels.
# This is used for metadata filtering during retrieval.
TOPIC_KEYWORDS = {
    "modi": "politics",
    "education": "education",
    "constitution": "law",
    "samvidhan": "law",
    "isro": "space",
    "space": "space",
    "chandrayaan": "space",
    "telangana": "geography",
    "hyderabad": "geography",
}


def detect_topic(filename: str, content: str) -> str:
    """
    Simple keyword-based topic detection from filename and content.
    For a real system, you'd use a classifier — but for our research,
    controlled topic labels let us measure retrieval by category.
    """
    text = (filename + " " + content[:500]).lower()
    for keyword, topic in TOPIC_KEYWORDS.items():
        if keyword in text:
            return topic
    return "general"


def load_text_file(filepath: Path) -> str:
    """Load a plain text file with UTF-8 encoding."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_json_file(filepath: Path) -> str:
    """
    Load a JSON file and convert it to readable text.
    Handles both single objects and arrays.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return "\n\n".join(json.dumps(item, ensure_ascii=False, indent=2) for item in data)
    return json.dumps(data, ensure_ascii=False, indent=2)


# Registry of file loaders by extension
FILE_LOADERS = {
    ".txt": load_text_file,
    ".json": load_json_file,
    ".md": load_text_file,
}


def load_documents(data_dir: str | Path) -> List[Document]:
    """
    Load all supported documents from a directory.

    Args:
        data_dir: Path to directory containing raw documents

    Returns:
        List of Document objects with content and metadata

    How it works:
    1. Scans the directory for supported file types (.txt, .json, .md)
    2. Reads each file using the appropriate loader
    3. Detects the topic from filename + content
    4. Wraps everything in a Document dataclass
    """
    data_dir = Path(data_dir)
    documents: List[Document] = []

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return documents

    supported_extensions = set(FILE_LOADERS.keys())

    for filepath in sorted(data_dir.iterdir()):
        if filepath.suffix.lower() not in supported_extensions:
            logger.debug(f"Skipping unsupported file: {filepath.name}")
            continue

        try:
            loader = FILE_LOADERS[filepath.suffix.lower()]
            content = loader(filepath)

            if not content.strip():
                logger.warning(f"Empty file skipped: {filepath.name}")
                continue

            topic = detect_topic(filepath.name, content)

            doc = Document(
                content=content,
                metadata={
                    "source": filepath.name,
                    "topic": topic,
                    "file_path": str(filepath),
                },
                doc_id=filepath.stem,  # filename without extension
            )
            documents.append(doc)
            logger.info(f"Loaded: {filepath.name} (topic={topic}, {len(content)} chars)")

        except Exception as e:
            logger.error(f"Failed to load {filepath.name}: {e}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents

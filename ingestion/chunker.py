"""
Text Chunker
=============
Splits documents into overlapping chunks for embedding.

Why chunking matters for RAG:
- A full document vector is a "bag of topics" — too generic for QA.
- Small chunks (200-500 chars) produce focused vectors that match
  specific questions better.
- Overlap (50 chars) prevents information loss at chunk boundaries.

Research note:
- We'll experiment with chunk sizes (300, 500, 800) to measure
  Recall@5 impact in our evaluation.
- Too small = low context. Too large = diluted embeddings.

Algorithm: Sliding Window
1. Clean whitespace / normalize
2. Walk through text in steps of (chunk_size - overlap)
3. Each window becomes a Chunk with inherited metadata
4. Discard chunks shorter than MIN_CHUNK_LENGTH
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List

# Import from sibling package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    A single chunk of text, ready for embedding.

    Attributes:
        text:      The chunk content
        chunk_id:  Unique ID: "{doc_id}_chunk_{index}"
        metadata:  Inherited from parent document + chunk-specific info
    """
    text: str
    chunk_id: str = ""
    metadata: dict = field(default_factory=dict)


def clean_text(text: str) -> str:
    """
    Normalize whitespace and remove artifacts.

    We intentionally do NOT strip Hindi/Telugu characters —
    the multilingual embedding model handles them natively.
    We only remove excessive whitespace and control characters.
    """
    # Replace multiple newlines with double newline (preserve paragraphs)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove control characters except newline
    text = re.sub(r'[\x00-\x09\x0b-\x0c\x0e-\x1f]', '', text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_length: int = MIN_CHUNK_LENGTH,
) -> List[str]:
    """
    Split text into overlapping chunks using a sliding window.

    Visual example (chunk_size=10, overlap=3):
        Text: "ABCDEFGHIJKLMNOP"
        Chunk 1: "ABCDEFGHIJ"    (positions 0-9)
        Chunk 2: "HIJKLMNOP"     (positions 7-15, overlaps "HIJ")

    Args:
        text:          Input text to chunk
        chunk_size:    Maximum characters per chunk
        chunk_overlap: Overlap between consecutive chunks
        min_length:    Minimum chunk length (shorter ones are discarded)

    Returns:
        List of text chunks
    """
    if not text or len(text) < min_length:
        return []

    text = clean_text(text)
    chunks = []
    step = chunk_size - chunk_overlap  # How far we advance each iteration

    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if len(chunk) >= min_length:
            chunks.append(chunk)

        # If we've consumed all text, stop
        if end >= len(text):
            break

    return chunks


def chunk_document(doc, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """
    Split a Document into a list of Chunks, preserving metadata.

    Each chunk inherits the parent document's metadata (source, topic)
    and gets additional chunk-specific metadata (chunk_index, char_start).

    Args:
        doc:           A Document object from loader.py
        chunk_size:    Characters per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of Chunk objects ready for embedding
    """
    text_chunks = chunk_text(doc.content, chunk_size, chunk_overlap)
    chunks = []

    step = chunk_size - chunk_overlap

    for i, text in enumerate(text_chunks):
        chunk = Chunk(
            text=text,
            chunk_id=f"{doc.doc_id}_chunk_{i}",
            metadata={
                **doc.metadata,              # Inherit document metadata
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "char_start": i * step,
                "doc_id": doc.doc_id,
            },
        )
        chunks.append(chunk)

    logger.info(
        f"Document '{doc.doc_id}' → {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def chunk_documents(documents: list, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """
    Chunk a list of documents into a flat list of Chunks.

    Args:
        documents:     List of Document objects
        chunk_size:    Characters per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        Flat list of all Chunks across all documents
    """
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    logger.info(f"Total: {len(documents)} documents → {len(all_chunks)} chunks")
    return all_chunks

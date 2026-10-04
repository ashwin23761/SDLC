"""
Reads reference/knowledge-base documents from the documents/ folder.

Selected files are converted into source documents, then src.rag chunks and
retrieves the most relevant passages for the Requirements Agent.
"""

import os
from typing import Dict, List

from .rag import SourceDocument

DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "documents")


def list_documents() -> List[str]:
    """Return the list of available reference document filenames."""
    if not os.path.isdir(DOCUMENTS_DIR):
        return []
    return sorted(
        f for f in os.listdir(DOCUMENTS_DIR)
        if f.lower().endswith((".md", ".txt"))
    )


def load_documents(filenames: List[str]) -> Dict[str, str]:
    """Load the content of the given filenames as {filename: text}."""
    contents = {}
    for name in filenames:
        path = os.path.join(DOCUMENTS_DIR, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                contents[name] = f.read()
    return contents


def load_source_documents(filenames: List[str]) -> List[SourceDocument]:
    """Load selected knowledge-base documents as retrievable sources."""
    docs = load_documents(filenames)
    return [
        SourceDocument(name=name, text=text, source_type="knowledge-base")
        for name, text in docs.items()
    ]


def format_documents_for_prompt(filenames: List[str]) -> str:
    """Return the selected documents formatted as labelled blocks for a prompt."""
    docs = load_documents(filenames)
    if not docs:
        return "No reference documents were provided."
    blocks = []
    for name, text in docs.items():
        blocks.append(f"--- START DOCUMENT: {name} ---\n{text.strip()}\n--- END DOCUMENT: {name} ---")
    return "\n\n".join(blocks)

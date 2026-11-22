"""
Reads reference/knowledge-base documents from the documents/ folder.

Selected files are converted into source documents, then src.rag chunks and
retrieves the most relevant passages for the Requirements Agent.
"""

import os
import json
from typing import Dict, List

from .rag import SourceDocument, extract_text_from_upload

DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "documents")
MANIFEST_PATH = os.path.join(DOCUMENTS_DIR, "manifest.json")


def _source_metadata(name: str) -> dict:
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        metadata = manifest.get(name, {})
    except (OSError, json.JSONDecodeError):
        metadata = {}
    required = ("jurisdiction", "effective_date", "version", "applicability", "owner")
    return {
        **metadata,
        "metadata_complete": all(metadata.get(key) for key in required),
        "missing_metadata": [key for key in required if not metadata.get(key)],
    }


def list_documents() -> List[str]:
    """Return the list of available reference document filenames."""
    if not os.path.isdir(DOCUMENTS_DIR):
        return []
    return sorted(
        f for f in os.listdir(DOCUMENTS_DIR)
        if f.lower() != "manifest.json" and f.lower().endswith((".md", ".txt", ".pdf", ".docx", ".csv", ".json", ".yaml", ".yml", ".xml", ".log", ".html", ".htm"))
    )


def load_documents(filenames: List[str]) -> Dict[str, str]:
    """Load the content of the given filenames as {filename: text}."""
    contents = {}
    for name in filenames:
        path = os.path.join(DOCUMENTS_DIR, name)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                contents[name] = extract_text_from_upload(name, f.read())
    return contents


def load_source_documents(filenames: List[str]) -> List[SourceDocument]:
    """Load selected knowledge-base documents as retrievable sources."""
    docs = load_documents(filenames)
    return [
        SourceDocument(name=name, text=text, source_type="knowledge-base", metadata=_source_metadata(name))
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

"""
Small helper around Ollama's REST API, used only to show status in the
sidebar (e.g. "Ollama is running and llama3.2:3b is available.").
"""

from typing import List, Tuple

import requests


def check_ollama(base_url: str, model_name: str, timeout: float = 2.0) -> Tuple[bool, bool, List[str]]:
    """
    Returns (server_running, model_available, all_model_names).
    Never raises: any connection problem is reported as server_running=False.
    """
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        names = [m.get("name", "") for m in data.get("models", [])]
        # Ollama model names look like "llama3.2:3b"; allow matching with or without tag.
        available = any(
            name == model_name or name.split(":")[0] == model_name.split(":")[0]
            for name in names
        )
        return True, available, names
    except requests.RequestException:
        return False, False, []

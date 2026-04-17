"""
Policy loader for the NASA Procedural Recall Assistant.

Reads policies/policy.yaml and returns it as a plain dict.  All pipeline
components (retrieval, LLM, safety scan, audit) pull their configuration
from this single source so governance decisions (model, thresholds, limits)
can be changed in one place without editing source code.

Key policy sections:
  llm         — Ollama model, temperature, timeout
  vlm         — vision rendering DPI, max images per query
  retrieval   — top_k, min_score threshold, max evidence characters
  routing     — per query-type response modes and fail-safe switch
  output_constraints — citation pattern, max response length, block patterns
  audit       — log path
"""

import yaml


def load_policy(path: str) -> dict:
    """Load and return the YAML policy file as a dict.  Raises if file is empty."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f.read())
    if data is None:
        raise ValueError(f"Policy file parsed as None (empty or invalid): {path}")
    return data
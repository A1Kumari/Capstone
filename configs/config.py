"""
configs/config.py
─────────────────────────────────────────────────────────────────
Central configuration loader.

Provides:
  LITELLM_MODEL        — LLM model used by all LiteLLM agents
  EMBEDDING_MODEL      — Embedding model used by the indexing agent
  RAG_UNKNOWN_RESPONSE — Fallback string when RAG has no answer
  ABBREVIATION_MAP     — Medical abbreviation dict from rules.yaml
  get_prompt(agent, key, **kwargs) — Fetch and format a system prompt
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_CONFIGS_DIR = Path(__file__).resolve().parent

# ── Model names (override via .env) ───────────────────────────────────────────
LITELLM_MODEL   = os.getenv("LITELLM_MODEL",   "gemini/gemini-2.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# ── Clinical constants ─────────────────────────────────────────────────────────
RAG_UNKNOWN_RESPONSE = "I don't know"

# ── Abbreviation map from rules.yaml ──────────────────────────────────────────
with open(_CONFIGS_DIR / "rules.yaml") as _f:
    _RULES = yaml.safe_load(_f) or {}
ABBREVIATION_MAP: dict = (
    _RULES.get("normalization_standards", {}).get("abbreviation_map", {})
)

# ── Prompts ────────────────────────────────────────────────────────────────────
with open(_CONFIGS_DIR / "prompts.yaml") as _f:
    _PROMPTS: dict = yaml.safe_load(_f)


def get_persona(agent: str) -> str:
    """Return the persona string for the given agent."""
    return _PROMPTS[agent]["persona"].strip()


def get_prompt(agent: str, key: str = "system", **kwargs) -> str:
    """Return the prompt for `agent`[`key`], formatted with any kwargs.

    {persona} is auto-injected from the agent's own persona field so callers
    don't need to pass it explicitly.
    """
    template: str = _PROMPTS[agent][key]
    if "{persona}" in template and "persona" not in kwargs:
        kwargs["persona"] = get_persona(agent)
    return template.format(**kwargs) if kwargs else template

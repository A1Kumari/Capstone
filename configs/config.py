import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_CONFIGS_DIR = Path(__file__).resolve().parent

# override via .env
FOUNDATION_MODEL_API_KEY = os.getenv("FOUNDATION_MODEL_API_KEY", "")
LLM_BASE_URL    = os.getenv("LLM_BASE_URL",    "https://llmgw-infy.tekstac.com/v1/")
EHR_BASE_URL    = os.getenv("EHR_BASE_URL",    "http://127.0.0.1:8000")
LITELLM_MODEL   = os.getenv("LITELLM_MODEL",   "openai/gpt-5.4-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

RAG_UNKNOWN_RESPONSE = "I don't know"

with open(_CONFIGS_DIR / "rules.yaml") as _f:
    _RULES = yaml.safe_load(_f) or {}
ABBREVIATION_MAP: dict = (
    _RULES.get("normalization_standards", {}).get("abbreviation_map", {})
)

with open(_CONFIGS_DIR / "prompts.yaml") as _f:
    _PROMPTS: dict = yaml.safe_load(_f)


def get_persona(agent: str) -> str:
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

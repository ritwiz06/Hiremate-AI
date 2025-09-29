"""Optional LLM helpers for resume parsing.

Supports two providers:
  • Ollama (local models, default: `mistral`)
  • Groq API (cloud, default: `llama3-8b-8192`)

Both are optional; if the dependency or API key is missing, the helpers
quietly fall back to heuristic parsing.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import requests

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

if TYPE_CHECKING:  # pragma: no cover
    from groq import Groq as GroqClient
    from groq.types.chat import ChatCompletionMessageParam
else:
    GroqClient = Any  # type: ignore
    ChatCompletionMessageParam = Dict[str, str]  # runtime fallback

try:  # Groq SDK is optional
    from groq import Groq as GroqSDK
except Exception:  # pragma: no cover - dependency missing
    GroqSDK = None  # type: ignore


_ENV_LOADED = False


def _load_local_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    _ENV_LOADED = True


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------


def _call_ollama(model: str, prompt: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        raw = data.get("response", "")
        logger.info("[llm] ollama raw response: %s", raw)
        return raw
    except Exception as err:
        logger.exception("[llm] ollama call failed: %s", err)
        return ""


def extract_name_via_ollama(text: str) -> Tuple[str, float]:
    model = os.getenv("OLLAMA_MODEL", "mistral")
    prompt = (
        "You are assisting a resume parser. Extract only the candidate's full name. "
        "If the name is unclear, return an empty string. Respond strictly as JSON with keys 'name' and 'confidence'.\n\n"
        f"Resume text:\n{text[:2000]}\n\nJSON response:"
    )

    raw = _call_ollama(model, prompt)
    if not raw.strip():
        return "", 0.0
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.info("[llm] ollama name response not JSON; skipping")
        return "", 0.0
    name = (data.get("name") or "").strip()
    confidence = float(data.get("confidence") or 0.0)
    return name, confidence


def refine_experience_via_ollama(raw_text: str, items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    model = os.getenv("OLLAMA_MODEL", "mistral")
    seed_json = json.dumps(items, ensure_ascii=False)
    prompt = (
        "Clean the following resume experience entries. For each entry, keep the original wording of the "
        "summary (concatenate the source lines verbatim). Respond with JSON only. Each object must have "
        "keys 'role', 'company', 'location', 'years', 'summary'. Leave values empty when unknown.\n\n"
        f"Resume text:\n{raw_text[:3000]}\n\n"
        f"Preliminary JSON:\n{seed_json}\n\nJSON:"
    )

    raw = _call_ollama(model, prompt)
    if not raw.strip():
        return items
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.info("[llm] ollama experience response not JSON; returning baseline")
        return items

    cleaned: List[Dict[str, str]] = []
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            cleaned.append(
                {
                    "role": entry.get("role", ""),
                    "company": entry.get("company", ""),
                    "location": entry.get("location", ""),
                    "years": entry.get("years", ""),
                    "summary": entry.get("summary", ""),
                }
            )
    return cleaned or items


# ---------------------------------------------------------------------------
# Groq helpers
# ---------------------------------------------------------------------------


def _get_groq_client() -> Optional[GroqClient]:
    _load_local_env()
    if GroqSDK is None:
        logger.info("[llm] groq SDK not installed; skip groq calls")
        return None
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.info("[llm] GROQ_API_KEY not set; skip groq calls")
        return None
    try:
        return GroqSDK(api_key=api_key)
    except Exception as err:
        logger.exception("[llm] failed to initialise Groq client: %s", err)
        return None


def _groq_chat(
    messages: List[ChatCompletionMessageParam],
    model: Optional[str] = None,
    expect_json: bool = False,
) -> str:
    client = _get_groq_client()
    if not client:
        return ""

    chosen_model = model or os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
    try:
        kwargs = {
            "model": chosen_model,
            "messages": messages,
            "temperature": 0,
        }
        if expect_json:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        logger.info("[llm] groq raw response (%s): %s", chosen_model, content)
        return content
    except Exception as err:
        logger.exception("[llm] groq chat call failed for %s: %s", chosen_model, err)
        return ""


def refine_experience_via_groq(
    raw_text: str,
    items: List[Dict[str, str]],
    model: Optional[str] = None,
) -> List[Dict[str, str]]:
    payload = json.dumps(items, ensure_ascii=False)
    prompt = (
        "Clean the following resume experience entries. For each entry, keep the summary text identical to the "
        "source (join lines exactly as provided, without paraphrasing). Return JSON only with keys 'role', "
        "'company', 'location', 'years', 'summary'. Leave values empty when unknown."
    )

    content = _groq_chat(
        [
            {"role": "system", "content": "You output JSON only."},
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\nResume text:\n{raw_text[:4000]}\n\n"
                    f"Preliminary JSON:\n{payload}\n\nJSON:"
                ),
            },
        ],
        model=model,
        expect_json=True,
    )

    if not content.strip():
        return items
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.info("[llm] groq experience response not JSON; returning baseline")
        return items

    cleaned: List[Dict[str, str]] = []
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            cleaned.append(
                {
                    "role": entry.get("role", ""),
                    "company": entry.get("company", ""),
                    "location": entry.get("location", ""),
                    "years": entry.get("years", ""),
                    "summary": entry.get("summary", ""),
                }
            )
    return cleaned or items


# ---------------------------------------------------------------------------
# Public API used by parser
# ---------------------------------------------------------------------------


def extract_name_via_llm(text: str) -> Tuple[str, float]:
    return extract_name_via_ollama(text)


def refine_experience_via_llm(raw_text: str, items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return refine_experience_via_ollama(raw_text, items)


def compare_experience_outputs(raw_text: str, items: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    outputs: Dict[str, List[Dict[str, str]]] = {}

    ollama_refined = refine_experience_via_ollama(raw_text, items)
    if ollama_refined:
        outputs["ollama"] = ollama_refined

    primary_groq_model = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
    secondary_groq_model = os.getenv("GROQ_SECOND_MODEL_NAME", "deepseek-r1-distill-llama-70b")
    tertiary_groq_model = os.getenv("GROQ_THIRD_MODEL_NAME", "openai/gpt-oss-120b")

    groq_models: List[str] = []
    for model in (primary_groq_model, secondary_groq_model, tertiary_groq_model):
        if model and model not in groq_models:
            groq_models.append(model)

    for model in groq_models:
        refined = refine_experience_via_groq(raw_text, items, model=model)
        if refined:
            outputs[f"groq:{model}"] = refined

    _log_comparison(items, outputs)
    return outputs


def _log_comparison(
    baseline: List[Dict[str, str]],
    outputs: Dict[str, List[Dict[str, str]]],
) -> None:
    if not outputs:
        return
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "experience_llm_comparison.log"

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "baseline": baseline,
        "models": {},
    }

    baseline_json = json.dumps(baseline, ensure_ascii=False, sort_keys=True)
    for model_name, refined in outputs.items():
        refined_json = json.dumps(refined, ensure_ascii=False, sort_keys=True)
        entry["models"][model_name] = {
            "output": refined,
            "differs_from_baseline": refined_json != baseline_json,
        }

    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as err:  # pragma: no cover
        logger.exception("[llm] failed to append comparison log: %s", err)

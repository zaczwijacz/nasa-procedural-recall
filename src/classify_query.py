"""
LLM-based query classifier for the NASA procedural recall assistant.

Uses a fast Ollama call with structured JSON output to classify queries
into routing modes. Falls back to regex keyword matching if the LLM call
fails, ensuring the pipeline never blocks on a classifier error.

Routing modes:
  safety_critical  — immediate risk to crew health or vehicle integrity
  procedural       — step-by-step operational or maintenance procedures
  informational    — general questions about systems, specs, or concepts
  prohibited       — out-of-scope or disallowed request
"""

import json
import re
import requests
from typing import Tuple

from llm_ollama import OLLAMA_BASE_URL, DEFAULT_MODEL  # DEFAULT_MODEL = qwen2.5vl:7b

# ---------------------------------------------------------------------------
# LLM classifier
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """You are a query classifier for a NASA crew procedural assistant.

This assistant supports astronauts and mission personnel asking questions in natural language
about anything relevant to spaceflight operations and crew wellbeing. Queries may be phrased
informally, urgently, or without explicit spacecraft context — assume good intent and a
legitimate operational need unless the query is clearly harmful.

Classify the user's query into exactly one of these categories:

- safety_critical: Any query involving an immediate threat to crew health or vehicle safety.
  Examples: medical emergencies (seizure, cardiac arrest, choking, bleeding, unconscious, injury),
  spacecraft emergencies (fire, smoke, decompression, toxic exposure, oxygen loss, pressure failure,
  radiation alert, evacuation, abort).

- procedural: Queries asking how to perform any task, operation, or step-by-step action relevant
  to spaceflight or crew operations. Cast this category broadly — if an astronaut might need to
  perform the action in space or during a mission, it is procedural.
  Examples: flight maneuvers (deorbit, reentry, rendezvous, docking, abort), EVA operations,
  system startup/shutdown, pre/post-flight checks, crew health procedures (dental care, wound
  treatment, medication administration, suturing, splinting, eye care), equipment repair and
  maintenance, habitat operations, sample collection, experiment setup, navigation, any
  "how do I" or "what are the steps for" questions about mission-relevant activities.

- informational: General questions about systems, concepts, specifications, figures, diagrams,
  or background knowledge related to spaceflight, vehicles, crew health, or mission operations.
  Examples: "what is TAEM guidance", "how does the RCS work", "explain Figure 2-5",
  "what are the symptoms of hypoxia", "what medication is used for X".

- prohibited: ONLY classify as prohibited if the query is clearly and unambiguously requesting
  something harmful or malicious with no plausible operational justification.
  Examples: intentionally harming crew, hacking or disabling safety-critical systems,
  bypassing alarms to hide a problem, requests involving weapons or sabotage.
  DO NOT classify as prohibited based on unfamiliar terminology, unconventional phrasing,
  or topics not explicitly listed in the procedural/informational examples above.

When in doubt between prohibited and any other category, choose the other category.

Return ONLY a JSON object with no other text:
{"type": "<category>", "confidence": <0.0-1.0>}"""


def _llm_classify(query: str) -> Tuple[str, float] | None:
    """
    Call Ollama to classify the query. Returns (type, confidence) or None on failure.
    Uses a short timeout so a slow model doesn't block the pipeline.
    """
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": _CLASSIFY_SYSTEM},
                    {"role": "user",   "content": f"Classify this query: {query}"},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=30,
        )
        r.raise_for_status()
        content = r.json()["message"]["content"]
        data = json.loads(content)
        qtype = data.get("type", "").lower()
        confidence = float(data.get("confidence", 0.5))
        if qtype in {"safety_critical", "procedural", "informational", "prohibited"}:
            return qtype, confidence
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Regex fallback (used when LLM call fails)
# ---------------------------------------------------------------------------

_PROHIBITED_RE = [
    r"\bhack\b", r"\boverride\s+safety\b", r"\bdisable\b.{0,20}\b(warning|alarm|sensor)\b",
    r"\bbypass\b.{0,20}\b(safety|check|lock)\b", r"\bweapon\b", r"\bexplosi",
]

_SAFETY_CRITICAL_RE = [
    r"\bemergency\b", r"\bcardiac\b", r"\bheart\s+attack\b", r"\bcpr\b",
    r"\bdefibrillat", r"\bunconscious\b", r"\bseizure\b", r"\banaphylax",
    r"\bbleed(ing)?\b", r"\bfracture\b", r"\bburn(s)?\b", r"\btrauma\b",
    r"\bdecompression\b", r"\bfire\b", r"\bsmoke\b", r"\btoxic\b",
    r"\bhypoxia\b", r"\boxygen\s+(deplet|fail|loss|low)\b",
    r"\bpressure\s+(loss|drop|failure)\b", r"\bevacuat", r"\babort\b",
    r"\bcontaminat", r"\bradiation\s+(exposure|emergency|alert)\b",
    r"\bloss\s+of\s+consciousness\b",
]

_PROCEDURAL_RE = [
    r"\bprocedure\b", r"\bchecklist\b", r"\bhow\s+to\b", r"\bsteps?\s+(for|to)\b",
    r"\boperat(e|ion|ing)\b", r"\bactivat(e|ion)\b", r"\bconfigur(e|ation)\b",
    r"\blaunch\b", r"\bdocking\b", r"\bundocking\b", r"\beva\b",
    r"\bspacewal(k|king)\b", r"\bdeploy\b", r"\bmaintenance\b", r"\brepair\b",
    r"\bshutdown\b", r"\bstart[- ]?up\b", r"\bmaneuver\b", r"\brendezvous\b",
    r"\bpre[- ]?flight\b", r"\bpost[- ]?flight\b",
]


def _regex_classify(query: str) -> Tuple[str, float]:
    q = query.lower()
    for pat in _PROHIBITED_RE:
        if re.search(pat, q):
            return "prohibited", 1.0
    hits = sum(1 for pat in _SAFETY_CRITICAL_RE if re.search(pat, q))
    if hits >= 2:
        return "safety_critical", 1.0
    if hits == 1:
        return "safety_critical", 0.75
    hits = sum(1 for pat in _PROCEDURAL_RE if re.search(pat, q))
    if hits >= 2:
        return "procedural", 1.0
    if hits == 1:
        return "procedural", 0.75
    return "informational", 0.5


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def classify_query(query: str) -> Tuple[str, float]:
    """
    Classify a query using the LLM first, falling back to regex if needed.
    Returns (routing_mode, confidence).
    """
    result = _llm_classify(query)
    if result is not None:
        return result
    # Fallback: regex-based classifier (instant, no LLM required)
    return _regex_classify(query)

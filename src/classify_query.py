"""
Keyword-based query classifier for the NASA procedural recall assistant.

Intentionally deterministic and keyword-driven — not LLM-based — so that
routing decisions are auditable, reproducible, and do not add latency.

Routing modes (from policy.yaml):
  safety_critical  — immediate risk to crew health or vehicle integrity
  procedural       — step-by-step operational or maintenance procedures
  informational    — general questions about systems, specs, or concepts
  prohibited       — out-of-scope or disallowed request
"""

import re
from typing import Tuple

# ---------------------------------------------------------------------------
# Term lists — ordered by specificity (checked top-down, first match wins)
# ---------------------------------------------------------------------------

_PROHIBITED = [
    r"\bhack\b", r"\boverride\s+safety\b", r"\bdisable\b.{0,20}\b(warning|alarm|sensor)\b",
    r"\bbypass\b.{0,20}\b(safety|check|lock)\b", r"\bweapon\b", r"\bexplosi",
]

_SAFETY_CRITICAL = [
    # Medical emergencies
    r"\bemergency\b", r"\bcardiac\b", r"\bheart\s+attack\b", r"\bcpr\b",
    r"\bdefibrillat", r"\bunconscious\b", r"\bseizure\b", r"\banaphylax",
    r"\bbleed(ing)?\b", r"\bfracture\b", r"\bburn(s)?\b", r"\btrauma\b",
    r"\bmedical\s+(crisis|emergency|incident)\b",
    # Spacecraft emergencies
    r"\bdecompression\b", r"\brapid\s+depressuri", r"\bfire\b", r"\bsmoke\b",
    r"\btoxic\b", r"\bhypoxia\b", r"\boxygen\s+(deplet|fail|loss|low)\b",
    r"\bpressure\s+(loss|drop|failure)\b", r"\bcabin\s+pressure\b",
    r"\bevacuat", r"\babort\b", r"\bcritical\s+failure\b",
    r"\bcontaminat", r"\bradiation\s+(exposure|emergency|alert)\b",
    r"\bloss\s+of\s+consciousness\b", r"\bdo\s+not\s+breathe\b",
]

_PROCEDURAL = [
    r"\bprocedure\b", r"\bchecklist\b", r"\bhow\s+to\b", r"\bsteps?\s+(for|to)\b",
    r"\boperat(e|ion|ing)\b", r"\bactivat(e|ion)\b", r"\bconfigur(e|ation)\b",
    r"\binitializ(e|ation)\b", r"\blaunch\b", r"\bdocking\b", r"\bundocking\b",
    r"\beva\b", r"\bspacewal(k|king)\b", r"\bdeploy\b", r"\bmaintenance\b",
    r"\brepair\b", r"\bshutdown\b", r"\bstart[- ]?up\b", r"\bcalibrat(e|ion)\b",
    r"\bdock(ing)?\b", r"\bmaneuver\b", r"\brendezvous\b", r"\btransfill\b", r"\bisolat(e|ion)\b",
    r"\bvent(ing)?\b", r"\bpurge\b", r"\bpre[- ]?flight\b", r"\bpost[- ]?flight\b",
    r"\bsystem\s+(reset|restart|recovery)\b",
]

# Informational is the default — no explicit term list needed.

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def classify_query(query: str) -> Tuple[str, float]:
    """
    Return (routing_mode, confidence) for a query string.

    confidence is a rough indicator:
      1.0  — strong keyword match
      0.75 — weak / single match
      0.5  — default (informational)

    Precedence: prohibited > safety_critical > procedural > informational
    """
    q = query.lower()

    for pat in _PROHIBITED:
        if re.search(pat, q):
            return "prohibited", 1.0

    hits = sum(1 for pat in _SAFETY_CRITICAL if re.search(pat, q))
    if hits >= 2:
        return "safety_critical", 1.0
    if hits == 1:
        return "safety_critical", 0.75

    hits = sum(1 for pat in _PROCEDURAL if re.search(pat, q))
    if hits >= 2:
        return "procedural", 1.0
    if hits == 1:
        return "procedural", 0.75

    return "informational", 0.5

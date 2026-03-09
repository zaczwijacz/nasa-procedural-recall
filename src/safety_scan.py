"""
Safety scan for the NASA procedural recall assistant.

Three layers of checks, applied in order:

  1. Pattern blocks  — reject responses containing dangerous action verbs or
                       numbered-step formats (policy: output_constraints.block_if_contains)

  2. Citation checks — every factual claim must cite a retrieved page; the cited
                       page must fall within the page range of a retrieved section.
                       SKIPPED when the response is a recognized fail-safe escalation.

  3. Length check    — reject responses exceeding max_answer_chars.
"""

import re
from typing import Any, Dict, List

# Responses that start with this prefix are fail-safe escalations.
# They contain no citations by design and must not be blocked for that reason.
_FAILSAFE_PREFIX = "not found in provided documentation"

# Canonical citation pattern: (Doc:<title>, Page:<number>)
_CITE_RE = re.compile(r"\(Doc:([^,\)]+),\s*Page:\s*(\d+)\)")


def scan_output(
    text: str,
    policy: dict,
    retrieved_pages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Validate an LLM response against the active policy.

    Returns a dict with:
      blocked          bool   — True if any check failed
      reasons          list   — human-readable block reasons
      missing_citations bool  — True if no citation found in a non-failsafe response
      bad_citations    list   — citation strings that don't match any retrieved page
      is_failsafe      bool   — True if the response is a fail-safe escalation message
    """
    reasons: list[str] = []
    blocked = False

    is_failsafe = text.lower().strip().startswith(_FAILSAFE_PREFIX)

    # ------------------------------------------------------------------
    # 1. Pattern blocks
    # ------------------------------------------------------------------
    for pat in policy["output_constraints"]["block_if_contains"]:
        if re.search(pat, text):
            blocked = True
            reasons.append(f"pattern_block:{pat}")

    # ------------------------------------------------------------------
    # 2. Citation checks  (skip for fail-safe responses)
    # ------------------------------------------------------------------
    missing_citations = False
    bad_citations: list[str] = []

    if not is_failsafe:
        cite_pat = policy["output_constraints"]["require_inline_citation_pattern"]
        missing_citations = re.search(cite_pat, text) is None
        if missing_citations:
            blocked = True
            reasons.append("missing_citations")

        if policy["output_constraints"]["citation_must_be_from_retrieved_pages"]:
            # Build allowed page ranges from retrieved sections.
            # page_number = section start; end_page = section end (inclusive).
            allowed: list[tuple[str, int, int]] = []
            for p in retrieved_pages:
                start = int(p["page_number"])
                end   = int(p.get("end_page", p["page_number"]))
                allowed.append((p["doc_title"], start, end))

            for m in _CITE_RE.finditer(text):
                title = m.group(1).strip()
                page  = int(m.group(2))
                if not any(t == title and s <= page <= e for t, s, e in allowed):
                    bad_citations.append(m.group(0))

            if bad_citations:
                blocked = True
                reasons.append("bad_citations")

    # ------------------------------------------------------------------
    # 3. Length check
    # ------------------------------------------------------------------
    max_chars = policy["output_constraints"].get("max_answer_chars", 2400)
    if len(text) > max_chars:
        blocked = True
        reasons.append("answer_too_long")

    return {
        "blocked":           blocked,
        "reasons":           reasons,
        "missing_citations": missing_citations,
        "bad_citations":     bad_citations,
        "is_failsafe":       is_failsafe,
    }

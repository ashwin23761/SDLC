"""Compare a run with a human-labelled reference JSON without inventing scores."""

from __future__ import annotations

import re
from difflib import SequenceMatcher


def evaluate_against_reference(run_record: dict, reference: dict, threshold: float = 0.55) -> dict:
    """Greedily align expected and extracted requirements and report transparent metrics.

    Reference schema: {"requirements": [{"id", "description", "ambiguous": bool,
    "controls": [str]}], "expected_sdlc": "..."}. Similarity-based matching is a
    review aid, not a substitute for expert scoring.
    """
    expected = reference.get("requirements", [])
    generated = (
        run_record.get("requirements", {}).get("functional_requirements", [])
        + run_record.get("requirements", {}).get("non_functional_requirements", [])
    )
    if not expected:
        raise ValueError("Reference JSON must contain a non-empty 'requirements' list.")
    candidates = []
    for ei, exp in enumerate(expected):
        for gi, gen in enumerate(generated):
            similarity = _similarity(exp.get("description", ""), gen.get("description", ""))
            candidates.append((similarity, ei, gi))
    matches = {}
    used_generated = set()
    for similarity, ei, gi in sorted(candidates, reverse=True):
        if similarity < threshold or ei in matches or gi in used_generated:
            continue
        matches[ei] = (gi, similarity)
        used_generated.add(gi)

    tp = len(matches)
    precision = tp / len(generated) if generated else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    quality = run_record.get("extended_analysis", {}).get("quality_issues", [])
    flagged_ids = {row.get("requirement_id") for row in quality}
    ambiguity = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    controls_expected = set()
    controls_found = set()
    alignment = []
    for ei, exp in enumerate(expected):
        controls_expected.update(str(x).strip().lower() for x in exp.get("controls", []))
        if ei not in matches:
            if exp.get("ambiguous"):
                ambiguity["fn"] += 1
            continue
        gi, similarity = matches[ei]
        gen = generated[gi]
        actual_flag = gen.get("id") in flagged_ids
        expected_flag = bool(exp.get("ambiguous", False))
        ambiguity["tp" if expected_flag and actual_flag else "fp" if actual_flag else "fn" if expected_flag else "tn"] += 1
        controls_found.update(str(x).strip().lower() for x in gen.get("applicable_regulations", []))
        alignment.append({"reference_id": exp.get("id", str(ei + 1)), "generated_id": gen.get("id"), "similarity": round(similarity, 3), "match_review_required": True})
    control_recall = len(controls_expected & controls_found) / len(controls_expected) if controls_expected else None
    sdlc_expected = str(reference.get("expected_sdlc", "")).strip().lower()
    sdlc_actual = str(run_record.get("sdlc", {}).get("recommended_approach", "")).strip().lower()
    return {
        "matching_method": f"Greedy statement similarity; threshold={threshold:.2f}",
        "warning": "Automated alignment is approximate. A domain expert must review matches and labels.",
        "requirements": {"expected": len(expected), "generated": len(generated), "matched": tp, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)},
        "ambiguity_detection": ambiguity,
        "candidate_control_recall": round(control_recall, 4) if control_recall is not None else "Not available: no expected controls supplied",
        "sdlc": {"expected": reference.get("expected_sdlc"), "generated": run_record.get("sdlc", {}).get("recommended_approach"), "match": sdlc_expected in sdlc_actual if sdlc_expected else "Not available"},
        "alignment": alignment,
    }


def _similarity(left: str, right: str) -> float:
    normalize = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    a, b = normalize(left), normalize(right)
    if not a or not b:
        return 0.0
    words_a, words_b = set(a.split()), set(b.split())
    jaccard = len(words_a & words_b) / max(1, len(words_a | words_b))
    sequence = SequenceMatcher(None, a, b).ratio()
    return 0.6 * jaccard + 0.4 * sequence

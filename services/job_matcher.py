"""Lightweight job matching service for the HireMate baseline."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "jobs_sample.json"


@lru_cache(maxsize=1)
def _load_jobs() -> List[Dict[str, Any]]:
    if not DATA_FILE.exists():
        return []
    with DATA_FILE.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            return []
    jobs: List[Dict[str, Any]] = []
    for raw in data or []:
        if isinstance(raw, dict):
            jobs.append(raw)
    return jobs


def _normalize_skills(items: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for item in items:
        if not item:
            continue
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            normalized.append(key)
    return normalized


def _score_job(job: Dict[str, Any], resume_skills: List[str]) -> Tuple[float, Dict[str, Any]]:
    job_skills = _normalize_skills(job.get("skills", []))
    if not job_skills:
        return 0.0, {}
    overlap = [skill for skill in job_skills if skill in resume_skills]
    coverage = len(overlap) / len(job_skills)
    score = coverage * 0.8

    # Bonus if job skills contain high priority skills from resume.
    priority_skills = {"python", "machine learning", "llm", "aws", "javascript"}
    if any(skill in priority_skills for skill in overlap):
        score += 0.1

    meta = {
        "overlap": overlap,
        "job_skills": job_skills,
        "coverage": coverage,
    }
    return score, meta


def match_jobs(resume: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    """Return a best-effort list of matching jobs for the resume."""

    skills = resume.get("skills", []) if isinstance(resume, dict) else []
    normalized_skills = _normalize_skills(skills if isinstance(skills, Iterable) else [])
    if not normalized_skills:
        return []

    ranked: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for job in _load_jobs():
        score, meta = _score_job(job, normalized_skills)
        if score <= 0:
            continue
        ranked.append((score, job, meta))

    ranked.sort(key=lambda item: item[0], reverse=True)
    matches: List[Dict[str, Any]] = []
    for score, job, meta in ranked[:limit]:
        matches.append(
            {
                "id": job.get("id"),
                "title": job.get("title"),
                "company": job.get("company"),
                "location": job.get("location"),
                "description": job.get("description"),
                "score": round(float(score), 2),
                "overlap": meta.get("overlap", []),
            }
        )
    return matches


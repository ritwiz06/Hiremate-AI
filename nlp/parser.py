# parser.py
# --- Structured CV parsing from raw text (clean-room) ---

import re
from typing import Any, Dict, List, Optional, Tuple
from rapidfuzz import fuzz, process

# Optional NER (will be used if installed; otherwise we fall back to heuristics)
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

try:
    from nlp.llm import (
        extract_name_via_llm,
        refine_experience_via_llm,
        compare_experience_outputs,
    )  # optional LLM helpers
except Exception:
    extract_name_via_llm = None
    refine_experience_via_llm = None
    compare_experience_outputs = None


# ---- Debug utilities ----

_DEBUG = False


def set_debug(enabled: bool) -> None:
    """Toggle step-by-step debug output for this module."""
    global _DEBUG
    _DEBUG = enabled


def _debug(step: str, detail: Optional[str] = None) -> None:
    """Emit a debug line when debugging is enabled."""
    if not _DEBUG:
        return
    if detail:
        print(f"[parser] {step}: {detail}")
    else:
        print(f"[parser] {step}")


# ---- Utilities ----

def _normalize(text: str) -> str:
    # Normalize bullets, whitespace
    text = text.replace("\u2022", "-").replace("•", "-").replace("·", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    normalized = text.strip()
    _debug("normalize", f"input chars={len(text)}, output chars={len(normalized)}")
    return normalized


def _lines(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    _debug("lines", f"kept {len(lines)} lines")
    return lines


# ---- Primitive extractors ----

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}"
)
_URL_RE = re.compile(r"\bhttps?://[^\s)]+", re.IGNORECASE)
_LINKEDIN_RE = re.compile(r"(linkedin\.com/in/[A-Za-z0-9\-_/]+)", re.IGNORECASE)
_GITHUB_RE = re.compile(r"(github\.com/[A-Za-z0-9\-_/]+)", re.IGNORECASE)

def extract_emails(text: str) -> List[str]:
    emails = sorted(set(_EMAIL_RE.findall(text)))
    _debug("extract_emails", f"found {len(emails)}")
    return emails

def extract_phones(text: str) -> List[str]:
    phones = sorted(set(_PHONE_RE.findall(text)))
    _debug("extract_phones", f"found {len(phones)}")
    return phones

def extract_links(text: str) -> Dict[str, List[str]]:
    urls = sorted(set(_URL_RE.findall(text)))
    linkedin = sorted(set(m.group(1) for m in _LINKEDIN_RE.finditer(text)))
    github = sorted(set(m.group(1) for m in _GITHUB_RE.finditer(text)))
    _debug("extract_links", f"urls={len(urls)}, linkedin={len(linkedin)}, github={len(github)}")
    return {"urls": urls, "linkedin": linkedin, "github": github}


# ---- Name extraction ----

COMMON_HEADERS = {"resume", "curriculum vitae", "cv", "profile"}
CONTACT_HINT_RE = re.compile(r"@|\bphone\b|\bemail\b|\blinked?in\b|\bgithub\b|\b\+?\d{3,}|\|", re.IGNORECASE)
LOCATION_TOKENS = {
    "india", "united", "states", "usa", "canada", "australia", "kingdom",
    "london", "delhi", "mumbai", "bangalore", "bengaluru", "maharashtra", "karnataka",
    "telangana", "hyderabad", "texas", "california", "pune", "chennai", "gujarat",
    "atlanta", "georgia", "mountain", "view", "ca", "ga", "ny", "tx", "wa", "co",
    "il", "ma",
}
LOCATION_PHRASES = {
    "new york", "san francisco", "tamil nadu", "united kingdom", "united states",
    "pune, maharashtra", "mumbai, maharashtra", "chennai, tamil nadu",
}


def _first_line_candidate(text: str) -> Optional[str]:
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if CONTACT_HINT_RE.search(line):
            continue
        words = [w for w in re.split(r"[^A-Za-z']+", line) if w]
        if len(words) < 2 or len(words) > 5:
            continue
        letters = sum(len(w) for w in words)
        if letters < 4:
            continue
        lower_tokens = {w.lower() for w in words}
        lower_line = line.lower()
        if lower_tokens & LOCATION_TOKENS:
            continue
        if any(phrase in lower_line for phrase in LOCATION_PHRASES):
            continue
        if any(keyword in lower_line for keyword in ("objective", "summary", "education", "experience", "skills")):
            continue
        return line
    return None


def _score_name_candidate(line: str, next_line: Optional[str], prev_line: Optional[str]) -> int:
    cleaned = re.sub(r"[^A-Za-z' ]", " ", line).strip()
    tokens = [tok for tok in cleaned.split() if tok]
    if not tokens or len(tokens) > 5:
        return 0
    if any(len(tok) == 1 for tok in tokens):
        return 0
    lower_line = cleaned.lower()
    if any(stop in lower_line for stop in COMMON_HEADERS):
        return 0
    if any(keyword in lower_line for keyword in ("bachelor", "master", "engineer", "objective", "summary", "education")):
        return 0
    if any(phrase in lower_line for phrase in LOCATION_PHRASES):
        return 1
    if any(word in LOCATION_TOKENS for word in lower_line.split()):
        return 1  # allow, but very low confidence

    alpha_tokens = [tok for tok in tokens if tok.isalpha()]
    if not alpha_tokens:
        return 0

    upper_tokens = sum(tok.isupper() for tok in alpha_tokens)
    title_tokens = sum(tok[0].isupper() and tok[1:].islower() for tok in alpha_tokens)
    score = 0
    if upper_tokens == len(alpha_tokens):
        score += 4
    elif title_tokens == len(alpha_tokens):
        score += 3
    elif any(tok[0].isupper() for tok in alpha_tokens):
        score += 1

    if CONTACT_HINT_RE.search(next_line or ""):
        score += 2
    if CONTACT_HINT_RE.search(prev_line or ""):
        score += 1
    if "," in line or "|" in line:
        score -= 1
    if any(char.isdigit() for char in line):
        score -= 2

    return score


def _token_structure_score(name: str) -> int:
    tokens = [tok for tok in re.split(r"[^A-Za-z']+", name) if tok]
    if not tokens or len(tokens) < 2 or len(tokens) > 6:
        return -2
    score = 0
    if all(tok.isupper() for tok in tokens):
        score += 4
    elif all(len(tok) > 1 and tok[0].isupper() and tok[1:].islower() for tok in tokens):
        score += 3
    elif any(tok[0].isupper() for tok in tokens):
        score += 1

    avg_len = sum(len(tok) for tok in tokens) / len(tokens)
    if avg_len >= 4:
        score += 1
    if any(len(tok) >= 8 for tok in tokens):
        score += 1

    lower_name = name.lower()
    if any(phrase in lower_name for phrase in LOCATION_PHRASES):
        score -= 3
    if any(tok.lower() in LOCATION_TOKENS for tok in tokens):
        score -= 2
    if any(char.isdigit() for char in name):
        score -= 4
    return score


def _collect_name_candidates(text: str) -> List[Dict[str, Any]]:
    lines = _lines(text)
    candidates: List[Dict[str, Any]] = []

    def register(name: str, score: float, method: str, detail: Optional[str] = None) -> None:
        clean = (name or "").strip()
        if not clean:
            return
        key = clean.lower()
        existing = next((c for c in candidates if c["name"].lower() == key), None)
        if existing:
            if score > existing["score"]:
                existing["score"] = score
                existing["method"] = method
                existing["detail"] = detail or existing.get("detail")
            return
        debug_detail = detail or ""
        extra = f" | {debug_detail}" if debug_detail else ""
        candidates.append({
            "name": clean,
            "score": score,
            "method": method,
            "detail": detail,
        })
        _debug("name_candidate", f"{clean} via {method} score={round(score,2)}{extra}")

    # First line heuristic
    first_candidate = _first_line_candidate(text)
    if first_candidate:
        idx = next((i for i, ln in enumerate(lines) if ln.strip() == first_candidate.strip()), 0)
        next_ln = lines[idx + 1] if idx + 1 < len(lines) else None
        score = _score_name_candidate(first_candidate, next_ln, None)
        base = 6 + score + _token_structure_score(first_candidate)
        register(first_candidate, base, "first_line", f"line_score={score}")

    # Top-line heuristics
    for idx, ln in enumerate(lines[:8]):
        if not ln or _EMAIL_RE.search(ln):
            continue
        prev_ln = lines[idx - 1] if idx - 1 >= 0 else None
        next_ln = lines[idx + 1] if idx + 1 < len(lines) else None
        score = _score_name_candidate(ln, next_ln, prev_ln)
        if score >= 2:
            base = 4 + score + _token_structure_score(ln)
            register(ln.strip(), base, "heuristic", f"heuristic_score={score}")

    # spaCy NER candidates
    if _NLP:
        try:
            doc = _NLP(text[:1000])
        except Exception as err:
            _debug("name_spacy_error", str(err))
        else:
            total = len(list(doc.ents))
            _debug("name_spacy", f"found {total} entities")
            for ent in doc.ents:
                if ent.label_ != "PERSON":
                    continue
                ent_text = ent.text.strip()
                if not ent_text:
                    continue
                base = 5 + _token_structure_score(ent_text)
                register(ent_text, base, "spacy", f"start={ent.start_char}")

    # Optional LLM candidate
    if extract_name_via_llm:
        try:
            _debug("name_llm", "invoking extract_name_via_llm")
            llm_result = extract_name_via_llm(text)
            if isinstance(llm_result, tuple):
                llm_name, confidence = llm_result
            else:
                llm_name, confidence = llm_result, 0.0
            if llm_name:
                conf_score = float(confidence) if confidence is not None else 0.0
                base = 8 + (conf_score * 4) + _token_structure_score(llm_name)
                register(llm_name, base, "llm", f"confidence={conf_score}")
            else:
                _debug("name_llm", "LLM returned empty")
        except Exception as err:
            _debug("name_llm_error", str(err))
    else:
        _debug("name_llm", "no LLM helper available")

    return candidates

def extract_name(text: str) -> Optional[str]:
    candidates = _collect_name_candidates(text)
    if not candidates:
        _debug("extract_name", "no match")
        return None
    best = max(candidates, key=lambda c: c["score"])
    summary = ", ".join(f"{c['name']}:{round(c['score'], 2)}" for c in candidates)
    _debug(
        "extract_name",
        f"{best['name']} via {best['method']} (score={round(best['score'], 2)}) | candidates=[{summary}]"
    )
    return best["name"]



# ---- Sectionizer (simple heuristics) ----

SECTION_HEADS = [
    "experience", "work experience", "professional experience", "employment",
    "projects", "project experience", "university projects",
    "education", "academics",
    "skills", "technical skills", "key skills",
    "certifications", "achievements", "awards", "publications", "research publications",
    "activities", "additional", "summary", "profile"
]

def split_sections(text: str) -> Dict[str, str]:
    lines = _lines(text)
    sections: Dict[str, List[str]] = {}
    current = "summary"
    sections[current] = []
    for ln in lines:
        low = ln.lower().strip(": ")
        if any(low.startswith(h) for h in SECTION_HEADS):
            current = next(h for h in SECTION_HEADS if low.startswith(h))
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, [])
            sections[current].append(ln)
    final = {k: "\n".join(v).strip() for k, v in sections.items() if v}
    _debug("split_sections", f"sections={list(final.keys())}")
    return final


# ---- Skills extraction ----

# You can expand this bank per your domains
SKILL_BANK = [
    # Programming / Data
    "python","java","c++","javascript","typescript","sql","nosql","scala","go","rust",
    "pandas","numpy","scikit-learn","sklearn","pytorch","tensorflow","keras","xgboost","lightgbm",
    "nlp","spacy","nltk","transformers","hugging face","llm","langchain","langgraph",
    "fastapi","flask","django","streamlit",
    "docker","kubernetes","aws","gcp","azure","airflow","mlops","ci/cd","git","postgres","mysql","mongodb","snowflake","databricks",
    "tableau","power bi","look ml","dbt",
    # General
    "data analysis","machine learning","deep learning","statistics","computer vision","time series",
]

def extract_skills(text: str, bank: Optional[List[str]] = None, threshold: int = 85) -> List[str]:
    bank = bank or SKILL_BANK
    found = set()
    low_text = text.lower()
    # exact / substring hits fast-path
    for sk in bank:
        if sk in low_text:
            found.add(sk)
    # fuzzy for near-misses
    remaining = [sk for sk in bank if sk not in found]
    for sk in remaining:
        score = fuzz.partial_ratio(sk, low_text)
        if score >= threshold:
            found.add(sk)
    # prettify: title-case some tokens
    def pretty(s: str) -> str:
        if s in {"sql","nlp","ci/cd","aws","gcp","gpt"}:
            return s.upper()
        if s in {"pandas","numpy","sklearn","pytorch","tensorflow","keras","xgboost","langchain","langgraph","dbt"}:
            return s
        return s.title()
    skills = sorted(pretty(s) for s in found)
    _debug("extract_skills", f"found {len(skills)}")
    return skills


# ---- Projects extraction (lightweight) ----

PROJECT_HEADER_LABELS = {
    "projects", "project", "project experience", "university projects", "academic projects"
}
PROJECT_BULLETS = "-•*·●▪‣o"
PROJECT_VERB_PREFIXES = {
    "published", "managed", "created", "designed", "implemented", "developed",
    "analyzed", "analysed", "optimized", "conducted", "built", "led", "organized",
    "coordinated", "executed", "implemented", "performed",
}
SKILL_HEADING_TOKENS = {"skill", "skills", "languages", "technologies", "tools", "tech stack"}


def _looks_like_skill_heading(line: str) -> bool:
    lower = line.lower().strip()
    if not lower:
        return False
    if lower.startswith(("languages:", "technologies:", "tools:", "tech stack:")):
        return True
    if any(token in lower for token in ("skills", "skill")) and len(lower.split()) <= 4:
        return True
    return False


def _looks_like_project_title(line: str) -> bool:
    line = line.strip()
    if not line or len(line.split()) > 16:
        return False
    letters = sum(ch.isalpha() for ch in line)
    if letters < 3:
        return False
    if line.isupper() and letters >= 4:
        return True
    words = line.split()
    alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
    if not alpha_words:
        return False
    lower_line = line.lower()
    first_word = ""
    for token in alpha_words:
        cleaned = re.sub(r"[^A-Za-z]", "", token)
        if cleaned:
            first_word = cleaned.lower()
            break
    if first_word and first_word in PROJECT_VERB_PREFIXES:
        return False
    title_case = sum(w[0].isupper() and w[1:].islower() for w in alpha_words if len(w) > 1)
    upper_case = sum(w.isupper() for w in alpha_words if len(w) > 1)
    ratio = (title_case + upper_case) / len(alpha_words)
    return ratio >= 0.8 and len(alpha_words) <= 8


def extract_projects(text: str) -> List[Dict[str, str]]:
    lines = _lines(text)
    projects: List[Dict[str, str]] = []
    current_title: Optional[str] = None
    details: List[str] = []
    pending_bullet = False

    def flush_current() -> None:
        nonlocal current_title, details
        if not current_title:
            return
        raw_title = current_title.strip(" -•")
        inline_detail = ""
        if ":" in raw_title:
            title_part, inline_detail = raw_title.split(":", 1)
            raw_title = title_part.strip()
            inline_detail = inline_detail.strip()
        detail_text = " ".join(details).strip()
        if inline_detail:
            detail_text = " ".join(filter(None, [inline_detail, detail_text])).strip()
        projects.append({"name": raw_title, "details": detail_text})
        current_title = None
        details = []

    for ln in lines:
        stripped = ln.strip()
        lower = stripped.lower()
        if lower in PROJECT_HEADER_LABELS:
            continue

        if not stripped:
            continue

        if _looks_like_skill_heading(stripped):
            flush_current()
            pending_bullet = False
            continue

        if ":" in stripped and not _looks_like_skill_heading(stripped):
            title_segment, detail_segment = stripped.split(":", 1)
            title_candidate = title_segment.strip()
            detail_candidate = detail_segment.strip()
            if title_candidate and any(ch.isalpha() for ch in title_candidate):
                flush_current()
                current_title = title_candidate
                details = []
                if detail_candidate:
                    details.append(detail_candidate)
                pending_bullet = False
                continue

        if all(ch in PROJECT_BULLETS for ch in stripped if not ch.isspace()):
            pending_bullet = True
            continue

        if stripped[0] in PROJECT_BULLETS:
            bullet_text = stripped.lstrip(PROJECT_BULLETS + " ")
            if not bullet_text:
                pending_bullet = True
                continue
            if current_title:
                details.append(bullet_text)
            else:
                current_title = bullet_text
            pending_bullet = False
            continue

        if pending_bullet:
            if current_title:
                details.append(stripped)
            else:
                current_title = stripped
            pending_bullet = False
            continue

        first_word = ""
        for token in stripped.split():
            candidate = re.sub(r"[^A-Za-z]", "", token)
            if candidate:
                first_word = candidate.lower()
                break

        if current_title and not _looks_like_project_title(stripped):
            lower = stripped.lower()
            if first_word and (first_word in PROJECT_VERB_PREFIXES or lower[0].islower()):
                details.append(stripped)
                continue
            if details and details[-1].endswith("…"):
                details[-1] = f"{details[-1]} {stripped}".strip()
                continue
            if current_title and not projects:
                details.append(stripped)
                continue

        if current_title:
            flush_current()

        if _looks_like_project_title(stripped):
            current_title = stripped.strip(" -•")
            details = []
        else:
            # treat as continuation of previous project's details if any
            if projects and stripped:
                combined = " ".join(filter(None, [projects[-1]["details"], stripped])).strip()
                projects[-1]["details"] = combined

    if current_title:
        flush_current()

    cleaned: List[Dict[str, str]] = []
    seen = set()
    for proj in projects:
        name = proj.get("name", "").strip()
        detail = proj.get("details", "").strip()
        if not name:
            continue
        if _looks_like_skill_heading(name):
            continue
        sig = (name.lower(), detail.lower())
        if sig in seen:
            continue
        seen.add(sig)
        cleaned.append({"name": name, "details": detail})

    _debug("extract_projects", f"found {len(cleaned)}")
    return cleaned


# ---- Publications extraction ----

PUBLICATION_HEADER_LABELS = {"publications", "research publications", "selected publications"}


def extract_publications(text: str) -> List[str]:
    lines = _lines(text)
    pubs: List[str] = []
    buffer: List[str] = []

    for ln in lines:
        stripped = ln.strip()
        lower = stripped.lower()
        if not stripped:
            continue
        if lower in PUBLICATION_HEADER_LABELS:
            continue

        if stripped[0] in PROJECT_BULLETS:
            if buffer:
                pubs.append(" ".join(buffer).strip())
                buffer = []
            buffer.append(stripped.lstrip(PROJECT_BULLETS + " "))
            continue

        if buffer:
            buffer.append(stripped)
        else:
            pubs.append(stripped)

    if buffer:
        pubs.append(" ".join(buffer).strip())

    cleaned = [pub.strip() for pub in pubs if pub.strip()]
    _debug("extract_publications", f"found {len(cleaned)}")
    return cleaned


# ---- Education extraction (lightweight) ----

EDU_DEG = (
    r"(?:Bachelor|Master)"
    r"(?:\s+of\s+(?:Technology|Engineering|Science|Arts|Commerce|Computer Applications|Computer Science|Business Administration|Management))?"
    r"(?:\s+in\s+[A-Za-z0-9 &/\-,()]{2,60})?"
    r"|B\.?Tech|B\.?E\.?|B\.?Sc|B\.?Com|BCA|BBA"
    r"|M\.?Tech|M\.?E\.?|M\.?Sc|M\.?S\.?|MCA|MBA|PGDM"
    r"|Ph\.?D\.?|Doctor(?:ate|\s+of\s+(?:Philosophy|Science|Engineering|Medicine))"
)
EDU_PAT = re.compile(
    rf"(?P<deg>{EDU_DEG})[ ,:;\-]*"
    r"(?P<inst>[A-Za-z0-9 .,&\-()]{0,80})"
    r"(?:,?\s*(?P<years>(?:19|20)\d{{2}}(?:\s*[-–—to]{{1,3}}\s*(?:Present|present|(?:19|20)\d{{2}}))?))?",
    re.IGNORECASE,
)
EDU_DEG_LINE_PAT = re.compile(rf"(?P<deg>{EDU_DEG})", re.IGNORECASE)
EDU_MONTH_PATTERN = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
YEAR_RANGE_PAT = re.compile(
    rf"(?:{EDU_MONTH_PATTERN}\s+)?(19|20)\d{{2}}\s*(?:[-–—to]{{1,3}}\s*(?:Present|present|Now|now|(?:{EDU_MONTH_PATTERN}\s+)?(?:19|20)\d{{2}}))?",
    re.IGNORECASE,
)
EDU_YEAR_TOKEN = re.compile(
    rf"(?:(?P<month>{EDU_MONTH_PATTERN})[\s\-]*)?(?P<year>(?:19|20)\d{{2}}|Present|present|Now|now)",
    re.IGNORECASE,
)
EDU_SCORE_PAT = re.compile(r"(cgpa|gpa|score|result|grade|percentage|percent|rank)", re.IGNORECASE)
INSTITUTION_KEYWORDS = {
    "university", "institute", "college", "academy", "school", "polytechnic", "faculty",
}


def _normalize_degree(text: str) -> str:
    deg = re.sub(r"\s+", " ", text).strip(" ,;-")
    key = deg.lower().replace(".", "")
    canonical = {
        "mba": "MBA",
        "bba": "BBA",
        "bca": "BCA",
        "btech": "B.Tech",
        "be": "B.E.",
        "mtech": "M.Tech",
        "me": "M.E.",
        "mca": "MCA",
        "phd": "PhD",
    }
    if key in canonical:
        return canonical[key]
    if deg.lower().startswith(("bachelor", "master", "doctor")):
        titled = deg.title()
        for word in ("Of", "In", "And"):
            titled = titled.replace(f" {word} ", f" {word.lower()} ")
        return titled
    return deg


def _isolated_degree_match(line: str):
    best = None
    best_len = -1
    for match in EDU_DEG_LINE_PAT.finditer(line):
        start, end = match.span()
        if start > 0 and line[start - 1].isalpha():
            continue
        if end < len(line) and line[end].isalpha():
            continue
        span_len = end - start
        if span_len > best_len:
            best = match
            best_len = span_len
    return best


def _split_blocks(text: str) -> List[List[str]]:
    blocks: List[List[str]] = []
    current: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _looks_like_degree_line(line: str) -> bool:
    if _isolated_degree_match(line):
        return True
    lower = line.lower()
    if any(keyword in lower for keyword in ("certificate", "diploma", "degree", "honours", "honors", "school certificate")):
        return True
    return False


def _looks_like_institution_line(line: str) -> bool:
    lower = line.lower()
    if EDU_SCORE_PAT.search(lower):
        return False
    if any(keyword in lower for keyword in INSTITUTION_KEYWORDS):
        return True
    if any(tok in lower for tok in LOCATION_TOKENS):
        return False
    if "," in lower:
        parts = [p.strip() for p in lower.split(",")]
        if any(part in LOCATION_TOKENS for part in parts):
            return False
    # trim leading bullets for heuristics
    core = line.lstrip("-•·●▪‣o \t")
    words = [w for w in core.split() if w]
    if not words:
        return False
    upper_count = sum(1 for w in words if w[0].isupper())
    if upper_count >= max(1, len(words) - 1) and len(words) <= 8:
        return True
    return False


def _extract_year_span_from_text(line: str) -> str:
    norm = line.replace("–", "-").replace("—", "-")
    matches = EDU_YEAR_TOKEN.findall(norm)
    if not matches:
        return ""
    parsed = []
    for month, year in matches:
        m = month.strip() if month else ""
        y = year.strip().capitalize()
        if y.lower() == "present":
            y = "Present"
        parsed.append((m, y))
    if not parsed:
        return ""

    def fmt(pair: Tuple[str, str]) -> str:
        m, y = pair
        return f"{m.title()} {y}".strip() if m else y

    if len(parsed) >= 2:
        start = fmt(parsed[0])
        end = fmt(parsed[-1])
        if start == end:
            return start
        return f"{start} - {end}"
    return fmt(parsed[0])


def _split_company_location(text: str) -> Tuple[str, str]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) >= 2:
        tail = parts[-1].lower()
        if any(tok in tail for tok in LOCATION_TOKENS):
            return ", ".join(parts[:-1]).strip(), parts[-1].strip()
    return text.strip(), ""


def _extract_year_and_remainder(line: str) -> Tuple[str, str]:
    years = _extract_year_span_from_text(line)
    if not years:
        return "", line
    remainder = EDU_YEAR_TOKEN.sub("", line)
    remainder = re.sub(r"[-–—]+", " ", remainder)
    remainder = re.sub(r"\s+", " ", remainder).strip(" ,;-")
    return years, remainder


def extract_education(text: str) -> List[Dict[str, str]]:
    lines = _lines(text)
    items: List[Dict[str, str]] = []
    entry: Optional[Dict[str, str]] = None
    summary_parts: List[str] = []

    def clean_fragment(value: str) -> str:
        cleaned = re.sub(r"^[\-•*·●▪‣o]+\s*", "", value).strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip(" ,;-.")

    def has_degree_keyword(value: str) -> bool:
        lower = value.lower()
        keywords = (
            "b.tech", "bachelor", "master", "m.tech", "phd",
            "senior secondary", "secondary school", "higher secondary",
            "diploma", "engineering", "science", "commerce",
        )
        return any(keyword in lower for keyword in keywords)

    def finalize_entry() -> None:
        nonlocal entry, summary_parts
        if not entry:
            return

        # Move year info from summary if missing
        if not entry["years"]:
            for idx, part in enumerate(summary_parts):
                years_val = _extract_year_span_from_text(part)
                if years_val:
                    entry["years"] = years_val
                    remainder = YEAR_RANGE_PAT.sub("", part).strip(" ,;-")
                    summary_parts[idx] = remainder
                    break

        trimmed = [clean_fragment(part) for part in summary_parts]
        summary_parts[:] = [part for part in trimmed if part]

        if not entry["degree"]:
            for idx, part in enumerate(summary_parts):
                if has_degree_keyword(part):
                    entry["degree"] = part
                    summary_parts.pop(idx)
                    break

        if not entry["institution"] and summary_parts:
            entry["institution"] = summary_parts.pop(0)

        entry["summary"] = " ".join(summary_parts).strip()

        if any(entry.values()):
            items.append(entry)

        entry = None
        summary_parts = []

    for raw in lines:
        if not raw:
            continue
        line = raw.strip()
        if not line:
            continue

        is_institution_line = _looks_like_institution_line(line)
        is_degree_line = _looks_like_degree_line(line)
        is_location_line = _is_location_line(line)
        has_pipe = "|" in line
        new_entry_marker = has_pipe or is_institution_line or is_degree_line
        if (
            new_entry_marker
            and entry
            and not (
                (is_degree_line and not entry["degree"])
                or (is_institution_line and not entry["institution"])
                or (has_pipe and not entry["institution"])
            )
            and (entry["institution"] or entry["degree"] or entry["years"] or summary_parts)
        ):
            finalize_entry()

        if entry is None:
            entry = {
                "degree": "",
                "institution": "",
                "location": "",
                "years": "",
                "summary": "",
            }
            summary_parts = []

        if is_location_line and not is_institution_line:
            if entry["location"] and (entry["institution"] or entry["degree"] or entry["years"] or summary_parts):
                finalize_entry()
                entry = {
                    "degree": "",
                    "institution": "",
                    "location": "",
                    "years": "",
                    "summary": "",
                }
                summary_parts = []
            if not entry["location"]:
                entry["location"] = line.rstrip(", ")
                continue

        if has_pipe and not entry["institution"]:
            raw_pieces = [p.strip() for p in line.split("|") if p.strip()]
            cleaned: List[str] = []
            for piece in raw_pieces:
                cleaned_piece = clean_fragment(piece)
                if cleaned_piece:
                    cleaned.append(cleaned_piece)
            if cleaned:
                entry["institution"] = cleaned[0]
                if len(cleaned) > 1:
                    summary_parts.extend(cleaned[1:])
            continue

        years_val = _extract_year_span_from_text(line)
        if years_val and not entry["years"]:
            entry["years"] = years_val
            continue

        if not entry["degree"] and is_degree_line:
            match = _isolated_degree_match(line)
            if match:
                degree_text = _normalize_degree(match.group("deg"))
                prefix = clean_fragment(line[: match.start()])
                suffix = clean_fragment(line[match.end():])
                if prefix:
                    summary_parts.append(prefix)
                if suffix:
                    if not _is_location_line(suffix) and not EDU_SCORE_PAT.search(suffix) and not YEAR_RANGE_PAT.search(suffix):
                        degree_text = f"{degree_text} {suffix}".strip()
                    else:
                        summary_parts.append(suffix)
                entry["degree"] = degree_text
            else:
                entry["degree"] = clean_fragment(line)
            continue

        if not entry["institution"] and is_institution_line:
            entry["institution"] = clean_fragment(line)
            continue

        summary_parts.append(line)

    finalize_entry()

    deduped = _dedupe_dict_list(items, keys=("degree", "institution", "years", "location"))
    _debug("extract_education", f"found {len(deduped)}")
    return deduped


# ---- Experience extraction (lightweight) ----

MONTH_PATTERN = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
MONTH_NAMES = {
    "jan", "january", "feb", "february", "mar", "march", "apr", "april",
    "may", "jun", "june", "jul", "july", "aug", "august", "sep", "sept",
    "september", "oct", "october", "nov", "november", "dec", "december",
}
DATE_RANGE = rf"(?:{MONTH_PATTERN}\s+)?(?:19|20)\d{{2}}\s*(?:-|to|–|—)\s*(?:Present|present|Now|now|(?:{MONTH_PATTERN}\s+)?(?:19|20)\d{{2}})"
DATE_LINE_PAT = re.compile(DATE_RANGE, re.IGNORECASE)
LOCATION_LINE_PAT = re.compile(
    r"\b(?:India|United|USA|Pune|Mumbai|Chennai|Bangalore|Bengaluru|Hyderabad|Delhi|Gurgaon|Noida|London|Singapore|Toronto|Gujarat|Maharashtra|Telangana|Atlanta|Georgia|California|Mountain View|\bCA\b|\bGA\b)\b",
    re.IGNORECASE,
)
ROLE_KEYWORDS = {
    "engineer", "developer", "manager", "intern", "consultant", "analyst", "lead",
    "specialist", "architect", "director", "coordinator", "associate", "scientist",
    "administrator", "designer", "officer", "supervisor", "trainer", "head",
}
COMPANY_SUFFIXES = {
    "inc", "ltd", "llc", "pvt", "pvt ltd", "technologies", "solutions", "systems",
    "labs", "services", "consulting", "studios", "corp", "corporation", "limited", "gmbh",
}
EXPERIENCE_BULLETS = "-•*·●▪‣o"


def _role_keyword_in_line(line: str) -> bool:
    tokens = {
        re.sub(r"[^a-z]", "", word.lower())
        for word in line.split()
    }
    tokens.discard("")
    return bool(tokens & ROLE_KEYWORDS)


def _is_location_line(line: str) -> bool:
    lower = line.lower()
    if LOCATION_LINE_PAT.search(line):
        return True
    if "," in line:
        parts = [p.strip() for p in lower.split(",")]
        if any(p in LOCATION_TOKENS for p in parts[-2:]):
            return True
    return False


def _is_company_line(line: str) -> bool:
    words = line.split()
    if not words:
        return False
    lower = line.lower()
    if DATE_LINE_PAT.search(line):
        return False
    if _role_keyword_in_line(line):
        return False
    letters = sum(ch.isalpha() for ch in line)
    if letters and line.isupper() and len(words) <= 6:
        return True
    tail = words[-1].lower()
    if tail in COMPANY_SUFFIXES:
        return True
    if len(words) <= 4 and all(w[0].isupper() for w in words if w[0].isalpha()):
        return True
    if _is_location_line(line):
        company, location = _split_company_location(line)
        if company and location and not _is_location_line(company):
            return True
        return False
    return False


def _is_role_line(line: str) -> bool:
    if _is_location_line(line):
        return False
    words = line.split()
    if not words or len(words) > 10:
        return False
    lower = line.lower()
    if _role_keyword_in_line(line):
        return True
    if words[0][0].isupper() and any(kw.capitalize() == words[-1] for kw in ROLE_KEYWORDS):
        return True
    if len(words) <= 5 and sum(w[0].isupper() for w in words if w) >= max(1, len(words) - 2):
        return True
    return False


def extract_experience(text: str) -> List[Dict[str, str]]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    entries: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    pending_location: Optional[str] = None
    last_company: Optional[str] = None
    last_location: Optional[str] = None

    def new_entry() -> Dict[str, Any]:
        return {
            "role": "",
            "company": "",
            "location": "",
            "years": "",
            "details": [],
        }

    def claim_pending_location(target: Dict[str, Any]) -> None:
        nonlocal pending_location, last_location
        if pending_location and not target.get("location"):
            target["location"] = pending_location
            last_location = pending_location
            pending_location = None

    def ensure_entry() -> Dict[str, Any]:
        nonlocal current
        if current is None:
            current = new_entry()
            claim_pending_location(current)
        return current

    def append_detail(cur: Dict[str, Any], detail: str) -> None:
        detail = detail.strip()
        if not detail:
            return
        details: List[str] = cur.setdefault("details", [])
        if details and details[-1] and not details[-1].endswith(('.', '!', '?')) and detail[:1].islower():
            details[-1] = f"{details[-1]} {detail}".strip()
        else:
            details.append(detail)

    def finalize_entry() -> None:
        nonlocal current, last_company, last_location
        if not current:
            return
        summary = " ".join(current.get("details", [])).strip()
        entry = {
            "role": current.get("role", "").strip(),
            "company": current.get("company", "").strip(),
            "location": current.get("location", "").strip(),
            "years": current.get("years", "").strip(),
            "summary": summary,
        }
        if entry["role"] or entry["company"] or entry["years"]:
            entries.append(entry)
            if entry.get("company"):
                last_company = entry["company"]
            if entry.get("location"):
                last_location = entry["location"]
        current = None

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue

        normalized = stripped.lstrip(EXPERIENCE_BULLETS + " \t")
        normalized = normalized.lstrip("- ")
        normalized = normalized.strip()
        if not normalized:
            continue

        if _is_company_line(normalized):
            company, location = _split_company_location(normalized)
            prev_last_company = last_company
            same_company = (
                current is not None
                and current.get("company")
                and company
                and current["company"].lower() == company.lower()
            )
            has_material = bool(
                current
                and (current.get("years") or current.get("details") or current.get("summary"))
            )
            if current and current.get("company") and not same_company and has_material:
                finalize_entry()
                current = None
            if current is None:
                current = new_entry()
                claim_pending_location(current)
            if company:
                current["company"] = company
                last_company = company
            elif not current.get("company") and last_company:
                current["company"] = last_company
            if location:
                current["location"] = location
                last_location = location
            elif company and prev_last_company and company.lower() != prev_last_company.lower():
                current["location"] = ""
                pending_location = None
            claim_pending_location(current)
            continue

        if _is_location_line(normalized):
            if current is not None and not current.get("location"):
                current["location"] = normalized
                last_location = normalized
            else:
                pending_location = normalized
                last_location = normalized
            continue

        years_val, remainder = _extract_year_and_remainder(normalized)
        if years_val:
            entry = ensure_entry()
            if not entry.get("years"):
                entry["years"] = years_val
            if remainder:
                append_detail(entry, remainder)
            continue

        if _is_role_line(normalized):
            if (
                current
                and current.get("role")
                and normalized != current.get("role")
                and (current.get("years") or current.get("company") or current.get("details"))
            ):
                if current.get("company"):
                    last_company = current["company"]
                if current.get("location"):
                    last_location = current["location"]
                finalize_entry()
                current = new_entry()
                if last_company and not current.get("company"):
                    current["company"] = last_company
                if (
                    last_location
                    and last_company
                    and current.get("company")
                    and current["company"].lower() == last_company.lower()
                    and not current.get("location")
                ):
                    current["location"] = last_location
            entry = ensure_entry()
            entry["role"] = normalized
            continue

        if stripped and stripped[0] in EXPERIENCE_BULLETS:
            if current is None:
                continue
            append_detail(current, normalized)
            continue

        entry = ensure_entry()
        append_detail(entry, normalized)

    finalize_entry()

    compact: List[Dict[str, str]] = []
    for entry in entries:
        compact.append(
            {
                "role": entry.get("role", ""),
                "company": entry.get("company", ""),
                "location": entry.get("location", ""),
                "years": entry.get("years", ""),
                "summary": entry.get("summary", ""),
            }
        )

    exp = _dedupe_dict_list(compact, keys=("role", "company", "years", "location"))
    _debug("extract_experience", f"found {len(exp)}")
    return exp

def _dedupe_dict_list(lst: List[Dict], keys: tuple) -> List[Dict]:
    seen = set()
    out = []
    for d in lst:
        sig = tuple((d.get(k,"") or "").lower() for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(d)
    return out


# ---- Master parse ----

def parse_cv_text(raw_text: str) -> Dict:
    _debug("parse", "start")
    text = _normalize(raw_text)
    sections = split_sections(text)

    name = extract_name(text)
    emails = extract_emails(text)
    phones = extract_phones(text)
    links = extract_links(text)

    skills_source = sections.get("skills", "") or text
    skills = extract_skills(skills_source)

    edu_source = sections.get("education", "") or text
    edu = extract_education(edu_source)
    exp_source = sections.get("experience", "") or sections.get("work experience", "") or text
    exp = extract_experience(exp_source)

    projects_source = (
        sections.get("projects", "")
        or sections.get("project experience", "")
        or sections.get("university projects", "")
        or ""
    )
    projects = extract_projects(projects_source) if projects_source else []

    publications_source = (
        sections.get("research publications", "")
        or sections.get("publications", "")
        or ""
    )
    publications = extract_publications(publications_source) if publications_source else []

    summary = sections.get("summary", "") or sections.get("profile", "")
    _debug("parse", "complete")

    return {
        "name": name,
        "contact": {
            "emails": emails,
            "phones": phones,
            "links": links,
        },
        "summary": summary,
        "skills": skills,
        "education": edu,
        "experience": exp,
        "projects": projects,
        "publications": publications,
        "sections_found": list(sections.keys())
    }

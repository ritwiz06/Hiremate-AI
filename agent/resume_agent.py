"""Simple rule-based resume agent for interactive querying.

This helper sits on top of the parsed resume JSON and provides
lightweight responses without depending on an external LLM. It is
designed so that we can later swap in richer models (e.g., KG/KGNN or an
LLM) while keeping the same interface.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


class ResumeAgent:
    """Responds to user prompts using parsed resume data."""

    def __init__(self, resume: Dict[str, Any] | None, jobs: List[Dict[str, Any]] | None = None) -> None:
        self.resume = resume or {}
        self.jobs = jobs or []

    # -- public API -----------------------------------------------------
    def answer(self, prompt: str) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            return "I need something to react to. Ask me about experience, skills, education, or contact details."

        lowered = prompt.lower()

        if any(word in lowered for word in ("hi", "hello", "hey")) and len(lowered.split()) <= 5:
            return self._greeting()

        if "help" in lowered or "what can" in lowered:
            return self._help_message()

        if any(keyword in lowered for keyword in ("summary", "overview", "about", "profile")):
            return self._summarise()

        if any(keyword in lowered for keyword in ("contact", "email", "phone", "reach")):
            return self._format_contact()

        if any(keyword in lowered for keyword in ("skill", "tech", "technology", "stack")):
            return self._format_skills()

        if any(keyword in lowered for keyword in ("education", "degree", "school", "college")):
            return self._format_education()

        if self.jobs and any(keyword in lowered for keyword in ("job match", "job suggestion", "job opening", "matching job", "open role", "jobs", "job opportunities", "open jobs")):
            return self._format_job_matches()

        if any(keyword in lowered for keyword in ("experience", "job", "work", "role", "project you did at")):
            return self._format_experience()

        if "project" in lowered:
            return self._format_projects()

        # lightweight keyword look-ups for specific entities
        if "years" in lowered or "timeline" in lowered:
            return self._format_experience(include_years_only=True)

        # Fallback: attempt to answer by detecting skill or company mention
        skill_response = self._match_skill(lowered)
        if skill_response:
            return skill_response

        company_response = self._match_company(lowered)
        if company_response:
            return company_response

        return (
            "I'm set up to discuss the candidate's summary, skills, education, experience, projects, contact details, "
            "and job matches. Try asking something like 'What are the key skills?' or 'Any matching job openings?'"
        )

    # -- formatting helpers --------------------------------------------
    def _greeting(self) -> str:
        name = self.resume.get("name")
        if name:
            return f"Hello! I'm reviewing {name}'s resume. Ask me anything about their experience, skills, or education."
        return "Hello! Ask me anything about the candidate's experience, skills, or education."

    def _help_message(self) -> str:
        topics = [
            "summary",
            "skills",
            "experience",
            "education",
            "projects",
            "contact details",
        ]
        topics.append("job matches")
        return "I can talk about " + ", ".join(topics[:-1]) + f", or {topics[-1]}."

    def _summarise(self) -> str:
        summary = self.resume.get("summary")
        if summary:
            return summary.strip()
        name = self.resume.get("name") or "the candidate"
        return f"I don't have a dedicated summary for {name}, but I can walk you through their skills, experience, or education."

    def _format_skills(self) -> str:
        skills = self.resume.get("skills") or []
        if not skills:
            return "I don't see a skills section on this resume."
        if isinstance(skills, str):
            return skills
        if isinstance(skills, Sequence):
            return "Key skills: " + ", ".join(skills)
        return "I see some skills listed but can't format them right now."

    def _format_education(self) -> str:
        education = self.resume.get("education") or []
        if not isinstance(education, Iterable) or not education:
            return "I couldn't find any education history in the parsed data."

        lines = []
        for entry in education:
            if not isinstance(entry, dict):
                continue
            degree = entry.get("degree")
            institution = entry.get("institution")
            years = entry.get("years")
            location = entry.get("location")
            chunk = []
            if degree:
                chunk.append(degree)
            if institution:
                chunk.append(f"at {institution}")
            if location:
                chunk.append(f"({location})")
            if years:
                chunk.append(f"[{years}]")
            summary = entry.get("summary") or ""
            text = " ".join(chunk).strip()
            if summary:
                text = f"{text} - {summary}"
            if text:
                lines.append(text)
        return "Education details:\n- " + "\n- ".join(lines) if lines else "I couldn't format the education details just yet."

    def _format_experience(self, include_years_only: bool = False) -> str:
        experience = self.resume.get("experience") or []
        if not isinstance(experience, Iterable) or not experience:
            return "I don't see experience entries in the parsed data."

        lines = []
        for entry in experience:
            if not isinstance(entry, dict):
                continue
            role = entry.get("role")
            company = entry.get("company")
            location = entry.get("location")
            years = entry.get("years")
            summary = entry.get("summary") or ""
            if include_years_only and years:
                lines.append(f"{role or company or 'Experience'} - {years}")
                continue

            pieces = []
            if role:
                pieces.append(role)
            if company:
                pieces.append(f"at {company}")
            if location:
                pieces.append(f"({location})")
            if years:
                pieces.append(f"[{years}]")
            text = " ".join(pieces).strip()
            if summary:
                text = f"{text} - {summary}" if text else summary
            if text:
                lines.append(text)
        return "Work experience:\n- " + "\n- ".join(lines) if lines else "I couldn't format the experience details just yet."

    def _format_projects(self) -> str:
        projects = self.resume.get("projects") or []
        if not isinstance(projects, Iterable) or not projects:
            return "I couldn't find project details in the parsed data."

        lines = []
        for project in projects:
            if not isinstance(project, dict):
                continue
            name = project.get("name")
            detail = project.get("details")
            if name and detail:
                lines.append(f"{name} - {detail}")
            elif name:
                lines.append(name)
        return "Projects:\n- " + "\n- ".join(lines) if lines else "I couldn't format any projects."

    def _format_contact(self) -> str:
        contact = self.resume.get("contact") or {}
        if not isinstance(contact, dict) or not contact:
            return "I didn't find contact information in the parsed data."

        parts: List[str] = []
        emails = contact.get("emails") or []
        phones = contact.get("phones") or []
        links = contact.get("links") or {}

        if emails:
            joined = ", ".join(emails)
            parts.append(f"Email: {joined}")
        if phones:
            joined = ", ".join(phones)
            parts.append(f"Phone: {joined}")
        if isinstance(links, dict):
            urls = links.get("urls") or []
            linkedin = links.get("linkedin") or []
            github = links.get("github") or []
            if urls:
                parts.append(f"Links: {', '.join(urls)}")
            if linkedin:
                parts.append(f"LinkedIn: {', '.join(linkedin)}")
            if github:
                parts.append(f"GitHub: {', '.join(github)}")

        return "\n".join(parts) if parts else "Contact details were not available."

    # -- targeted matching ---------------------------------------------
    def _match_skill(self, lowered_prompt: str) -> str | None:
        skills = self.resume.get("skills") or []
        if not isinstance(skills, Sequence):
            return None
        matched = [skill for skill in skills if skill and skill.lower() in lowered_prompt]
        if matched:
            return f"Yes, the candidate lists {', '.join(matched)} among their skills."
        return None

    def _match_company(self, lowered_prompt: str) -> str | None:
        experience = self.resume.get("experience") or []
        if not isinstance(experience, Sequence):
            return None
        for entry in experience:
            if not isinstance(entry, dict):
                continue
            company = entry.get("company")
            if company and company.lower() in lowered_prompt:
                role = entry.get("role") or "a role"
                years = entry.get("years")
                if years:
                    return f"At {company}, the candidate held {role} during {years}."
                return f"The candidate held {role} at {company}."
        return None

    def _format_job_matches(self) -> str:
        if not self.jobs:
            return "I don't have matching jobs calculated yet."
        lines = []
        for job in self.jobs:
            title = job.get("title") or "Role"
            company = job.get("company") or "Company"
            location = job.get("location")
            raw_score = job.get("score")
            score = f"{float(raw_score):.2f}" if isinstance(raw_score, (int, float)) else "n/a"
            overlap = job.get("overlap") or []
            bullet = f"{title} at {company}"
            if location:
                bullet += f" ({location})"
            if score != "n/a":
                bullet += f" [score {score}]"
            if overlap:
                bullet += f" - shared skills: {', '.join(overlap)}"
            lines.append(bullet)
        return "Top job matches:\n- " + "\n- ".join(lines)

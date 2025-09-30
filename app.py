# app.py

import contextlib
import hashlib
import io
import os
import tempfile
from pathlib import Path

import streamlit as st

from agent import ResumeAgent
from data_loader import load_resume
from nlp.parser import parse_cv_text, set_debug
from services.job_matcher import match_jobs

try:
    from nlp.llm import (
        refine_experience_via_ollama,
        refine_experience_via_groq,
        _log_comparison,
    )
except Exception:  # pragma: no cover - optional LLM helpers missing
    refine_experience_via_ollama = None
    refine_experience_via_groq = None
    _log_comparison = None

st.set_page_config(page_title="HireMate CV Parser", page_icon="🧾", layout="wide")
st.title("HireMate")

uploaded = st.file_uploader("Upload your resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])

if uploaded is not None:
    debug_enabled = st.checkbox("Enable parser debug logs", value=True)

    suffix = "." + uploaded.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        text = load_resume(tmp_path)
    except Exception as e:
        st.error(f"Parsing failed: {e}")
    else:
        log_buffer = io.StringIO()
        parsed_data = None
        parse_error = None
        st.session_state["resume_text"] = text
        resume_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
        resume_changed = st.session_state.get("_resume_hash") != resume_hash
        if resume_changed:
            st.session_state["_resume_hash"] = resume_hash
            st.session_state.pop("agent_history", None)
            st.session_state.pop("experience_cache", None)
            st.session_state.pop("parsed_data", None)
            st.session_state.pop("last_parse_log", None)
            st.session_state.pop("job_matches_cache", None)

        cached_data = st.session_state.get("parsed_data") if not resume_changed else None

        if cached_data is None:
            with contextlib.redirect_stdout(log_buffer):
                try:
                    set_debug(debug_enabled)
                    parsed_data = parse_cv_text(text)
                except Exception as err:  # keep UI responsive even if parsing fails
                    parse_error = err
                    parsed_data = None
                finally:
                    set_debug(False)

            log_output = log_buffer.getvalue()
            if parse_error:
                log_output = log_output.rstrip() + ("\n" if log_output else "") + f"[parser] ERROR: {parse_error}\n"
            else:
                st.session_state["parsed_data"] = parsed_data
                if parsed_data is not None:
                    base_experience = parsed_data.get("experience", []) or []
                    st.session_state["experience_cache"] = {"Heuristic": base_experience}
                    st.session_state["job_matches_cache"] = {}
                st.session_state["last_parse_log"] = log_output
        else:
            parsed_data = cached_data
            log_output = st.session_state.get("last_parse_log", "")
            parse_error = None

        col_main, col_logs = st.columns([2, 1], gap="large")

        if parsed_data is not None:
            experience_cache = st.session_state.get("experience_cache")
            if experience_cache is None:
                base_experience = parsed_data.get("experience", []) or []
                experience_cache = {"Heuristic": base_experience}
                st.session_state["experience_cache"] = experience_cache
        if parse_error:
            st.session_state.pop("parsed_data", None)
            st.session_state.pop("experience_cache", None)
            st.session_state.pop("last_parse_log", None)
            st.session_state.pop("job_matches_cache", None)

        with col_main:
            st.success("Resume text extracted.")
            st.text_area("Extracted Text", value=text, height=500)
            st.download_button("Download as .txt", data=text, file_name="parsed_resume.txt")

            active_data = st.session_state.get("parsed_data")
            if active_data:
                experience_cache = st.session_state.get("experience_cache", {})
                base_experience = experience_cache.get("Heuristic", [])

                options = ["Heuristic"]
                model_actions = {}

                if refine_experience_via_ollama is not None:
                    ollama_fn = refine_experience_via_ollama

                    def _run_ollama(fn=ollama_fn):
                        return fn(st.session_state.get("resume_text", ""), base_experience)

                    options.append("Ollama")
                    model_actions["Ollama"] = _run_ollama

                if refine_experience_via_groq is not None:
                    groq_fn = refine_experience_via_groq
                    groq_primary = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
                    groq_secondary = os.getenv(
                        "GROQ_SECOND_MODEL_NAME", "deepseek-r1-distill-llama-70b"
                    )
                    groq_tertiary = os.getenv(
                        "GROQ_THIRD_MODEL_NAME", "openai/gpt-oss-120b"
                    )
                    for model in [groq_primary, groq_secondary, groq_tertiary]:
                        if not model:
                            continue
                        label = f"Groq: {model}"
                        if label not in options:
                            options.append(label)

                            model_actions[label] = lambda model_id=model, fn=groq_fn: fn(
                                st.session_state.get("resume_text", ""),
                                base_experience,
                                model=model_id,
                            )

                selection = st.selectbox(
                    "Select model to inspect",
                    options=options,
                    index=0,
                    key="experience_view_selector",
                )

                st.subheader("Structured Resume Snapshot")
                display_data = dict(active_data)

                if selection not in experience_cache:
                    action = model_actions.get(selection)
                    if action is not None:
                        refined = action() or []
                        experience_cache[selection] = refined or base_experience
                        st.session_state["experience_cache"] = experience_cache
                        if _log_comparison is not None:
                            _log_comparison(base_experience, {selection: refined})

                selected_experience = experience_cache.get(selection, base_experience)
                if selected_experience is not None:
                    display_data = {**display_data, "experience": selected_experience}
                st.json(display_data)

                st.subheader(f"Experience - {selection}")
                st.json(experience_cache.get(selection, base_experience))

                job_cache = st.session_state.setdefault("job_matches_cache", {})
                resume_variant = display_data
                if selection not in job_cache:
                    job_cache[selection] = match_jobs(resume_variant)
                job_matches = job_cache.get(selection, [])

                st.subheader("Suggested Jobs")
                if job_matches:
                    for job in job_matches:
                        title = job.get("title") or "Role"
                        company = job.get("company") or "Company"
                        location = job.get("location") or "Location"
                        score = job.get("score")
                        overlap = job.get("overlap") or []
                        description = job.get("description")
                        st.markdown(
                            f"**{title}** at {company} ({location})\\nScore: {score}\\nShared skills: {', '.join(overlap) if overlap else 'n/a'}"
                        )
                        if description:
                            st.caption(description)
                        st.markdown("---")
                else:
                    st.info("No job matches yet. Upload a resume with clear skills to see suggestions.")

                try:
                    st.divider()
                except AttributeError:
                    st.markdown("---")

                st.subheader("Resume Agent")

                agent_payload = resume_variant
                agent = ResumeAgent(agent_payload, jobs=job_matches)

                history_key = "agent_history"
                history = st.session_state.setdefault(history_key, [])

                if not history:
                    intro = agent.answer("hello")
                    history.append({"role": "assistant", "content": intro})

                def _is_short_greeting(text: str) -> bool:
                    lowered = text.strip().lower()
                    if not lowered:
                        return False
                    stripped = lowered.rstrip("!.")
                    return stripped in {"hi", "hello", "hey"}

                for message in history:
                    role = message.get("role", "assistant")
                    content = message.get("content", "")
                    with st.chat_message(role):
                        st.markdown(content)

                user_prompt = st.chat_input("Ask about this resume", key="agent_chat_input")
                if user_prompt:
                    history.append({"role": "user", "content": user_prompt})
                    if _is_short_greeting(user_prompt) and any(msg.get("role") == "assistant" for msg in history[:-1]):
                        reply = "Hi again! Let me know what you want to explore - skills, experience, education, projects, or job matches."
                    else:
                        reply = agent.answer(user_prompt)
                    history.append({"role": "assistant", "content": reply})
                    st.session_state[history_key] = history
                    rerun_fn = getattr(st, "rerun", None)
                    if callable(rerun_fn):
                        rerun_fn()
                    else:
                        legacy_rerun = getattr(st, "experimental_rerun", None)
                        if callable(legacy_rerun):
                            legacy_rerun()

            if parse_error:
                st.warning(f"Structured parsing encountered an issue: {parse_error}")

        with col_logs:
            log_tab, = st.tabs(["Logs"])
            with log_tab:
                if log_output.strip():
                    st.code(log_output.rstrip())
                else:
                    if debug_enabled:
                        st.info("No debug output was produced for this file.")
                    else:
                        st.info("Enable parser debug logs to see step-by-step output here.")

    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

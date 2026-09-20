"""
Forkcast — decision simulation and cross-disciplinary research.

Runs on Streamlit Community Cloud. Groq calls happen server-side, so the API
key never reaches the browser and there is no CORS to fight.
"""

import json
import os
import re
from datetime import datetime

import requests
import streamlit as st

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_RESEARCH = "groq/compound"
MODEL_REASONING = "openai/gpt-oss-120b"
LOG_PATH = "forkcast_log.json"
MAX_LOG_ENTRIES = 50

# --------------------------------------------------------------------------
# Output schemas — Groq enforces these, so the response is always valid JSON
# --------------------------------------------------------------------------

DECISION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "decision_analysis",
        "schema": {
            "type": "object",
            "properties": {
                "recommendation": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "rationale": {"type": "string"},
                        "key_assumption": {"type": "string"},
                        "counter_argument": {"type": "string"},
                        "weighted_outlook": {"type": "string"},
                        "tail_risk": {"type": "string"},
                        "revisit_timeframe": {"type": "string"},
                    },
                    "required": [
                        "summary", "confidence", "rationale", "key_assumption",
                        "counter_argument", "weighted_outlook", "tail_risk",
                        "revisit_timeframe",
                    ],
                },
                "scenarios": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "narrative": {"type": "string"},
                            "likelihood": {"type": "number"},
                            "estimate": {"type": "string"},
                            "estimate_basis": {"type": "string"},
                            "tradeoffs": {"type": "array", "items": {"type": "string"}},
                            "risks": {"type": "array", "items": {"type": "string"}},
                            "pivot_condition": {"type": "string"},
                        },
                        "required": [
                            "name", "narrative", "likelihood", "estimate",
                            "estimate_basis", "tradeoffs", "risks", "pivot_condition",
                        ],
                    },
                },
                "sources": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["recommendation", "scenarios", "sources"],
        },
    },
}

EXPLORE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "cross_discipline_explanation",
        "schema": {
            "type": "object",
            "properties": {
                "overview": {"type": "string"},
                "lenses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "discipline": {"type": "string"},
                            "explanation": {"type": "string"},
                        },
                        "required": ["discipline", "explanation"],
                    },
                },
                "connections": {"type": "string"},
                "open_questions": {"type": "array", "items": {"type": "string"}},
                "sources": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["overview", "lenses", "connections", "sources"],
        },
    },
}

DECISION_SYSTEM = (
    "You are a rigorous decision-simulation analyst for engineering and AI teams, working with "
    "three internal specialists before you answer: a cost/economics analyst, a risk analyst, and a "
    "feasibility/execution analyst. Have each specialist independently assess the decision using the "
    "research notes. Then, as a skeptical devil's advocate, challenge whichever path the specialists "
    "lean toward — find its weakest point. Finally, as lead analyst, synthesize all of this into "
    "exactly 3 distinct, genuinely different final scenarios (not optimistic/neutral/pessimistic "
    "restatements of one path) and one overall recommendation naming: the key assumption it depends "
    "on; the single strongest argument against it; a likelihood-weighted outlook that synthesizes all "
    "3 scenarios into one realistic overall expectation (not just the top pick in isolation); the "
    "specific tail risk worth guarding against even though it is not the most likely path; and a rough "
    "sense of when this decision is worth revisiting. For each scenario give a concrete estimate (cost, "
    "timeframe, or both — use reasoned round numbers where research gives no exact figure), a one-line "
    "note on what that estimate is grounded in, and the specific pivot_condition that would need to "
    "change for a different scenario to win instead. Likelihoods should reflect genuine assessment, "
    "not an even split."
)

EXPLORE_SYSTEM = (
    "You are a cross-disciplinary research analyst. Given a question and research notes: identify "
    "which distinct fields genuinely bear on it — only ones that actually apply, never force a field "
    "in just to pad the list — and explain the question clearly through each one. Then explain how the "
    "lenses connect or reinforce each other; that synthesis is the valuable part, not parallel "
    "restatements. Note genuinely open or actively debated aspects only if they truly exist — do not "
    "invent controversy in settled science. Draft your explanation, then critique it for accuracy and "
    "revise before finalizing."
)

# --------------------------------------------------------------------------
# Groq plumbing
# --------------------------------------------------------------------------


def get_api_key() -> str:
    """Prefer the deployed secret; fall back to a key typed into the sidebar."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "") or st.session_state.get("manual_key", "")


def call_groq(model: str, messages: list, **options) -> dict:
    key = get_api_key()
    if not key:
        raise RuntimeError("No Groq API key found. Add one in the sidebar, or set GROQ_API_KEY in your app secrets.")

    def do_request(opts):
        payload = {"model": model, "messages": messages, "temperature": 0.4, **opts}
        return requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=180,
        )

    resp = do_request(options)

    # Groq returns 413 when the *requested max_tokens ceiling* — not the actual
    # prompt — exceeds what a single request is allowed to ask for on this
    # account. A two-word prompt fails just as fast as a long one if max_tokens
    # is set above that ceiling. Rather than guess the right number, read the
    # real limit out of Groq's own error message and retry once at a safe size.
    if resp.status_code == 413 and "max_tokens" in options:
        try:
            err_text = resp.json().get("error", {}).get("message", "")
        except Exception:
            err_text = resp.text
        match = re.search(r"[Ll]imit[:\s]+(\d+)", err_text)
        if match:
            safe_tokens = max(256, int(match.group(1)) - 400)  # headroom for input tokens
            if safe_tokens < options["max_tokens"]:
                resp = do_request({**options, "max_tokens": safe_tokens})

    if resp.status_code == 429:
        raise RuntimeError(
            "Groq rate limit hit — too many requests or tokens used this minute. Wait a moment and retry, "
            "or raise the limit by adding a card at console.groq.com (free, no minimum spend)."
        )
    if not resp.ok:
        detail = resp.text
        try:
            detail = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            pass
        raise RuntimeError(f"Groq API error ({resp.status_code}): {detail}")
    return resp.json()


def strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


def research(prompt: str) -> tuple:
    data = call_groq(MODEL_RESEARCH, [{"role": "user", "content": prompt}], max_tokens=800)
    msg = data.get("choices", [{}])[0].get("message", {})
    return msg.get("content", ""), msg.get("executed_tools")


def reason(system: str, user: str, schema: dict, max_tokens: int, effort: str) -> tuple:
    data = call_groq(
        MODEL_REASONING,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        reasoning_format="parsed",
        reasoning_effort=effort,
        response_format=schema,
    )
    msg = data.get("choices", [{}])[0].get("message", {})
    raw = strip_fences(msg.get("content", ""))
    trace = msg.get("reasoning") or msg.get("reasoning_content") or ""
    try:
        return json.loads(raw), trace
    except json.JSONDecodeError:
        raise RuntimeError("Couldn't parse the model's output as JSON. Try running it again.")


# --------------------------------------------------------------------------
# Decision log — session state is the source of truth; the file is a bonus
# --------------------------------------------------------------------------


def load_log() -> list:
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []


def save_log(log: list) -> None:
    try:
        with open(LOG_PATH, "w", encoding="utf-8") as fh:
            json.dump(log, fh, indent=2)
    except Exception:
        pass  # ephemeral disk on Community Cloud; session state still holds it


def calibration(log: list):
    scored = [e for e in log if e.get("matched") and e["matched"] != "__none__" and e.get("scenarios")]
    if not scored:
        return None
    correct = sum(
        1 for e in scored
        if max(e["scenarios"], key=lambda s: s.get("likelihood", 0))["name"] == e["matched"]
    )
    return correct, len(scored)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

st.set_page_config(page_title="Forkcast", page_icon="🔱", layout="centered")

st.markdown(
    """
    <style>
      .fc-tag { display:inline-block; font-size:11.5px; font-family:monospace; color:#E0A458;
                border:1px solid #E0A458; border-radius:20px; padding:2px 10px; }
      .fc-lens { display:inline-block; font-size:11.5px; font-family:monospace; color:#4FA8A0;
                 border:1px solid #4FA8A0; border-radius:20px; padding:2px 10px; margin-bottom:6px; }
      .fc-bar-track { height:5px; background:#23262C; border-radius:3px; margin:8px 0 4px 0; }
      .fc-bar-fill { height:100%; border-radius:3px; }
      .fc-est { font-family:monospace; font-size:12.5px; color:#4FA8A0; }
      .fc-against { border-left:2px solid #C4614B; padding-left:10px; font-size:14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

for key, default in [
    ("log", None), ("result", None), ("trace", ""), ("mode", "Decision"),
    ("last_input", ""), ("followups", []), ("manual_key", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default
if st.session_state.log is None:
    st.session_state.log = load_log()

# ---------------------------------- sidebar --------------------------------

with st.sidebar:
    st.subheader("Setup")
    secret_key = ""
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    secret_key = secret_key or os.environ.get("GROQ_API_KEY", "")

    if secret_key:
        st.success("Groq key loaded from secrets", icon="✅")
    else:
        st.text_input("Groq API key", type="password", key="manual_key")
        if st.session_state.manual_key:
            st.caption("Key set for this session.")
        else:
            st.caption("Free key at console.groq.com/keys — or set GROQ_API_KEY in app secrets so it never reaches the browser.")

    risk = st.radio("Risk tolerance", ["conservative", "balanced", "aggressive"], index=1)

    st.divider()
    st.subheader("Decision log")
    cal = calibration(st.session_state.log)
    if cal:
        st.metric("Top pick was right", f"{cal[0]} of {cal[1]}")
    if not st.session_state.log:
        st.caption("Nothing logged yet.")
    for entry in st.session_state.log:
        with st.expander(entry["decision"][:60] + ("…" if len(entry["decision"]) > 60 else "")):
            st.caption(entry.get("summary", ""))
            st.caption(entry.get("date", ""))
            names = [s["name"] for s in entry.get("scenarios", [])]
            if names:
                choice = st.selectbox(
                    "Which scenario came closest?",
                    ["(not yet)"] + names + ["None of these"],
                    index=0 if not entry.get("matched") else (
                        names.index(entry["matched"]) + 1 if entry["matched"] in names else len(names) + 1
                    ),
                    key=f"match_{entry['id']}",
                )
                resolved = "" if choice == "(not yet)" else ("__none__" if choice == "None of these" else choice)
                if resolved != entry.get("matched", ""):
                    entry["matched"] = resolved
                    save_log(st.session_state.log)
                    st.rerun()
            note = st.text_input("What actually happened?", value=entry.get("outcome", ""), key=f"out_{entry['id']}")
            if note != entry.get("outcome", ""):
                entry["outcome"] = note
                save_log(st.session_state.log)
            if st.button("Delete", key=f"del_{entry['id']}"):
                st.session_state.log = [e for e in st.session_state.log if e["id"] != entry["id"]]
                save_log(st.session_state.log)
                st.rerun()

    if st.session_state.log:
        st.download_button(
            "Export log (JSON)",
            data=json.dumps(st.session_state.log, indent=2),
            file_name="forkcast-log.json",
            mime="application/json",
        )

    restore_file = st.file_uploader("Restore a log export", type="json", label_visibility="visible")
    if restore_file is not None:
        try:
            restored = json.load(restore_file)
            if isinstance(restored, list):
                st.session_state.log = restored
                save_log(st.session_state.log)
                st.success(f"Restored {len(restored)} entries.")
            else:
                st.error("That file doesn't look like a Forkcast log export.")
        except Exception:
            st.error("Couldn't read that file as JSON.")

    st.caption(
        "This app's storage resets on redeploy — export before you push changes, restore after. "
        "A real fix (survives redeploys automatically) needs an external database; ask me to wire one up "
        "when you're ready, it's a bigger step than this stopgap."
    )

# ----------------------------------- main ----------------------------------

st.title("Forkcast")
st.caption(
    "Feed it a real decision or an open question. It researches live data, debates it internally "
    "across several angles, then maps out scenarios or breaks the answer down across every field that applies."
)

with st.expander("Why not just ask ChatGPT or Claude?"):
    st.write(
        "Honestly, for any single question, a good enough prompt gets you most of the way there — the "
        "underlying model isn't exclusive to this app. What Forkcast actually does differently: it forces "
        "the same rigor every time — live research, three specialists arguing their own angle, a devil's "
        "advocate challenge — whether or not you remembered to ask for all of that today. And it "
        "remembers. Every decision below gets logged; once you've logged a few real outcomes, the sidebar "
        "shows how often the top pick was actually right. A stateless chat has no track record to check "
        "against. This one does, and it grows every time you use it."
    )

mode = st.radio("Mode", ["Decision", "Explore a question"], horizontal=True, label_visibility="collapsed")
is_decision = mode == "Decision"

user_input = st.text_area(
    "The decision" if is_decision else "The question",
    placeholder=(
        "e.g. Should we fine-tune our own model for support-ticket classification, or keep calling a hosted API?"
        if is_decision else "e.g. How is protein actually made, at the molecular level?"
    ),
    height=110,
)

if st.button("Run simulation" if is_decision else "Explore this", type="primary"):
    if not user_input.strip():
        st.warning("Type something in first.")
    else:
        try:
            with st.status("Researching…", expanded=True) as status:
                if is_decision:
                    rprompt = (
                        f'Research current, real information relevant to evaluating this technical decision: '
                        f'"{user_input}". Prioritize authoritative sources appropriate to the subject — official '
                        f'docs, benchmarks, and pricing pages for technical questions, recent industry reporting '
                        f'for market questions. When a search result looks especially relevant, visit the actual '
                        f'page rather than relying on the snippet alone. Look for concrete benchmarks, pricing, '
                        f'timelines, precedents, recent incidents, or competing approaches — numbers over '
                        f'generalities where you can find them. Summarize the specific facts and data points you '
                        f'find, with sources.'
                    )
                else:
                    rprompt = (
                        f'Research current, accurate background information relevant to this question, across '
                        f'whichever fields genuinely apply: "{user_input}". Prioritize authoritative sources '
                        f'appropriate to each field. When a search result looks especially relevant, visit the '
                        f'actual page rather than relying on the snippet alone. Summarize the key facts and '
                        f'mechanisms each relevant field would point to, with sources.'
                    )
                notes, _tools = research(rprompt)
                preview = (notes[:240] + "…") if len(notes) > 240 else notes
                st.write(preview if preview else "No additional research surfaced beyond what's already known.")

                status.update(label="Specialists debating, critiquing, revising…")
                if is_decision:
                    system = DECISION_SYSTEM + (
                        f" The user's stated risk tolerance is {risk} — let this influence which scenario you name "
                        f"as the top recommendation, but keep every scenario's likelihood and framing objective "
                        f"regardless of that tolerance."
                    )
                else:
                    system = EXPLORE_SYSTEM
                parsed, trace = reason(
                    system,
                    f"{'Decision' if is_decision else 'Question'}: {user_input}\n\n"
                    f"Research notes:\n{notes or '(no additional research found)'}\n\nProduce the analysis now.",
                    DECISION_SCHEMA if is_decision else EXPLORE_SCHEMA,
                    5000 if is_decision else 3500,
                    "high" if is_decision else "medium",
                )
                st.write(
                    "Weighed cost, risk, and feasibility angles, ran a devil's-advocate challenge against the "
                    "strongest path, then synthesized a final recommendation."
                    if is_decision else
                    "Identified the disciplines that genuinely apply and drafted, then double-checked, an "
                    "explanation through each."
                )
                status.update(label="Done", state="complete")

            parsed["_mode"] = "decision" if is_decision else "explore"
            st.session_state.result = parsed
            st.session_state.trace = trace
            st.session_state.last_input = user_input
            st.session_state.followups = []

            if is_decision:
                st.session_state.log.insert(0, {
                    "id": str(datetime.now().timestamp()),
                    "decision": user_input,
                    "date": datetime.now().strftime("%d %b %Y"),
                    "summary": parsed["recommendation"]["summary"],
                    "confidence": parsed["recommendation"]["confidence"],
                    "scenarios": [
                        {"name": s["name"], "likelihood": s["likelihood"]} for s in parsed["scenarios"]
                    ],
                    "outcome": "",
                    "matched": "",
                })
                st.session_state.log = st.session_state.log[:MAX_LOG_ENTRIES]
                save_log(st.session_state.log)
        except Exception as exc:
            st.error(str(exc))

# --------------------------------- results ---------------------------------

result = st.session_state.result
if result:
    st.divider()

    if result["_mode"] == "explore":
        st.subheader(result["overview"])
        st.markdown("##### Through each lens")
        for lens in result.get("lenses", []):
            st.markdown(f'<span class="fc-lens">{lens["discipline"]}</span>', unsafe_allow_html=True)
            st.write(lens["explanation"])
        if result.get("connections"):
            st.markdown("##### How they connect")
            st.write(result["connections"])
        if result.get("open_questions"):
            st.markdown("##### Still genuinely open")
            for q in result["open_questions"]:
                st.markdown(f"- {q}")
    else:
        rec = result["recommendation"]
        st.subheader(rec["summary"])
        st.markdown(f'<span class="fc-tag">{rec["confidence"]} confidence</span>', unsafe_allow_html=True)
        st.write("")
        st.write(rec["rationale"])
        st.markdown(f"**Betting on:** {rec['key_assumption']}")
        st.markdown(f'<div class="fc-against"><b>Strongest case against:</b> {rec["counter_argument"]}</div>',
                    unsafe_allow_html=True)

        with st.expander("Deeper analysis — weighted outlook, tail risk, revisit timing"):
            st.markdown(f"**Weighted outlook:** {rec['weighted_outlook']}")
            st.markdown(f"**Tail risk:** {rec['tail_risk']}")
            st.markdown(f"**Worth revisiting:** {rec['revisit_timeframe']}")

        st.markdown("##### How it could play out")
        for sc in result["scenarios"]:
            pct = sc["likelihood"]
            color = "#4FA8A0" if pct >= 55 else ("#8D919A" if pct >= 30 else "#2E323A")
            st.markdown(f"**{sc['name']}** &nbsp; `{pct}%`", unsafe_allow_html=True)
            st.markdown(f'<div class="fc-est">{sc["estimate"]}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="fc-bar-track"><div class="fc-bar-fill" '
                f'style="width:{pct}%;background:{color}"></div></div>',
                unsafe_allow_html=True,
            )
            st.write(sc["narrative"])
            with st.expander("Tradeoffs, risks, and what would change this"):
                if sc.get("tradeoffs"):
                    st.markdown("**Tradeoffs**")
                    for t in sc["tradeoffs"]:
                        st.markdown(f"- {t}")
                if sc.get("risks"):
                    st.markdown("**Risks**")
                    for r in sc["risks"]:
                        st.markdown(f"- {r}")
                st.markdown(f"**What would change this:** {sc['pivot_condition']}")
                st.markdown(f"**Estimate based on:** {sc['estimate_basis']}")
            st.write("")

    if st.session_state.trace:
        with st.expander("Analysis notes (specialists → challenge → synthesis)"):
            st.text(st.session_state.trace)

    if result.get("sources"):
        st.caption("Sources referenced: " + " · ".join(result["sources"]))

    # ------------------------------ dig deeper -----------------------------

    if result["_mode"] == "decision":
        if st.button("⤵ Dig deeper (one more targeted research pass)"):
            try:
                with st.status("Finding the weak points…", expanded=True) as status:
                    draft = json.dumps({
                        "recommendation": result["recommendation"],
                        "scenarios": result["scenarios"],
                    })
                    gap_notes, _ = research(
                        f'Here is a draft decision analysis for: "{st.session_state.last_input}". '
                        f"Draft so far: {draft}. Identify what's still uncertain, thinly supported, or missing "
                        f"from this draft, then research specifically to address those weak points with concrete, "
                        f"current data. Visit full pages where a snippet isn't enough. Summarize what you find, "
                        f"with sources."
                    )
                    gap_preview = (gap_notes[:240] + "…") if len(gap_notes) > 240 else gap_notes
                    st.write(gap_preview if gap_preview else "No further gaps surfaced new data.")
                    status.update(label="Revising the analysis…")
                    revised, trace = reason(
                        "You are the same decision-simulation analyst revising your prior analysis now that more "
                        "targeted research is in. Update the scenarios and recommendation to incorporate this new "
                        "information — keep what still holds, revise what the new research contradicts or sharpens. "
                        "Keep the same rigor (specialist reasoning, a devil's advocate check) and the same output "
                        f"shape. The user's stated risk tolerance is {risk}.",
                        f"Decision: {st.session_state.last_input}\n\nPrior draft:\n{draft}\n\n"
                        f"New targeted research:\n{gap_notes}\n\nProduce the revised, final analysis now.",
                        DECISION_SCHEMA, 5000, "high",
                    )
                    st.write("Re-weighed the analysis against the new evidence and revised where it held or broke.")
                    status.update(label="Done", state="complete")
                revised["_mode"] = "decision"
                revised["sources"] = list(dict.fromkeys(result.get("sources", []) + revised.get("sources", [])))
                st.session_state.result = revised
                st.session_state.trace = trace
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    # ------------------------------ follow-ups -----------------------------

    st.markdown("##### Ask a follow-up")

    def ask(question: str):
        try:
            context = json.dumps(
                {k: v for k, v in result.items() if k not in ("_mode", "sources")}, indent=2
            )
            prior = "\n\n".join(f"Q: {f['q']}\nA: {f['a']}" for f in st.session_state.followups)
            data = call_groq(
                MODEL_REASONING,
                [
                    {"role": "system", "content":
                        "You are the same analyst, answering a follow-up about the analysis below. Be direct and "
                        "specific, referencing scenarios by name where relevant. A few sentences, plain text."},
                    {"role": "user", "content":
                        f"Original: {st.session_state.last_input}\n\nFinal analysis:\n{context}"
                        + (f"\n\nEarlier follow-ups:\n{prior}" if prior else "")
                        + f"\n\nNew question: {question}"},
                ],
                max_tokens=700,
                reasoning_format="hidden",
            )
            answer = data["choices"][0]["message"]["content"]
        except Exception as exc:
            answer = f"Error: {exc}"
        st.session_state.followups.append({"q": question, "a": answer})

    if result["_mode"] == "decision":
        cols = st.columns(3)
        for col, chip in zip(cols, ["Half the budget", "Double the timeline", "Half the team size"]):
            if col.button(chip):
                with st.spinner("Thinking…"):
                    ask(f"What if {chip.lower()}?")
                st.rerun()

    with st.form("followup", clear_on_submit=True):
        q = st.text_input(
            "Your question",
            placeholder="e.g. What if we only have 2 engineers, not 5?",
            label_visibility="collapsed",
        )
        if st.form_submit_button("Ask") and q.strip():
            with st.spinner("Thinking…"):
                ask(q.strip())
            st.rerun()

    for f in st.session_state.followups:
        st.caption(f["q"])
        st.write(f["a"])

st.divider()
st.caption(f"`{MODEL_RESEARCH}` for live research · `{MODEL_REASONING}` for reasoning")

"""Agentic Research Workflow: Planner -> Researcher -> Writer -> Editor -> Revision.
Run:  pip install streamlit openai requests
      export LLM_API_KEY=...   (optional: TAVILY_API_KEY)
      streamlit run app.py
"""
import json, os, re, time, xml.etree.ElementTree as ET
import requests, streamlit as st
from openai import (OpenAI, AuthenticationError, NotFoundError, RateLimitError,
                    APIConnectionError, APITimeoutError, InternalServerError)

NL = chr(10)
ERRORS = []
HEADERS = {"User-Agent": "AgenticResearchLab/1.0 (https://github.com/arjha10-max/Agentic-Research-Lab)"}

st.set_page_config(page_title="Agentic Research Lab", page_icon=":material/psychology:", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem}
.agent{border-radius:14px;padding:14px 18px;margin-bottom:10px;color:#fff;font-weight:600}
.a1{background:linear-gradient(135deg,#6366f1,#8b5cf6)}
.a2{background:linear-gradient(135deg,#0ea5e9,#06b6d4)}
.a3{background:linear-gradient(135deg,#f59e0b,#ef4444)}
</style>""", unsafe_allow_html=True)

# ---------- LLM helper ----------
def llm(client, system, prompt, max_tokens=2000):
    """Chat call with retries for transient errors (free models are often busy)."""
    last = None
    for attempt in range(3):
        try:
            r = client.chat.completions.create(model=MODEL, max_tokens=max_tokens, messages=[
                {"role": "system", "content": system}, {"role": "user", "content": prompt}])
            text = r.choices[0].message.content
            if text:
                return text
            last = RuntimeError("The model returned an empty reply")
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
            last = e
        time.sleep(3 * (attempt + 1))
    raise last

# ---------- Research tools ----------
# Each tool returns a list of sources. On any failure it records the reason in ERRORS
# and returns [] so one flaky API never breaks the pipeline or pollutes the report.
def wikipedia(q):
    try:
        resp = requests.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "generator": "search", "gsrsearch": q, "gsrlimit": 2,
            "prop": "extracts", "exintro": 1, "explaintext": 1, "format": "json"},
            headers=HEADERS, timeout=15)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {}).values()
        return [{"source": "Wikipedia", "title": p["title"], "text": p.get("extract", "")[:1200]}
                for p in pages if p.get("extract")]
    except Exception as e:
        ERRORS.append(f"Wikipedia ({type(e).__name__})")
        return []

def arxiv(q):
    try:
        resp = requests.get("http://export.arxiv.org/api/query", params={
            "search_query": f"all:{q}", "max_results": 2}, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        return [{"source": "arXiv", "title": " ".join(e.find("a:title", ns).text.split()),
                 "text": " ".join(e.find("a:summary", ns).text.split())[:1000]}
                for e in ET.fromstring(resp.text).findall("a:entry", ns)]
    except Exception as e:
        ERRORS.append(f"arXiv ({type(e).__name__})")
        return []

def tavily(q, key):
    if not key:
        return []
    try:
        resp = requests.post("https://api.tavily.com/search", json={
            "api_key": key, "query": q, "max_results": 2}, timeout=20)
        resp.raise_for_status()
        return [{"source": "Tavily", "title": x["title"], "text": x["content"][:1000]}
                for x in resp.json().get("results", [])]
    except Exception as e:
        ERRORS.append(f"Tavily ({type(e).__name__})")
        return []

# ---------- Sidebar ----------
with st.sidebar:
    st.title("Settings")
    st.selectbox("Provider", ["OpenRouter (free models)"])
    BASE_URL = "https://openrouter.ai/api/v1"
    MODEL = st.text_input("Model (must end in :free)", "nvidia/nemotron-3-super-120b-a12b:free")
    akey = st.text_input("API key", os.getenv("LLM_API_KEY", ""), type="password").strip()
    tkey = st.text_input("Tavily API key (optional)", os.getenv("TAVILY_API_KEY", ""), type="password")
    st.markdown("---")
    st.markdown(NL.join(["**Pipeline**", "1. Planner", "2. Researcher", "3. Writer", "4. Editor", "5. Revision"]))

st.title("Agentic Research Lab")
st.caption("Planning > external tools (Arxiv, Tavily, Wikipedia) > writing > editorial feedback")
topic = st.text_input("Research topic", "How do multi-agent LLM systems improve reasoning?")
go = st.button("Run workflow", type="primary", disabled=not (akey and topic))

if go:
    try:
        client = OpenAI(api_key=akey, base_url=BASE_URL)
        t0, S = time.time(), st.session_state
        prog = st.progress(0, "Starting...")

        with st.status("Planning Agent: building outline...", expanded=False) as s:
            raw = llm(client, "You are a planning agent. Reply with JSON only.",
                      f'Topic: {topic}{NL}Return {{"outline":[section titles x4],"queries":[3 short search queries]}}')
            try:
                plan = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
                assert plan["outline"] and plan["queries"]
            except Exception:
                plan = {"outline": ["Overview", "Key developments", "Challenges", "Outlook"],
                        "queries": [topic, topic + " overview", topic + " trends"]}
            s.update(label="Planning Agent: done", state="complete")
        prog.progress(25, "Researching...")

        with st.status("Research Agent: calling tools...", expanded=False) as s:
            notes = []
            for q in plan["queries"]:
                notes += wikipedia(q) + arxiv(q) + tavily(q, tkey)
            s.update(label=f"Research Agent: {len(notes)} sources gathered", state="complete")
        prog.progress(50, "Writing...")
        ctx = (NL * 2).join(f"[{n['source']}] {n['title']}: {n['text']}" for n in notes)

        with st.status("Writer: drafting...", expanded=False) as s:
            draft = llm(client, "You write concise, well-structured research reports in markdown. Cite sources by name.",
                        f"Topic: {topic}{NL}Outline: {plan['outline']}{NL}{NL}Sources:{NL}{ctx}{NL}{NL}Write a ~500 word report.", 2500)
            s.update(label="Writer: draft complete", state="complete")
        prog.progress(75, "Editing...")

        with st.status("Editor Agent: reflecting...", expanded=False) as s:
            feedback = llm(client, "You are a critical editor. Give 4-5 specific, actionable suggestions as a bullet list.",
                           f"Review this report:{NL}{NL}{draft}")
            s.update(label="Editor Agent: feedback ready", state="complete")

        with st.status("Revising with feedback...", expanded=False) as s:
            final = llm(client, "Revise the report applying every suggestion. Output markdown only.",
                        f"Report:{NL}{draft}{NL}{NL}Suggestions:{NL}{feedback}", 2500)
            s.update(label="Revision complete", state="complete")
        prog.progress(100, "Done")
        S.result = dict(errors=sorted(set(ERRORS)), plan=plan, notes=notes, draft=draft, feedback=feedback, final=final,
                        secs=round(time.time() - t0, 1))
    except AuthenticationError:
        st.error("The API key was rejected (401). Create a new key at openrouter.ai/keys and paste the whole key with no spaces.")
    except NotFoundError:
        st.error("Model not found. Use a model name that ends in :free and is listed on openrouter.ai/models.")
    except RateLimitError:
        st.error("The free model is rate-limited or busy. Wait a minute and click Run workflow again, or try another :free model.")
    except Exception as e:
        st.error(f"Workflow failed: {type(e).__name__}: {e}")

if "result" in st.session_state:
    R = st.session_state.result
    c = st.columns(4)
    c[0].metric("Sections planned", len(R["plan"]["outline"]))
    c[1].metric("Sources gathered", len(R["notes"]))
    c[2].metric("Final words", len(R["final"].split()))
    c[3].metric("Runtime (s)", R["secs"])

    t = st.tabs(["Plan", "Research", "Draft", "Editor", "Final"])
    with t[0]:
        st.markdown('<div class="agent a1">Planning Agent</div>', unsafe_allow_html=True)
        st.write("**Outline**")
        for o in R["plan"]["outline"]:
            st.markdown(f"- {o}")
        st.write("**Search queries**")
        for q in R["plan"]["queries"]:
            st.code(q)
    with t[1]:
        st.markdown('<div class="agent a2">Research Agent</div>', unsafe_allow_html=True)
        if R.get("errors"):
            st.caption("Some tools were unavailable this run: " + ", ".join(R["errors"]))
        for n in R["notes"]:
            with st.expander(f"{n['source']} - {n['title']}"):
                st.write(n["text"])
    with t[2]:
        st.markdown(R["draft"])
    with t[3]:
        st.markdown('<div class="agent a3">Editor Agent</div>', unsafe_allow_html=True)
        st.markdown(R["feedback"])
    with t[4]:
        st.markdown(R["final"])
        st.download_button("Download report (.md)", R["final"], "report.md")

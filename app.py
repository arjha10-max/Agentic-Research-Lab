"""Agentic Research Workflow: Planner -> Researcher -> Writer -> Editor -> Revision.
Run:  pip install streamlit anthropic requests
      export ANTHROPIC_API_KEY=sk-...   (optional: TAVILY_API_KEY)
      streamlit run app.py
"""
import json, os, re, time, xml.etree.ElementTree as ET
import requests, streamlit as st
from anthropic import Anthropic

MODEL = "claude-sonnet-5-5"
st.set_page_config(page_title="Agentic Research Lab", page_icon="🧠", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem}
.agent{border-radius:14px;padding:14px 18px;margin-bottom:10px;color:#fff;font-weight:600}
.a1{background:linear-gradient(135deg,#6366f1,#8b5cf6)}
.a2{background:linear-gradient(135deg,#0ea5e9,#06b6d4)}
.a3{background:linear-gradient(135deg,#f59e0b,#ef4444)}
</style>""", unsafe_allow_html=True)

# ---------- LLM helper ----------
def llm(client, system, prompt, max_tokens=2000):
    r = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                               messages=[{"role": "user", "content": prompt}])
    return r.content[0].text

# ---------- Research tools ----------
def wikipedia(q):
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "generator": "search", "gsrsearch": q, "gsrlimit": 2,
            "prop": "extracts", "exintro": 1, "explaintext": 1, "format": "json"}, timeout=15).json()
        return [{"source": "Wikipedia", "title": p["title"], "text": p.get("extract", "")[:1200]}
                for p in r.get("query", {}).get("pages", {}).values()]
    except Exception as e:
        return [{"source": "Wikipedia", "title": "error", "text": str(e)}]

def arxiv(q):
    try:
        x = requests.get("http://export.arxiv.org/api/query", params={
            "search_query": f"all:{q}", "max_results": 2}, timeout=20).text
        ns = {"a": "http://www.w3.org/2005/Atom"}
        return [{"source": "arXiv", "title": " ".join(e.find("a:title", ns).text.split()),
                 "text": " ".join(e.find("a:summary", ns).text.split())[:1000]}
                for e in ET.fromstring(x).findall("a:entry", ns)]
    except Exception as e:
        return [{"source": "arXiv", "title": "error", "text": str(e)}]

def tavily(q, key):
    if not key:
        return []
    try:
        r = requests.post("https://api.tavily.com/search", json={
            "api_key": key, "query": q, "max_results": 2}, timeout=20).json()
        return [{"source": "Tavily", "title": x["title"], "text": x["content"][:1000]}
                for x in r.get("results", [])]
    except Exception as e:
        return [{"source": "Tavily", "title": "error", "text": str(e)}]

# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚙️ Settings")
    akey = st.text_input("Anthropic API key", os.getenv("ANTHROPIC_API_KEY", ""), type="password")
    tkey = st.text_input("Tavily API key (optional)", os.getenv("TAVILY_API_KEY", ""), type="password")
    st.markdown("---")
    st.markdown("**Pipeline**\n1. 🗂 Planner\n2. 🔎 Researcher\n3. ✍️ Writer\n4. 🧐 Editor\n5. ✅ Revision")

st.title("🧠 Agentic Research Lab")
st.caption("Planning → external tools (Arxiv, Tavily, Wikipedia) → writing → editorial feedback")
topic = st.text_input("Research topic", "How do multi-agent LLM systems improve reasoning?")
go = st.button("🚀 Run workflow", type="primary", disabled=not (akey and topic))

if go:
    client = Anthropic(api_key=akey)
    t0, S = time.time(), st.session_state
    prog = st.progress(0, "Starting…")

    with st.status("🗂 Planning Agent: building outline…", expanded=False) as s:
        raw = llm(client, "You are a planning agent. Reply with JSON only.",
                  f'Topic: {topic}\nReturn {{"outline":[section titles x4],"queries":[3 short search queries]}}')
        plan = json.loads(re.search(r"\{.*\}", raw, re.S).group())
        s.update(label="🗂 Planning Agent: done", state="complete")
    prog.progress(25, "Researching…")

    with st.status("🔎 Research Agent: calling tools…", expanded=False) as s:
        notes = []
        for q in plan["queries"]:
            notes += wikipedia(q) + arxiv(q) + tavily(q, tkey)
        s.update(label=f"🔎 Research Agent: {len(notes)} sources gathered", state="complete")
    prog.progress(50, "Writing…")
    ctx = "\n\n".join(f"[{n['source']}] {n['title']}: {n['text']}" for n in notes)

    with st.status("✍️ Writer: drafting…", expanded=False) as s:
        draft = llm(client, "You write concise, well-structured research reports in markdown. Cite sources by name.",
                    f"Topic: {topic}\nOutline: {plan['outline']}\n\nSources:\n{ctx}\n\nWrite a ~500 word report.", 2500)
        s.update(label="✍️ Writer: draft complete", state="complete")
    prog.progress(75, "Editing…")

    with st.status("🧐 Editor Agent: reflecting…", expanded=False) as s:
        feedback = llm(client, "You are a critical editor. Give 4-5 specific, actionable suggestions as a bullet list.",
                       f"Review this report:\n\n{draft}")
        s.update(label="🧐 Editor Agent: feedback ready", state="complete")

    with st.status("✅ Revising with feedback…", expanded=False) as s:
        final = llm(client, "Revise the report applying every suggestion. Output markdown only.",
                    f"Report:\n{draft}\n\nSuggestions:\n{feedback}", 2500)
        s.update(label="✅ Revision complete", state="complete")
    prog.progress(100, "Done")
    S.result = dict(plan=plan, notes=notes, draft=draft, feedback=feedback, final=final,
                    secs=round(time.time() - t0, 1))

if "result" in st.session_state:
    R = st.session_state.result
    c = st.columns(4)
    c[0].metric("Sections planned", len(R["plan"]["outline"]))
    c[1].metric("Sources gathered", len(R["notes"]))
    c[2].metric("Final words", len(R["final"].split()))
    c[3].metric("Runtime (s)", R["secs"])

    t = st.tabs(["🗂 Plan", "🔎 Research", "✍️ Draft", "🧐 Editor", "✅ Final"])
    with t[0]:
        st.markdown('<div class="agent a1">Planning Agent</div>', unsafe_allow_html=True)
        st.write("**Outline**"); [st.markdown(f"- {o}") for o in R["plan"]["outline"]]
        st.write("**Search queries**"); [st.code(q) for q in R["plan"]["queries"]]
    with t[1]:
        st.markdown('<div class="agent a2">Research Agent</div>', unsafe_allow_html=True)
        for n in R["notes"]:
            with st.expander(f"{n['source']} · {n['title']}"):
                st.write(n["text"])
    with t[2]:
        st.markdown(R["draft"])
    with t[3]:
        st.markdown('<div class="agent a3">Editor Agent</div>', unsafe_allow_html=True)
        st.markdown(R["feedback"])
    with t[4]:
        st.markdown(R["final"])
        st.download_button("⬇️ Download report (.md)", R["final"], "report.md")

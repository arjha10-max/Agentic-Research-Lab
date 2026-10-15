"""One-Page POV Agent: Planner -> Researcher -> Writer -> Editor -> Revision.
Type a topic, get a one-page point of view built from marketing, consumer and polling sources.
Run:  pip install streamlit openai requests ddgs
      export LLM_API_KEY=...   (recommended: TAVILY_API_KEY)
      streamlit run app.py
"""
import json, os, time
from urllib.parse import urlparse
import requests, streamlit as st
from openai import (OpenAI, AuthenticationError, NotFoundError, RateLimitError,
                    APIConnectionError, APITimeoutError, InternalServerError)

NL = chr(10)
HEADERS = {"User-Agent": "AgenticResearchLab/1.0 (https://github.com/arjha10-max/Agentic-Research-Lab)"}
ERRORS = []

# Sources the researcher prefers: market research, consulting, polling and Canadian business/government.
TRUSTED = [
    "statista.com", "mckinsey.com", "nielseniq.com", "nielsen.com", "deloitte.com", "pwc.com",
    "kpmg.com", "bcg.com", "bain.com", "ipsos.com", "kantar.com", "euromonitor.com", "mintel.com",
    "emarketer.com", "thinkwithgoogle.com", "pewresearch.org", "gallup.com", "hbr.org",
    "statcan.gc.ca", "canada.ca", "elections.ca", "leger360.com", "angusreid.org", "abacusdata.ca",
    "mainstreetresearch.ca", "strategyonline.ca", "marketingmag.ca", "canadiangrocer.com",
    "financialpost.com", "theglobeandmail.com", "cbc.ca", "thecanadianpressnews.ca",
]
TRUSTED_SHORT = ["statista.com", "mckinsey.com", "nielseniq.com", "deloitte.com", "ipsos.com",
                 "leger360.com", "angusreid.org", "statcan.gc.ca", "marketingmag.ca", "canadiangrocer.com"]

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
# Each tool returns a list of sources: {source, title, text, url}. On any failure it records the
# reason in ERRORS and returns [] so one flaky service never breaks the run.
def domain_of(url):
    return urlparse(url).netloc.replace("www.", "") or "web"


def wikipedia(q):
    try:
        resp = requests.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "generator": "search", "gsrsearch": q, "gsrlimit": 2,
            "prop": "extracts", "exintro": 1, "explaintext": 1, "format": "json"},
            headers=HEADERS, timeout=15)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {}).values()
        return [{"source": "Wikipedia", "title": p["title"], "text": p.get("extract", "")[:900],
                 "url": "https://en.wikipedia.org/wiki/" + p["title"].replace(" ", "_")}
                for p in pages if p.get("extract")]
    except Exception as e:
        ERRORS.append(f"Wikipedia ({type(e).__name__})")
        return []


def tavily_call(q, key, domains):
    body = {"query": q, "max_results": 4, "search_depth": "basic", "api_key": key}
    if domains:
        body["include_domains"] = domains
    resp = requests.post("https://api.tavily.com/search", json=body,
                         headers={"Authorization": "Bearer " + key}, timeout=25)
    resp.raise_for_status()
    return [{"source": domain_of(x["url"]), "title": x["title"], "text": x["content"][:900], "url": x["url"]}
            for x in resp.json().get("results", []) if x.get("content")]


def web_search_tavily(q, key):
    """Trusted marketing and research sites first; open web if those return too little."""
    try:
        rows = tavily_call(q, key, TRUSTED)
        if len(rows) < 2:
            rows += tavily_call(q, key, None)
        return rows
    except Exception as e:
        ERRORS.append(f"Tavily ({type(e).__name__})")
        return []


def web_search_ddg(q):
    """Keyless fallback using DuckDuckGo. Less reliable on shared cloud servers than Tavily."""
    try:
        from ddgs import DDGS
        sites = " OR ".join("site:" + d for d in TRUSTED_SHORT)
        with DDGS() as d:
            rows = list(d.text(q + " (" + sites + ")", region="ca-en", max_results=5))
            if len(rows) < 2:
                rows += list(d.text(q, region="ca-en", max_results=5))
        return [{"source": domain_of(r.get("href", "")), "title": r.get("title", ""),
                 "text": (r.get("body") or "")[:900], "url": r.get("href", "")}
                for r in rows if r.get("href") and r.get("body")]
    except Exception as e:
        ERRORS.append(f"Web search ({type(e).__name__})")
        return []


def cited_numbers(text):
    """Find source tags like [3] or [1, 4] in the report."""
    nums, i = set(), 0
    while True:
        i = text.find("[", i)
        if i < 0:
            break
        j = text.find("]", i)
        if j < 0:
            break
        for part in text[i + 1:j].split(","):
            if part.strip().isdigit():
                nums.add(int(part.strip()))
        i = j + 1
    return nums


# ---------- Sidebar ----------
with st.sidebar:
    st.title("Settings")
    st.selectbox("Provider", ["OpenRouter (free models)"])
    BASE_URL = "https://openrouter.ai/api/v1"
    MODEL = st.text_input("Model (must end in :free)", "nvidia/nemotron-3-super-120b-a12b:free")
    akey = st.text_input("API key", os.getenv("LLM_API_KEY", ""), type="password").strip()
    tkey = st.text_input("Tavily API key (recommended, free tier)", os.getenv("TAVILY_API_KEY", ""),
                         type="password").strip()
    st.caption("Tavily searches marketing, consulting, polling and Canadian business sources. "
               "Without a key the app falls back to a keyless web search, which is less reliable.")
    st.markdown("---")
    st.markdown(NL.join(["**Pipeline**", "1. Planner", "2. Researcher", "3. Writer", "4. Editor", "5. Revision"]))

st.title("Agentic Research Lab")
st.caption("Type a topic. Get a one-page point of view built from marketing, consumer and polling sources.")
topic = st.text_input("Topic", "consumer acceptance of zero sugar drinks in Canada")
st.caption("Examples: elections in Canada | how Lunar New Year is celebrated in Canada | EV adoption in Canada")
go = st.button("Build one-pager", type="primary", disabled=not (akey and topic))

if go:
    try:
        ERRORS.clear()
        client = OpenAI(api_key=akey, base_url=BASE_URL)
        t0, S = time.time(), st.session_state
        prog = st.progress(0, "Starting...")

        with st.status("Planning Agent: framing the question...", expanded=False) as s:
            raw = llm(client, "You are a planning agent for a consumer-insights team. Reply with JSON only.",
                      f"""Topic: {topic}
Plan a one-page point of view (POV) on this topic.
Return a JSON object with two keys.
"angle": one sentence stating the question the POV must answer.
"queries": a list of 4 short web search queries (max 8 words each) aimed at market data,
consumer attitudes and surveys, the Canadian context, and recent trends.
Keep any place named in the topic inside every query.""")
            try:
                plan = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
                assert plan["angle"] and len(plan["queries"]) >= 2
            except Exception:
                plan = {"angle": f"What is the current state of {topic}, and what should decision makers do about it?",
                        "queries": [topic, topic + " survey consumer attitudes", topic + " market trends",
                                    topic + " statistics report"]}
            plan["queries"] = plan["queries"][:4]
            s.update(label="Planning Agent: done", state="complete")
        prog.progress(20, "Researching...")

        with st.status("Research Agent: searching trusted sources...", expanded=False) as s:
            found = []
            for q in plan["queries"]:
                found += web_search_tavily(q, tkey) if tkey else web_search_ddg(q)
            found += wikipedia(topic)
            seen, notes = set(), []
            for n in found:
                if n["url"] and n["url"] not in seen:
                    seen.add(n["url"])
                    notes.append(n)
            notes = notes[:14]
            s.update(label=f"Research Agent: {len(notes)} sources gathered", state="complete")
        if not notes:
            raise RuntimeError("No sources were found. Add a free Tavily key in the sidebar and run again.")
        prog.progress(45, "Writing...")
        ctx = (NL * 2).join(f"[{i}] {n['title']} ({n['source']}): {n['text']}" for i, n in enumerate(notes, 1))

        with st.status("Writer: drafting the one-pager...", expanded=False) as s:
            draft = llm(client, "You write one-page point-of-view briefs for marketing and strategy leaders. "
                                "Plain, confident and specific.",
                        f"""Topic: {topic}
Question to answer: {plan['angle']}

Sources (cite them as [n]):
{ctx}

Write a one-page POV of 350 to 450 words in markdown with exactly these parts:
## The POV
2 to 3 sentences that take a clear position, not a summary.
## What the evidence says
4 to 6 bullets. Each bullet holds one specific fact or number and ends with its source tag, like [2].
## Why it matters
3 bullets on what this means for brands, marketers or decision makers.
## Watch-outs
2 bullets on gaps, risks or things to verify.

Rules: use only facts that appear in the sources. Never invent numbers, dates or quotes.
If the sources lack data on something, say so under Watch-outs. Do not write a sources list.""", 2500)
            s.update(label="Writer: draft complete", state="complete")
        prog.progress(70, "Editing...")

        with st.status("Editor Agent: reviewing...", expanded=False) as s:
            feedback = llm(client, "You are a critical editor of strategy briefs. Give 4-5 specific, actionable "
                                   "suggestions as a bullet list.",
                           f"""Review this one-pager against its sources.
Check: is every number or claim supported by a source, is the POV a clear position, is the
'so what' useful to a marketer, and is it under 450 words.

One-pager:
{draft}

Sources:
{ctx}""")
            s.update(label="Editor Agent: feedback ready", state="complete")

        with st.status("Revising with feedback...", expanded=False) as s:
            final = llm(client, "Revise the one-pager applying every suggestion. Output markdown only.",
                        f"""Keep the same four parts, keep it between 350 and 450 words, keep the [n] source tags,
and remove any claim the sources do not support. Do not add a sources list.

One-pager:
{draft}

Suggestions:
{feedback}""", 2500)
            s.update(label="Revision complete", state="complete")
        prog.progress(100, "Done")
        S.result = dict(errors=sorted(set(ERRORS)), plan=plan, notes=notes, draft=draft,
                        feedback=feedback, final=final, secs=round(time.time() - t0, 1))
    except AuthenticationError:
        st.error("The API key was rejected (401). Create a new key at openrouter.ai/keys and paste the whole key with no spaces.")
    except NotFoundError:
        st.error("Model not found. Use a model name that ends in :free and is listed on openrouter.ai/models.")
    except RateLimitError:
        st.error("The free model is rate-limited or busy. Wait a minute and click Build one-pager again, or try another :free model.")
    except Exception as e:
        st.error(f"Workflow failed: {type(e).__name__}: {e}")

if "result" in st.session_state:
    R = st.session_state.result
    used = sorted(n for n in cited_numbers(R["final"]) if 1 <= n <= len(R["notes"]))
    sources_md = NL.join(f"{n}. [{R['notes'][n - 1]['title']}]({R['notes'][n - 1]['url']}) - {R['notes'][n - 1]['source']}"
                         for n in (used or range(1, len(R["notes"]) + 1)))
    full = R["final"] + NL * 2 + "### Sources" + NL + sources_md

    c = st.columns(4)
    c[0].metric("Sources gathered", len(R["notes"]))
    c[1].metric("Sources cited", len(used))
    c[2].metric("One-pager words", len(R["final"].split()))
    c[3].metric("Runtime (s)", R["secs"])

    t = st.tabs(["Plan", "Research", "Draft", "Editor", "One-pager"])
    with t[0]:
        st.markdown('<div class="agent a1">Planning Agent</div>', unsafe_allow_html=True)
        st.write("**Question the POV answers**")
        st.write(R["plan"]["angle"])
        st.write("**Search queries**")
        for q in R["plan"]["queries"]:
            st.code(q)
    with t[1]:
        st.markdown('<div class="agent a2">Research Agent</div>', unsafe_allow_html=True)
        if R.get("errors"):
            st.caption("Some tools were unavailable this run: " + ", ".join(R["errors"]))
        for i, n in enumerate(R["notes"], 1):
            with st.expander(f"[{i}] {n['source']} - {n['title']}"):
                st.write(n["text"])
                st.markdown(f"[Open source]({n['url']})")
    with t[2]:
        st.markdown(R["draft"])
    with t[3]:
        st.markdown('<div class="agent a3">Editor Agent</div>', unsafe_allow_html=True)
        st.markdown(R["feedback"])
    with t[4]:
        st.markdown(full)
        st.download_button("Download one-pager (.md)", full, "one_pager.md")
